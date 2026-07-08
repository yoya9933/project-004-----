param(
    [string]$InputCsv,
    [string]$OutputCsv,
    [string]$SummaryDir
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
if (-not $InputCsv) {
    $InputCsv = Join-Path $ProjectRoot "06_交付物\ai_rag_annotation\annotation_workbook.csv"
}
if (-not $OutputCsv) {
    $OutputCsv = $InputCsv
}
if (-not $SummaryDir) {
    $SummaryDir = Split-Path -Parent $OutputCsv
}

function Get-Field {
    param(
        [object]$Row,
        [string]$Name
    )

    $property = $Row.PSObject.Properties[$Name]
    if ($null -eq $property -or $null -eq $property.Value) {
        return ""
    }
    return [string]$property.Value
}

function Normalize-CandidateText {
    param([string]$Text)

    if ($null -eq $Text) {
        return ""
    }
    $normalized = $Text -replace "(?<=[0-9])\s+(?=[0-9])", ""
    return (($normalized -replace "\s+", " ").Trim())
}

function Convert-Number {
    param([string]$Text)

    if ([string]::IsNullOrWhiteSpace($Text)) {
        return $null
    }
    $clean = $Text.Replace(",", "").Replace("，", "").Trim()
    $number = 0.0
    if ([double]::TryParse($clean, [ref]$number)) {
        return $number
    }
    return $null
}

function Get-AmountsFromText {
    param([string]$Text)

    $text = Normalize-CandidateText $Text
    $amounts = New-Object System.Collections.Generic.List[double]

    $wanPattern = [regex]"([0-9][0-9,，]*(?:\.[0-9]+)?)\s*萬\s*([0-9][0-9,，]*)?\s*(?:元|圓)?"
    foreach ($match in $wanPattern.Matches($text)) {
        $main = Convert-Number $match.Groups[1].Value
        $tail = Convert-Number $match.Groups[2].Value
        if ($null -ne $main) {
            $value = [double]$main * 10000.0
            if ($null -ne $tail) {
                $value += [double]$tail
            }
            [void]$amounts.Add($value)
        }
    }

    $yuanPattern = [regex]"([0-9][0-9,，]{2,})\s*(?:元|圓)"
    foreach ($match in $yuanPattern.Matches($text)) {
        $value = Convert-Number $match.Groups[1].Value
        if ($null -ne $value) {
            [void]$amounts.Add([double]$value)
        }
    }

    return @($amounts | Sort-Object -Unique)
}

function Get-CandidateItems {
    param([string]$CandidateText)

    $items = New-Object System.Collections.Generic.List[object]
    if ([string]::IsNullOrWhiteSpace($CandidateText)) {
        return @()
    }

    $parts = $CandidateText -split "\s+;\s+"
    foreach ($part in $parts) {
        if ([string]::IsNullOrWhiteSpace($part)) {
            continue
        }
        $prefix = ""
        $context = $part
        if ($part.Contains("｜")) {
            $bits = $part.Split([string[]]@("｜"), 2, [System.StringSplitOptions]::None)
            $prefix = $bits[0]
            $context = $bits[1]
        }

        $amounts = @(Get-AmountsFromText $context)
        if ($amounts.Count -eq 0) {
            $prefixNumber = Convert-Number $prefix
            if ($null -ne $prefixNumber) {
                $amounts = @([double]$prefixNumber)
            }
        }
        foreach ($amount in $amounts) {
            [void]$items.Add([pscustomobject]@{
                Amount = [double]$amount
                Prefix = $prefix
                Context = (Normalize-CandidateText $context)
            })
        }
    }
    return @($items | ForEach-Object { $_ })
}

function Add-Score {
    param(
        [string]$Text,
        [string[]]$Patterns,
        [double]$Weight
    )

    foreach ($pattern in $Patterns) {
        if ($Text -match $pattern) {
            return $Weight
        }
    }
    return 0.0
}

function Select-AmountCandidate {
    param(
        [string]$CandidateText,
        [string]$Kind,
        [Nullable[double]]$ClaimedPenalty
    )

    $items = @(Get-CandidateItems $CandidateText)
    if ($items.Count -eq 0) {
        return $null
    }

    $scored = New-Object System.Collections.Generic.List[object]
    foreach ($item in $items) {
        $context = [string]$item.Context
        $score = 0.0
        if ($Kind -eq "contract") {
            $score += Add-Score $context @("契約總價", "契約價金", "契約金額", "契約總價金", "工程總價", "總工程款", "變更後.*工程款") 15
            $score += Add-Score $context @("總價", "價金") 5
            $score -= Add-Score $context @("違約金", "罰款", "利息", "聲明", "管理費", "保險費", "履約保證金", "工程款短少") 8
        }
        elseif ($Kind -eq "claimed") {
            $score += Add-Score $context @("逾期違約金", "違約金", "扣罰", "罰款", "沒收") 15
            $score += Add-Score $context @("主張", "請求", "要求", "不當扣罰", "溢扣") 6
            $score -= Add-Score $context @("契約總價", "契約價金", "總工程款", "工程款短少", "管理費", "保險費", "利息", "聲明") 8
        }
        elseif ($Kind -eq "allowed") {
            $score += Add-Score $context @("酌減為", "核減為", "減為", "准許", "得請求", "應給付", "應返還", "有理由", "無理由", "主 文", "判決如下") 14
            $score += Add-Score $context @("逾期違約金", "違約金", "扣罰", "罰款", "返還") 5
            $score -= Add-Score $context @("聲明", "原告主張", "被告答辯", "契約總價", "總工程款", "工程款短少", "管理費", "保險費", "利息") 8
            if ($null -ne $ClaimedPenalty) {
                if ([double]$item.Amount -le [double]$ClaimedPenalty) {
                    $score += 8
                }
                else {
                    $score -= 12
                }
            }
        }

        [void]$scored.Add([pscustomobject]@{
            Amount = [double]$item.Amount
            Score = [double]$score
            Context = $context
        })
    }

    $selected = $scored |
        Sort-Object @{ Expression = "Score"; Descending = $true }, @{ Expression = "Amount"; Descending = $true } |
        Select-Object -First 1
    return $selected
}

function Select-DayCandidate {
    param([string]$CandidateText)

    if ([string]::IsNullOrWhiteSpace($CandidateText)) {
        return $null
    }
    $items = New-Object System.Collections.Generic.List[object]
    $parts = $CandidateText -split "\s+;\s+"
    foreach ($part in $parts) {
        if ([string]::IsNullOrWhiteSpace($part)) {
            continue
        }
        $prefix = ""
        $context = $part
        if ($part.Contains("｜")) {
            $bits = $part.Split([string[]]@("｜"), 2, [System.StringSplitOptions]::None)
            $prefix = $bits[0]
            $context = $bits[1]
        }
        $value = Convert-Number $prefix
        if ($null -eq $value -or $value -le 0 -or $value -gt 5000) {
            continue
        }
        $context = Normalize-CandidateText $context
        $score = 0.0
        $score += Add-Score $context @("逾期.*?(?:日|天)", "遲延.*?(?:日|天)", "逾期違約金") 12
        $score += Add-Score $context @("工期", "展延", "展期", "完工期限") 6
        $score -= Add-Score $context @("年[0-9]*月", "月[0-9]*日", "第[0-9]+條") 5
        if ($value -le 3) {
            $score -= 2
        }
        [void]$items.Add([pscustomobject]@{
            Days = [double]$value
            Score = [double]$score
            Context = $context
        })
    }
    if ($items.Count -eq 0) {
        return $null
    }

    return ($items |
        Sort-Object @{ Expression = "Score"; Descending = $true }, @{ Expression = "Days"; Descending = $true } |
        Select-Object -First 1)
}

function Get-Flag {
    param([string]$Value)

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return 0
    }
    $text = $Value.Trim().ToLowerInvariant()
    if ($text -in @("1", "true", "yes", "y", "是", "有")) {
        return 1
    }
    return 0
}

function Compact-Reason {
    param([string]$Text)

    $text = Normalize-CandidateText $Text
    if ([string]::IsNullOrWhiteSpace($text)) {
        return ""
    }
    if ($text.Length -le 360) {
        return $text
    }
    return $text.Substring(0, 359) + "…"
}

function Format-Number {
    param([Nullable[double]]$Value)

    if ($null -eq $Value) {
        return ""
    }
    return ([Math]::Round([double]$Value, 6)).ToString("0.######", [System.Globalization.CultureInfo]::InvariantCulture)
}

$rows = Import-Csv -Encoding UTF8 -LiteralPath $InputCsv
New-Item -ItemType Directory -Force -Path $SummaryDir | Out-Null

if ((Resolve-Path -LiteralPath $InputCsv).Path -eq (Resolve-Path -LiteralPath (Split-Path -Parent $OutputCsv)).Path) {
    throw "OutputCsv resolves to a directory. Please provide a file path."
}

if ((Test-Path -LiteralPath $OutputCsv) -and ((Resolve-Path -LiteralPath $InputCsv).Path -eq (Resolve-Path -LiteralPath $OutputCsv).Path)) {
    $backupName = "annotation_workbook.before_ai_assumed_review_$(Get-Date -Format 'yyyyMMdd_HHmmss').csv"
    $backupPath = Join-Path (Split-Path -Parent $InputCsv) $backupName
    Copy-Item -LiteralPath $InputCsv -Destination $backupPath
}
else {
    $backupPath = ""
}

$updated = New-Object System.Collections.Generic.List[object]
$qualityRows = New-Object System.Collections.Generic.List[object]
$today = Get-Date -Format "yyyy-MM-dd"

foreach ($row in $rows) {
    $contract = Select-AmountCandidate (Get-Field $row "ai_contract_price_candidates") "contract" $null
    $claimed = Select-AmountCandidate (Get-Field $row "ai_claimed_penalty_candidates") "claimed" $null
    $claimedValue = if ($null -eq $claimed) { $null } else { [double]$claimed.Amount }
    $allowed = Select-AmountCandidate (Get-Field $row "ai_allowed_penalty_candidates") "allowed" $claimedValue
    $allowedValue = if ($null -eq $allowed) { $null } else { [double]$allowed.Amount }
    $delay = Select-DayCandidate (Get-Field $row "ai_delay_days_candidates")

    $remainingRatio = $null
    $reductionRate = $null
    if ($null -ne $claimedValue -and $null -ne $allowedValue -and $claimedValue -gt 0 -and $allowedValue -ge 0 -and $allowedValue -le $claimedValue) {
        $remainingRatio = $allowedValue / $claimedValue
        $reductionRate = 1.0 - $remainingRatio
    }

    $isReduced = ""
    if ($null -ne $claimedValue -and $null -ne $allowedValue) {
        $isReduced = [int]($allowedValue -lt $claimedValue)
    }
    elseif ((Get-Field $row "matched_strong_reduction_terms") -match "酌減|過高|252" -or (Get-Field $row "source_snippet_reduction") -match "酌減|過高|民法第\s*252") {
        $isReduced = 1
    }
    else {
        $isReduced = 0
    }

    $row.annotation_status = "ai_assumed_review"
    $row.manual_checked = "ai_assumed"
    $row.contract_price = if ($null -eq $contract) { "" } else { Format-Number ([double]$contract.Amount) }
    $row.delay_days = if ($null -eq $delay) { "" } else { Format-Number ([double]$delay.Days) }
    $row.claimed_penalty = Format-Number $claimedValue
    $row.allowed_penalty = Format-Number $allowedValue
    $row.is_reduced = [string]$isReduced
    $row.remaining_ratio = Format-Number $remainingRatio
    $row.reduction_rate = Format-Number $reductionRate
    $row.issue_owner_fault = [string](Get-Flag (Get-Field $row "ai_suggest_issue_owner_fault"))
    $row.issue_contractor_fault = [string](Get-Flag (Get-Field $row "ai_suggest_issue_contractor_fault"))
    $row.issue_extension_request = [string](Get-Flag (Get-Field $row "ai_suggest_issue_extension_request"))
    $row.issue_actual_damage_unclear = [string](Get-Flag (Get-Field $row "ai_suggest_issue_actual_damage_unclear"))
    $row.issue_partial_completion = [string](Get-Flag (Get-Field $row "ai_suggest_issue_partial_completion"))
    $row.issue_used_by_owner = [string](Get-Flag (Get-Field $row "ai_suggest_issue_used_by_owner"))
    $row.key_reason = Compact-Reason (Get-Field $row "source_snippet_reduction")

    $note = "AI-assumed review ${today}: official fields populated from ai_* candidates/suggestions without independent full-text manual verification."
    $oldNote = Get-Field $row "manual_notes"
    if ([string]::IsNullOrWhiteSpace($oldNote)) {
        $row.manual_notes = $note
    }
    elseif ($oldNote -notmatch "AI-assumed review") {
        $row.manual_notes = "$oldNote | $note"
    }

    [void]$qualityRows.Add([pscustomobject]@{
        JID = Get-Field $row "JID"
        annotation_priority = Get-Field $row "annotation_priority"
        contract_price_selected = Get-Field $row "contract_price"
        contract_score = if ($null -eq $contract) { "" } else { $contract.Score }
        delay_days_selected = Get-Field $row "delay_days"
        delay_score = if ($null -eq $delay) { "" } else { $delay.Score }
        claimed_penalty_selected = Get-Field $row "claimed_penalty"
        claimed_score = if ($null -eq $claimed) { "" } else { $claimed.Score }
        allowed_penalty_selected = Get-Field $row "allowed_penalty"
        allowed_score = if ($null -eq $allowed) { "" } else { $allowed.Score }
        is_reduced = Get-Field $row "is_reduced"
        ratio_valid = [int]($null -ne $remainingRatio)
    })
    [void]$updated.Add($row)
}

$fieldNames = @($rows[0].PSObject.Properties.Name)
$updated | Select-Object $fieldNames | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $OutputCsv

$qualityPath = Join-Path $SummaryDir "ai_assumed_review_quality.csv"
$qualityRows | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $qualityPath

$ratioValid = @($qualityRows | Where-Object { $_.ratio_valid -eq 1 }).Count
$reducedCount = @($updated | Where-Object { [string]$_.is_reduced -eq "1" }).Count
$notReducedCount = @($updated | Where-Object { [string]$_.is_reduced -eq "0" }).Count
$backupLeaf = if ([string]::IsNullOrWhiteSpace($backupPath)) { "未覆寫原檔，未建立備份" } else { Split-Path -Leaf $backupPath }

$summaryPath = Join-Path $SummaryDir "人工智慧假設版人工審核摘要.md"
$summary = @(
    "# AI 假設版人工審核摘要",
    "",
    "## 說明",
    "",
    '依使用者指示，本次先假設 AI 判斷為正確，將 `ai_*` 候選與建議轉入正式欄位，用於快速推進模型回測。',
    "",
    '這不是逐案回到原判決全文的嚴格人工查核；`manual_checked` 以 `ai_assumed` 標示，`manual_notes` 也保留來源註記。',
    "",
    "## 筆數",
    "",
    "- 更新列數：$($updated.Count)",
    ('- `is_reduced=1`：' + $reducedCount),
    ('- `is_reduced=0`：' + $notReducedCount),
    ('- 可計算 `remaining_ratio` 的列數：' + $ratioValid),
    "",
    "## 輸出",
    "",
    ('- 更新後標註表：`' + (Split-Path -Leaf $OutputCsv) + '`'),
    ('- 原始備份：`' + $backupLeaf + '`'),
    '- 選值品質表：`ai_assumed_review_quality.csv`',
    "",
    "## 重要限制",
    "",
    "- 金額欄位是從 AI 候選與規則選值推定，可能把工程款、利息、管理費或聲明總額誤認為違約金。",
    '- `allowed_penalty` 若高於 `claimed_penalty`，比例欄位會留空，避免產生不合理比例。',
    "- 正式報告若使用本版結果，應表述為 AI 假設回測或敏感度分析，不應稱為完成嚴格人工標註。"
)
$summary -join "`n" | Set-Content -Encoding UTF8 -LiteralPath $summaryPath

Write-Host "Updated rows: $($updated.Count)"
Write-Host "Reduced labels: $reducedCount"
Write-Host "Not reduced labels: $notReducedCount"
Write-Host "Valid remaining_ratio rows: $ratioValid"
Write-Host "OutputCsv: $OutputCsv"
Write-Host "Backup: $backupPath"
Write-Host "Summary: $summaryPath"
