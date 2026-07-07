param(
    [string]$IndexCsv,
    [string]$JsonRoot,
    [string]$OutputDir,
    [int]$SampleSize = 120
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$DatabaseRoot = Join-Path $ProjectRoot "02_輸入資料\法律課程資料庫-20260707T080337Z-3-001\法律課程資料庫"
if (-not $IndexCsv) {
    $IndexCsv = Join-Path $DatabaseRoot "2021-2026判決總表.csv"
}
if (-not $JsonRoot) {
    $JsonRoot = Join-Path $DatabaseRoot "依年份分類"
}
if (-not $OutputDir) {
    $OutputDir = Join-Path $ProjectRoot "06_交付物\keyword_screening"
}

$PenaltyPatterns = @(
    [pscustomobject]@{ Label = "違約金"; Pattern = "違約金" },
    [pscustomobject]@{ Label = "逾期違約金"; Pattern = "逾期違約金" }
)

$DelayPatterns = @(
    [pscustomobject]@{ Label = "逾期"; Pattern = "逾期" },
    [pscustomobject]@{ Label = "遲延"; Pattern = "遲延" },
    [pscustomobject]@{ Label = "工期"; Pattern = "工期" },
    [pscustomobject]@{ Label = "展延"; Pattern = "展延" },
    [pscustomobject]@{ Label = "展期"; Pattern = "展期" },
    [pscustomobject]@{ Label = "完工期限"; Pattern = "完工期限" }
)

$ReductionPatterns = @(
    [pscustomobject]@{ Label = "酌減"; Pattern = "酌減" },
    [pscustomobject]@{ Label = "過高"; Pattern = "過高" },
    [pscustomobject]@{ Label = "相當"; Pattern = "相當" },
    [pscustomobject]@{ Label = "民法第252條"; Pattern = "民法第\s*252\s*條" },
    [pscustomobject]@{ Label = "民法第２５２條"; Pattern = "民法第\s*２５２\s*條" },
    [pscustomobject]@{ Label = "民法第二百五十二條"; Pattern = "民法第?二百五十二條" },
    [pscustomobject]@{ Label = "第252條"; Pattern = "第\s*252\s*條" }
)

$StrongReductionPatterns = @(
    [pscustomobject]@{ Label = "酌減"; Pattern = "酌減" },
    [pscustomobject]@{ Label = "過高"; Pattern = "過高" },
    [pscustomobject]@{ Label = "民法第252條"; Pattern = "民法第\s*252\s*條" },
    [pscustomobject]@{ Label = "民法第２５２條"; Pattern = "民法第\s*２５２\s*條" },
    [pscustomobject]@{ Label = "民法第二百五十二條"; Pattern = "民法第?二百五十二條" },
    [pscustomobject]@{ Label = "第252條"; Pattern = "第\s*252\s*條" }
)

function Get-PatternStats {
    param(
        [string]$Text,
        [object[]]$Patterns
    )

    $count = 0
    $labels = New-Object System.Collections.Generic.List[string]
    foreach ($pattern in $Patterns) {
        $matches = [regex]::Matches($Text, $pattern.Pattern)
        if ($matches.Count -gt 0) {
            $count += $matches.Count
            [void]$labels.Add($pattern.Label)
        }
    }

    [pscustomobject]@{
        Count  = $count
        Labels = @($labels | Sort-Object -Unique)
    }
}

function Get-Snippet {
    param(
        [string]$Text,
        [object[]]$Patterns,
        [int]$Radius = 90
    )

    $bestIndex = $null
    $bestLength = 0
    foreach ($pattern in $Patterns) {
        $match = [regex]::Match($Text, $pattern.Pattern)
        if ($match.Success -and ($null -eq $bestIndex -or $match.Index -lt $bestIndex)) {
            $bestIndex = $match.Index
            $bestLength = $match.Length
        }
    }

    if ($null -eq $bestIndex) {
        return ""
    }

    $start = [Math]::Max(0, $bestIndex - $Radius)
    $end = [Math]::Min($Text.Length, $bestIndex + $bestLength + $Radius)
    $snippet = $Text.Substring($start, $end - $start)
    return (($snippet -replace "\s+", " ").Trim())
}

function Get-DecisionYear {
    param(
        [object]$JDate,
        [object]$Month
    )

    $rawDate = [string]$JDate
    if ($rawDate -match "^\d{8}$") {
        return $rawDate.Substring(0, 4)
    }
    $rawMonth = [string]$Month
    if ($rawMonth -match "^\d{6}$") {
        return $rawMonth.Substring(0, 4)
    }
    return ""
}

function Read-JudgmentText {
    param([string]$JsonPath)

    if (-not (Test-Path -LiteralPath $JsonPath)) {
        return [pscustomobject]@{ Text = ""; Available = 0; Status = "json_missing" }
    }

    try {
        $json = Get-Content -Raw -Encoding UTF8 -LiteralPath $JsonPath | ConvertFrom-Json
    }
    catch {
        return [pscustomobject]@{ Text = ""; Available = 0; Status = "json_decode_error" }
    }

    $text = [string]$json.JFULL
    if ([string]::IsNullOrWhiteSpace($text)) {
        return [pscustomobject]@{ Text = ""; Available = 0; Status = "empty_jfull" }
    }

    [pscustomobject]@{ Text = $text; Available = 1; Status = "ok" }
}

function Get-StratifiedPrioritySample {
    param(
        [object[]]$Data,
        [int]$SampleSize
    )

    if ($Data.Count -le $SampleSize) {
        return @($Data)
    }

    $selected = New-Object System.Collections.Generic.List[object]
    $selectedIds = @{}
    $groups = $Data | Group-Object decision_year
    foreach ($group in $groups) {
        $n = [Math]::Floor($SampleSize * ($group.Count / $Data.Count))
        $n = [Math]::Max(1, $n)
        $n = [Math]::Min($n, $group.Count)
        $part = $group.Group |
            Sort-Object @{ Expression = "relevance_score"; Descending = $true }, JID |
            Select-Object -First $n
        foreach ($item in $part) {
            if (-not $selectedIds.ContainsKey($item.JID)) {
                [void]$selected.Add($item)
                $selectedIds[$item.JID] = $true
            }
        }
    }

    if ($selected.Count -lt $SampleSize) {
        $fill = $Data |
            Where-Object { -not $selectedIds.ContainsKey($_.JID) } |
            Sort-Object @{ Expression = "relevance_score"; Descending = $true }, decision_year, month, JID |
            Select-Object -First ($SampleSize - $selected.Count)
        foreach ($item in $fill) {
            [void]$selected.Add($item)
            $selectedIds[$item.JID] = $true
        }
    }

    if ($selected.Count -gt $SampleSize) {
        return @($selected |
            Sort-Object @{ Expression = "relevance_score"; Descending = $true }, decision_year, month, JID |
            Select-Object -First $SampleSize)
    }

    return @($selected | ForEach-Object { $_ })
}

function Get-YearCountsMarkdown {
    param([object[]]$Data)

    if ($Data.Count -eq 0) {
        return "無"
    }

    $lines = $Data |
        Group-Object decision_year |
        Sort-Object Name |
        ForEach-Object { "- $($_.Name): $($_.Count)" }
    return ($lines -join "`n")
}

Write-Host "Reading index: $IndexCsv"
$index = Import-Csv -Encoding UTF8 -LiteralPath $IndexCsv
$allPatterns = @($PenaltyPatterns + $DelayPatterns)
$screening = New-Object System.Collections.Generic.List[object]

foreach ($row in $index) {
    $relativeSrc = ([string]$row.src_path).Replace("/", [System.IO.Path]::DirectorySeparatorChar)
    $jsonPath = Join-Path $JsonRoot $relativeSrc
    $read = Read-JudgmentText -JsonPath $jsonPath
    $title = [string]$row.JTITLE
    $searchable = "$title`n$($read.Text)"

    $penalty = Get-PatternStats -Text $searchable -Patterns $PenaltyPatterns
    $delay = Get-PatternStats -Text $searchable -Patterns $DelayPatterns
    $reduction = Get-PatternStats -Text $searchable -Patterns $ReductionPatterns
    $strongReduction = Get-PatternStats -Text $searchable -Patterns $StrongReductionPatterns
    $titlePenalty = Get-PatternStats -Text $title -Patterns $PenaltyPatterns
    $titleDelay = Get-PatternStats -Text $title -Patterns $DelayPatterns
    $titleReduction = Get-PatternStats -Text $title -Patterns $ReductionPatterns

    $isOverduePenaltyCandidate = [int](($penalty.Count -gt 0) -and ($delay.Count -gt 0))
    $isPenaltyReductionHighRelevance = [int](($isOverduePenaltyCandidate -eq 1) -and ($strongReduction.Count -gt 0))
    $firstLayerScore = $penalty.Count + $delay.Count
    $secondLayerScore = $reduction.Count
    $strongReductionScore = $strongReduction.Count
    $relevanceScore = $firstLayerScore + (2 * $secondLayerScore) + (4 * $strongReductionScore) + (3 * $titlePenalty.Count) + $titleDelay.Count + $titleReduction.Count

    $jsonRelative = ""
    if (Test-Path -LiteralPath $jsonPath) {
        $fullJsonPath = (Resolve-Path -LiteralPath $jsonPath).Path
        if ($fullJsonPath.StartsWith($ProjectRoot)) {
            $jsonRelative = $fullJsonPath.Substring($ProjectRoot.Length).TrimStart("\", "/").Replace("\", "/")
        }
        else {
            $jsonRelative = $fullJsonPath.Replace("\", "/")
        }
    }

    [void]$screening.Add([pscustomobject]@{
        JID = $row.JID
        month = $row.month
        decision_year = Get-DecisionYear -JDate $row.JDATE -Month $row.month
        court = $row.court
        JYEAR = $row.JYEAR
        JCASE = $row.JCASE
        JNO = $row.JNO
        JDATE = $row.JDATE
        JTITLE = $title
        JFULL_len = $row.JFULL_len
        工程次數 = $row.工程次數
        src_path = ([string]$row.src_path).Replace("\", "/")
        json_file = $jsonRelative
        read_status = $read.Status
        judgment_text_available = $read.Available
        penalty_keyword_count = $penalty.Count
        delay_keyword_count = $delay.Count
        reduction_keyword_count = $reduction.Count
        strong_reduction_keyword_count = $strongReduction.Count
        first_layer_score = $firstLayerScore
        second_layer_score = $secondLayerScore
        strong_reduction_score = $strongReductionScore
        relevance_score = $relevanceScore
        matched_penalty_terms = ($penalty.Labels -join ";")
        matched_delay_terms = ($delay.Labels -join ";")
        matched_reduction_terms = ($reduction.Labels -join ";")
        matched_strong_reduction_terms = ($strongReduction.Labels -join ";")
        matched_first_layer_terms = (@($penalty.Labels + $delay.Labels) | Sort-Object -Unique) -join ";"
        is_overdue_penalty_candidate = $isOverduePenaltyCandidate
        is_penalty_reduction_high_relevance = $isPenaltyReductionHighRelevance
        first_layer_snippet = Get-Snippet -Text $searchable -Patterns $allPatterns
        reduction_snippet = Get-Snippet -Text $searchable -Patterns $ReductionPatterns
    })
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$allScreeningPath = Join-Path $OutputDir "all_keyword_screening.csv"
$overduePath = Join-Path $OutputDir "overdue_penalty_candidates.csv"
$highPath = Join-Path $OutputDir "penalty_reduction_high_relevance.csv"
$samplePath = Join-Path $OutputDir "penalty_reduction_annotation_sample.csv"
$summaryPath = Join-Path $OutputDir "keyword_screening_summary.md"

$overdue = @($screening | Where-Object { $_.is_overdue_penalty_candidate -eq 1 } | Sort-Object decision_year, month, court, JID)
$high = @($screening | Where-Object { $_.is_penalty_reduction_high_relevance -eq 1 } | Sort-Object @{ Expression = "relevance_score"; Descending = $true }, decision_year, month, court, JID)
$sampleRaw = @(Get-StratifiedPrioritySample -Data $high -SampleSize $SampleSize)
$sampleRaw = @($sampleRaw | Sort-Object decision_year, month, @{ Expression = "relevance_score"; Descending = $true }, JID)

$sample = New-Object System.Collections.Generic.List[object]
$priority = 1
foreach ($item in $sampleRaw) {
    [void]$sample.Add([pscustomobject]@{
        annotation_status = "needs_manual_review"
        annotation_priority = $priority
        JID = $item.JID
        month = $item.month
        decision_year = $item.decision_year
        court = $item.court
        JYEAR = $item.JYEAR
        JCASE = $item.JCASE
        JNO = $item.JNO
        JDATE = $item.JDATE
        JTITLE = $item.JTITLE
        JFULL_len = $item.JFULL_len
        工程次數 = $item.工程次數
        src_path = $item.src_path
        json_file = $item.json_file
        read_status = $item.read_status
        judgment_text_available = $item.judgment_text_available
        penalty_keyword_count = $item.penalty_keyword_count
        delay_keyword_count = $item.delay_keyword_count
        reduction_keyword_count = $item.reduction_keyword_count
        strong_reduction_keyword_count = $item.strong_reduction_keyword_count
        first_layer_score = $item.first_layer_score
        second_layer_score = $item.second_layer_score
        strong_reduction_score = $item.strong_reduction_score
        relevance_score = $item.relevance_score
        matched_penalty_terms = $item.matched_penalty_terms
        matched_delay_terms = $item.matched_delay_terms
        matched_reduction_terms = $item.matched_reduction_terms
        matched_strong_reduction_terms = $item.matched_strong_reduction_terms
        matched_first_layer_terms = $item.matched_first_layer_terms
        is_overdue_penalty_candidate = $item.is_overdue_penalty_candidate
        is_penalty_reduction_high_relevance = $item.is_penalty_reduction_high_relevance
        first_layer_snippet = $item.first_layer_snippet
        reduction_snippet = $item.reduction_snippet
    })
    $priority += 1
}
$sampleRows = @($sample | ForEach-Object { $_ })

$screening | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $allScreeningPath
$overdue | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $overduePath
$high | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $highPath
$sampleRows | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $samplePath

$summary = @(
    "# 關鍵詞篩選摘要",
    "",
    "## 篩選規則",
    "",
    "- 母體：2021-2026 年工程相關判決 JSON 全文。",
    "- 第一層逾期違約金候選池：同時命中違約金語彙與逾期/工期/展延語彙。",
    "- 第二層違約金酌減高度相關池：第一層候選池中，再命中酌減、民法第252條、過高等較強酌減語彙；相當保留為輔助命中詞，但不單獨使案件進入高度相關池。",
    "- 抽樣精標清單：自第二層高度相關池依裁判年度分層，優先選取命中分數較高的案件。",
    "",
    "## 輸出筆數",
    "",
    "- 母體總筆數：$($screening.Count)",
    "- 逾期違約金候選池：$($overdue.Count)",
    "- 違約金酌減高度相關池：$($high.Count)",
    "- 抽樣精標清單：$($sampleRows.Count) / 目標 $SampleSize",
    "",
    "## 高度相關池年度分布",
    "",
    (Get-YearCountsMarkdown -Data $high),
    "",
    "## 抽樣精標清單年度分布",
    "",
    (Get-YearCountsMarkdown -Data $sampleRows),
    "",
    "## 注意事項",
    "",
    "- 本篩選只建立候選池，不代表案件必然以逾期違約金酌減為核心爭點。",
    "- `相當` 屬於較寬鬆語彙，本腳本保留其命中紀錄，但高度相關池不接受只命中 `相當` 的案件。",
    "- 金額、逾期天數、酌減比例與法院核心理由不得只依關鍵詞自動判定，仍需回到判決全文查核。"
)
$summary -join "`n" | Set-Content -Encoding UTF8 -LiteralPath $summaryPath

Write-Host "母體總筆數: $($screening.Count)"
Write-Host "逾期違約金候選池: $($overdue.Count)"
Write-Host "違約金酌減高度相關池: $($high.Count)"
Write-Host "抽樣精標清單: $($sampleRows.Count)"
Write-Host "all_screening: $allScreeningPath"
Write-Host "overdue_candidates: $overduePath"
Write-Host "high_relevance: $highPath"
Write-Host "annotation_sample: $samplePath"
Write-Host "summary: $summaryPath"





