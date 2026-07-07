param(
    [string]$InputCsv,
    [string]$OutputDir,
    [int]$MinTargetRows = 30,
    [int]$TrainStartYear = 2021,
    [int]$TrainEndYear = 2023,
    [int]$ValidationYear = 2024,
    [int]$TestYear = 2025,
    [int]$LatestCheckYear = 2026,
    [int]$Iterations = 3000,
    [double]$LearningRate = 0.03,
    [double]$RidgeLambda = 0.05,
    [double]$LassoLambda = 0.01
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
if (-not $InputCsv) {
    $InputCsv = Join-Path $ProjectRoot "06_交付物\ai_rag_annotation\annotation_workbook.csv"
}
if (-not $OutputDir) {
    $OutputDir = Join-Path $ProjectRoot "06_交付物\reduction_ratio_model"
}

$FeatureNames = @(
    "x_log_contract_price",
    "x_log_claimed_penalty",
    "x_claim_to_contract_ratio",
    "x_delay_days",
    "x_penalty_per_delay_day",
    "x_issue_owner_fault",
    "x_issue_contractor_fault",
    "x_issue_extension_request",
    "x_issue_actual_damage_unclear",
    "x_issue_partial_completion",
    "x_issue_used_by_owner",
    "x_ai_issue_owner_fault",
    "x_ai_issue_contractor_fault",
    "x_ai_issue_extension_request",
    "x_ai_issue_actual_damage_unclear",
    "x_ai_issue_partial_completion",
    "x_ai_issue_used_by_owner",
    "x_money_candidate_count",
    "x_delay_candidate_count"
)

$ScreeningOnlyFeatures = @(
    "x_relevance_score",
    "x_penalty_term_count",
    "x_delay_term_count",
    "x_strong_reduction_term_count",
    "x_has_mfa_252",
    "x_has_discretion",
    "x_has_over_high"
)

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

function Get-NumberOrNull {
    param([object]$Value)

    if ($null -eq $Value) {
        return $null
    }
    $text = ([string]$Value).Trim()
    if ([string]::IsNullOrWhiteSpace($text)) {
        return $null
    }
    $text = $text.Replace(",", "").Replace("，", "").Replace("%", "")
    $number = 0.0
    if ([double]::TryParse($text, [ref]$number)) {
        return $number
    }
    return $null
}

function Get-FeatureNumber {
    param([object]$Value)

    if ($null -eq $Value) {
        return 0.0
    }
    return [double]$Value
}

function Get-NullableLabel {
    param([object]$Value)

    if ($null -eq $Value) {
        return $null
    }
    $text = ([string]$Value).Trim().ToLowerInvariant()
    if ([string]::IsNullOrWhiteSpace($text)) {
        return $null
    }
    if ($text -in @("1", "true", "t", "yes", "y", "是", "有", "酌減", "已酌減")) {
        return 1
    }
    if ($text -in @("0", "false", "f", "no", "n", "否", "無", "未酌減", "不酌減")) {
        return 0
    }
    $number = Get-NumberOrNull $text
    if ($null -ne $number) {
        return [int]($number -ne 0)
    }
    return $null
}

function Get-Flag {
    param([object]$Value)

    $label = Get-NullableLabel $Value
    if ($null -eq $label) {
        return 0
    }
    return [int]$label
}

function Get-DelimitedCount {
    param([object]$Value)

    $text = ([string]$Value).Trim()
    if ([string]::IsNullOrWhiteSpace($text)) {
        return 0
    }
    $parts = @($text -split ";" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($parts.Count -gt 0) {
        return $parts.Count
    }
    return 1
}

function Get-CandidateCount {
    param([object]$Value)

    $text = ([string]$Value).Trim()
    if ([string]::IsNullOrWhiteSpace($text)) {
        return 0
    }
    $pipeCount = [regex]::Matches($text, "｜").Count
    if ($pipeCount -gt 0) {
        return $pipeCount
    }
    return Get-DelimitedCount $text
}

function Join-RowText {
    param([object]$Row)

    $fields = @(
        "JTITLE",
        "key_reason",
        "source_snippet_penalty",
        "source_snippet_delay",
        "source_snippet_reduction",
        "source_snippets_combined",
        "ai_evidence_issue_owner_fault",
        "ai_evidence_issue_contractor_fault",
        "ai_evidence_issue_extension_request",
        "ai_evidence_issue_actual_damage_unclear",
        "ai_evidence_issue_partial_completion",
        "ai_evidence_issue_used_by_owner"
    )
    return (($fields | ForEach-Object { Get-Field $Row $_ }) -join "`n")
}

function Test-TextAny {
    param(
        [string]$Text,
        [string[]]$Terms
    )

    foreach ($term in $Terms) {
        if ($Text.Contains($term)) {
            return 1
        }
    }
    return 0
}

function Get-LogOrZero {
    param([object]$Value)

    if ($null -eq $Value -or [double]$Value -le 0) {
        return 0.0
    }
    return [Math]::Log([double]$Value)
}

function Get-ReductionBucket {
    param([object]$Ratio)

    if ($null -eq $Ratio -or [string]::IsNullOrWhiteSpace([string]$Ratio)) {
        return ""
    }
    $r = [double]$Ratio
    if ($r -le 0.05) {
        return "全免或近乎全免"
    }
    if ($r -lt 0.30) {
        return "大幅酌減"
    }
    if ($r -lt 0.70) {
        return "中度酌減"
    }
    if ($r -lt 0.99) {
        return "小幅酌減"
    }
    return "未酌減"
}

function Get-TargetQuality {
    param(
        [object]$ClaimedPenalty,
        [object]$AllowedPenalty,
        [object]$RemainingRatio
    )

    if ($null -eq $ClaimedPenalty -or $null -eq $AllowedPenalty) {
        return "missing_claimed_or_allowed_penalty"
    }
    if ([double]$ClaimedPenalty -le 0) {
        return "claimed_penalty_nonpositive"
    }
    if ([double]$AllowedPenalty -lt 0) {
        return "allowed_penalty_negative"
    }
    if ([double]$RemainingRatio -lt 0 -or [double]$RemainingRatio -gt 1) {
        return "remaining_ratio_out_of_0_1_range"
    }
    return "ok"
}

function Export-ObjectCsv {
    param(
        [object[]]$Rows,
        [string]$Path,
        [string[]]$Headers
    )

    if ($Rows.Count -gt 0) {
        $Rows | Select-Object $Headers | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $Path
    }
    else {
        ($Headers -join ",") | Set-Content -Encoding UTF8 -LiteralPath $Path
    }
}

function Get-PropDouble {
    param(
        [object]$Row,
        [string]$Name
    )

    $value = $Row.PSObject.Properties[$Name].Value
    if ($null -eq $value -or [string]::IsNullOrWhiteSpace([string]$value)) {
        return 0.0
    }
    return [double]$value
}

function Get-Scaler {
    param(
        [object[]]$Rows,
        [string[]]$Names
    )

    $means = @{}
    $stds = @{}
    foreach ($name in $Names) {
        $values = @($Rows | ForEach-Object { Get-PropDouble $_ $name })
        $mean = 0.0
        if ($values.Count -gt 0) {
            $mean = ($values | Measure-Object -Average).Average
        }
        $variance = 0.0
        foreach ($value in $values) {
            $variance += [Math]::Pow(($value - $mean), 2)
        }
        if ($values.Count -gt 0) {
            $variance = $variance / $values.Count
        }
        $std = [Math]::Sqrt($variance)
        if ($std -lt 0.000000001) {
            $std = 1.0
        }
        $means[$name] = [double]$mean
        $stds[$name] = [double]$std
    }
    [pscustomobject]@{
        Means = $means
        Stds = $stds
    }
}

function Get-ScaledVector {
    param(
        [object]$Row,
        [string[]]$Names,
        [hashtable]$Means,
        [hashtable]$Stds
    )

    $vector = New-Object double[] $Names.Count
    for ($i = 0; $i -lt $Names.Count; $i += 1) {
        $name = $Names[$i]
        $vector[$i] = ((Get-PropDouble $Row $name) - $Means[$name]) / $Stds[$name]
    }
    return $vector
}

function Get-SoftThreshold {
    param(
        [double]$Value,
        [double]$Threshold
    )

    if ($Value -gt $Threshold) {
        return $Value - $Threshold
    }
    if ($Value -lt -$Threshold) {
        return $Value + $Threshold
    }
    return 0.0
}

function Train-LinearModel {
    param(
        [object[]]$Rows,
        [string[]]$Names,
        [hashtable]$Means,
        [hashtable]$Stds,
        [string]$ModelKind,
        [double]$Lambda,
        [int]$Iterations,
        [double]$LearningRate
    )

    $weights = New-Object double[] ($Names.Count + 1)
    $n = [double]$Rows.Count
    for ($iter = 0; $iter -lt $Iterations; $iter += 1) {
        $grad = New-Object double[] ($Names.Count + 1)
        foreach ($row in $Rows) {
            $x = Get-ScaledVector -Row $row -Names $Names -Means $Means -Stds $Stds
            $pred = $weights[0]
            for ($i = 0; $i -lt $x.Count; $i += 1) {
                $pred += $weights[$i + 1] * $x[$i]
            }
            $error = $pred - [double]$row.remaining_ratio
            $grad[0] += $error
            for ($i = 0; $i -lt $x.Count; $i += 1) {
                $grad[$i + 1] += $error * $x[$i]
            }
        }
        $weights[0] -= $LearningRate * ($grad[0] / $n)
        for ($j = 1; $j -lt $weights.Count; $j += 1) {
            $step = $weights[$j] - ($LearningRate * ($grad[$j] / $n))
            if ($ModelKind -eq "ridge") {
                $step -= $LearningRate * $Lambda * $weights[$j]
            }
            elseif ($ModelKind -eq "lasso") {
                $step = Get-SoftThreshold -Value $step -Threshold ($LearningRate * $Lambda)
            }
            $weights[$j] = $step
        }
    }
    return $weights
}

function Get-LinearPrediction {
    param(
        [object]$Row,
        [string[]]$Names,
        [hashtable]$Means,
        [hashtable]$Stds,
        [double[]]$Weights
    )

    $x = Get-ScaledVector -Row $Row -Names $Names -Means $Means -Stds $Stds
    $pred = $Weights[0]
    for ($i = 0; $i -lt $x.Count; $i += 1) {
        $pred += $Weights[$i + 1] * $x[$i]
    }
    return $pred
}

function Get-ClippedRatio {
    param([double]$Value)

    if ($Value -lt 0) {
        return 0.0
    }
    if ($Value -gt 1) {
        return 1.0
    }
    return $Value
}

function Add-PredictionRows {
    param(
        [System.Collections.Generic.List[object]]$Output,
        [object[]]$Rows,
        [string]$SplitName,
        [string]$ModelName,
        [scriptblock]$PredictionScript
    )

    foreach ($row in $Rows) {
        $rawPred = [double](& $PredictionScript $row)
        $predRatio = Get-ClippedRatio $rawPred
        $actualRatio = [double]$row.remaining_ratio
        [void]$Output.Add([pscustomobject]@{
            model = $ModelName
            split = $SplitName
            JID = $row.JID
            decision_year = $row.decision_year
            actual_remaining_ratio = [Math]::Round($actualRatio, 6)
            predicted_remaining_ratio_raw = [Math]::Round($rawPred, 6)
            predicted_remaining_ratio = [Math]::Round($predRatio, 6)
            actual_reduction_rate = [Math]::Round((1.0 - $actualRatio), 6)
            predicted_reduction_rate = [Math]::Round((1.0 - $predRatio), 6)
            actual_bucket = Get-ReductionBucket $actualRatio
            predicted_bucket = Get-ReductionBucket $predRatio
        })
    }
}

function Format-NullableMetric {
    param([object]$Value)

    if ($null -eq $Value) {
        return ""
    }
    return [Math]::Round([double]$Value, 4)
}

function Get-MetricsRow {
    param(
        [string]$ModelName,
        [string]$SplitName,
        [object[]]$Predictions
    )

    $n = $Predictions.Count
    if ($n -eq 0) {
        return [pscustomobject]@{
            model = $ModelName
            split = $SplitName
            n = 0
            mae = ""
            rmse = ""
            r2 = ""
            bucket_accuracy = ""
        }
    }

    $absError = 0.0
    $squaredError = 0.0
    $actualValues = @($Predictions | ForEach-Object { [double]$_.actual_remaining_ratio })
    $meanActual = ($actualValues | Measure-Object -Average).Average
    $sst = 0.0
    $bucketHits = 0
    foreach ($prediction in $Predictions) {
        $actual = [double]$prediction.actual_remaining_ratio
        $pred = [double]$prediction.predicted_remaining_ratio
        $absError += [Math]::Abs($actual - $pred)
        $squaredError += [Math]::Pow(($actual - $pred), 2)
        $sst += [Math]::Pow(($actual - $meanActual), 2)
        if ($prediction.actual_bucket -eq $prediction.predicted_bucket) {
            $bucketHits += 1
        }
    }

    $r2 = $null
    if ($sst -gt 0) {
        $r2 = 1.0 - ($squaredError / $sst)
    }

    [pscustomobject]@{
        model = $ModelName
        split = $SplitName
        n = $n
        mae = [Math]::Round(($absError / $n), 4)
        rmse = [Math]::Round([Math]::Sqrt($squaredError / $n), 4)
        r2 = Format-NullableMetric $r2
        bucket_accuracy = [Math]::Round(($bucketHits / [double]$n), 4)
    }
}

function Get-QualityCounts {
    param([object[]]$Rows)

    $counts = [ordered]@{}
    foreach ($group in ($Rows | Group-Object target_quality | Sort-Object Name)) {
        $counts[$group.Name] = $group.Count
    }
    return $counts
}

function Write-SkippedStatus {
    param(
        [string]$Reason,
        [hashtable]$Details
    )

    $status = [ordered]@{
        status = "skipped"
        reason = $Reason
        input_csv = $InputCsv
        total_rows = $Details.total_rows
        usable_target_rows = $Details.usable_target_rows
        min_target_rows = $MinTargetRows
        target_quality_counts = $Details.target_quality_counts
        model_features = $FeatureNames
        screening_only_features = $ScreeningOnlyFeatures
        split_policy = [ordered]@{
            train = "$TrainStartYear-$TrainEndYear"
            validation = "$ValidationYear"
            test = "$TestYear"
            latest_check = "$LatestCheckYear"
        }
        outputs = $Details.outputs
        note = "Targets are derived only from manually verified claimed_penalty and allowed_penalty. Raw penalty amount is not used as the target."
    }
    $status | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $OutputDir "model_status.json")

    $lines = @(
        "# 酌減比例迴歸模型狀態",
        "",
        "- 狀態：skipped",
        "- 原因：$Reason",
        ("- 輸入資料：" + $InputCsv),
        "- 全部案件：$($Details.total_rows)",
        "- 可用比例目標列數：$($Details.usable_target_rows)",
        "- 最低訓練門檻：$MinTargetRows",
        "",
        "## 已產出",
        "",
        '- `ratio_model_frame.csv`：全部待精標案件的比例目標與模型特徵。',
        '- `usable_ratio_model_frame.csv`：目前可用於比例模型的子表。',
        '- `target_quality_summary.csv`：比例目標品質摘要。',
        '- `model_status.json`：機器可讀狀態。',
        "",
        "## 後續補標",
        "",
        '- 先在 `annotation_workbook.csv` 補 `claimed_penalty` 與 `allowed_penalty`。',
        '- `remaining_ratio = allowed_penalty / claimed_penalty`，需落在 0 到 1 之間才會進入模型。',
        "- 補標後重新執行本腳本，即可依 $TrainStartYear-$TrainEndYear 訓練、$ValidationYear 驗證、$TestYear 測試、$LatestCheckYear 最新年度檢查輸出 MAE、RMSE、R2 與分段命中率。",
        "",
        "## 防呆說明",
        "",
        '- 本腳本不直接預測原始判賠金額，而是預測 `remaining_ratio`。',
        '- `allowed_penalty` 只用於建立目標變數，不會被放入模型特徵。'
    )
    $lines -join "`n" | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $OutputDir "迴歸模型狀態.md")
}

Write-Host "Reading annotation workbook: $InputCsv"
$rows = Import-Csv -Encoding UTF8 -LiteralPath $InputCsv
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$modelRows = New-Object System.Collections.Generic.List[object]
foreach ($row in $rows) {
    $decisionYear = Get-NumberOrNull (Get-Field $row "decision_year")
    $contractPrice = Get-NumberOrNull (Get-Field $row "contract_price")
    $delayDays = Get-NumberOrNull (Get-Field $row "delay_days")
    $claimedPenalty = Get-NumberOrNull (Get-Field $row "claimed_penalty")
    $allowedPenalty = Get-NumberOrNull (Get-Field $row "allowed_penalty")

    $remainingRatio = $null
    $reductionRate = $null
    if ($null -ne $claimedPenalty -and $null -ne $allowedPenalty -and $claimedPenalty -gt 0) {
        $remainingRatio = [double]$allowedPenalty / [double]$claimedPenalty
        $reductionRate = 1.0 - $remainingRatio
    }
    $targetQuality = Get-TargetQuality -ClaimedPenalty $claimedPenalty -AllowedPenalty $allowedPenalty -RemainingRatio $remainingRatio

    $claimToContractRatio = 0.0
    if ($null -ne $contractPrice -and $contractPrice -gt 0 -and $null -ne $claimedPenalty) {
        $claimToContractRatio = [double]$claimedPenalty / [double]$contractPrice
    }
    $penaltyPerDelayDay = 0.0
    if ($null -ne $claimedPenalty -and $null -ne $delayDays -and $delayDays -gt 0) {
        $penaltyPerDelayDay = [double]$claimedPenalty / [double]$delayDays
    }

    $combinedText = Join-RowText $row
    $relevanceScore = Get-NumberOrNull (Get-Field $row "relevance_score")
    $penaltyTerms = Get-Field $row "matched_penalty_terms"
    $delayTerms = Get-Field $row "matched_delay_terms"
    $strongReductionTerms = Get-Field $row "matched_strong_reduction_terms"
    $hasMfa252 = [int]($combinedText -match "民法第\s*(252|２５２|二百五十二)\s*條|第\s*(252|２５２)\s*條")
    $hasDiscretion = Test-TextAny -Text $combinedText -Terms @("酌減", "核減", "酌予")
    $hasOverHigh = Test-TextAny -Text $combinedText -Terms @("過高", "顯非合理", "顯屬過高")

    [void]$modelRows.Add([pscustomobject][ordered]@{
        annotation_status = Get-Field $row "annotation_status"
        annotation_priority = Get-Field $row "annotation_priority"
        JID = Get-Field $row "JID"
        decision_year = Get-FeatureNumber $decisionYear
        court = Get-Field $row "court"
        JTITLE = Get-Field $row "JTITLE"
        JCASE = Get-Field $row "JCASE"
        JNO = Get-Field $row "JNO"
        JDATE = Get-Field $row "JDATE"
        json_file = Get-Field $row "json_file"
        contract_price = if ($null -eq $contractPrice) { "" } else { [double]$contractPrice }
        delay_days = if ($null -eq $delayDays) { "" } else { [double]$delayDays }
        claimed_penalty = if ($null -eq $claimedPenalty) { "" } else { [double]$claimedPenalty }
        allowed_penalty = if ($null -eq $allowedPenalty) { "" } else { [double]$allowedPenalty }
        remaining_ratio = if ($null -eq $remainingRatio) { "" } else { [Math]::Round($remainingRatio, 8) }
        reduction_rate = if ($null -eq $reductionRate) { "" } else { [Math]::Round($reductionRate, 8) }
        reduction_bucket = Get-ReductionBucket $remainingRatio
        target_quality = $targetQuality
        x_log_contract_price = Get-LogOrZero $contractPrice
        x_log_claimed_penalty = Get-LogOrZero $claimedPenalty
        x_claim_to_contract_ratio = $claimToContractRatio
        x_delay_days = Get-FeatureNumber $delayDays
        x_penalty_per_delay_day = $penaltyPerDelayDay
        x_issue_owner_fault = Get-Flag (Get-Field $row "issue_owner_fault")
        x_issue_contractor_fault = Get-Flag (Get-Field $row "issue_contractor_fault")
        x_issue_extension_request = Get-Flag (Get-Field $row "issue_extension_request")
        x_issue_actual_damage_unclear = Get-Flag (Get-Field $row "issue_actual_damage_unclear")
        x_issue_partial_completion = Get-Flag (Get-Field $row "issue_partial_completion")
        x_issue_used_by_owner = Get-Flag (Get-Field $row "issue_used_by_owner")
        x_ai_issue_owner_fault = Get-Flag (Get-Field $row "ai_suggest_issue_owner_fault")
        x_ai_issue_contractor_fault = Get-Flag (Get-Field $row "ai_suggest_issue_contractor_fault")
        x_ai_issue_extension_request = Get-Flag (Get-Field $row "ai_suggest_issue_extension_request")
        x_ai_issue_actual_damage_unclear = Get-Flag (Get-Field $row "ai_suggest_issue_actual_damage_unclear")
        x_ai_issue_partial_completion = Get-Flag (Get-Field $row "ai_suggest_issue_partial_completion")
        x_ai_issue_used_by_owner = Get-Flag (Get-Field $row "ai_suggest_issue_used_by_owner")
        x_money_candidate_count = (Get-CandidateCount (Get-Field $row "ai_contract_price_candidates")) + (Get-CandidateCount (Get-Field $row "ai_claimed_penalty_candidates")) + (Get-CandidateCount (Get-Field $row "ai_allowed_penalty_candidates"))
        x_delay_candidate_count = Get-CandidateCount (Get-Field $row "ai_delay_days_candidates")
        x_relevance_score = Get-FeatureNumber $relevanceScore
        x_penalty_term_count = Get-DelimitedCount $penaltyTerms
        x_delay_term_count = Get-DelimitedCount $delayTerms
        x_strong_reduction_term_count = Get-DelimitedCount $strongReductionTerms
        x_has_mfa_252 = $hasMfa252
        x_has_discretion = $hasDiscretion
        x_has_over_high = $hasOverHigh
    })
}

$modelFramePath = Join-Path $OutputDir "ratio_model_frame.csv"
$usableFramePath = Join-Path $OutputDir "usable_ratio_model_frame.csv"
$qualityPath = Join-Path $OutputDir "target_quality_summary.csv"
$headers = @($modelRows[0].PSObject.Properties.Name)
$modelArray = @($modelRows | ForEach-Object { $_ })
$usableRows = @($modelArray | Where-Object { $_.target_quality -eq "ok" })
Export-ObjectCsv -Rows $modelArray -Path $modelFramePath -Headers $headers
Export-ObjectCsv -Rows $usableRows -Path $usableFramePath -Headers $headers

$qualityRows = @($modelArray |
    Group-Object target_quality |
    Sort-Object Name |
    ForEach-Object {
        [pscustomobject]@{
            target_quality = $_.Name
            count = $_.Count
        }
    })
$qualityRows | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $qualityPath

$outputs = [ordered]@{
    ratio_model_frame = $modelFramePath
    usable_ratio_model_frame = $usableFramePath
    target_quality_summary = $qualityPath
}
$qualityCounts = Get-QualityCounts $modelArray

if ($usableRows.Count -lt $MinTargetRows) {
    Write-SkippedStatus -Reason "not_enough_verified_penalty_ratio_targets" -Details @{
        total_rows = $modelArray.Count
        usable_target_rows = $usableRows.Count
        target_quality_counts = $qualityCounts
        outputs = $outputs
    }
    Write-Host "Skipped model: only $($usableRows.Count) usable ratio targets. Model frame written to $modelFramePath"
    exit 0
}

$trainRows = @($usableRows | Where-Object { [int]$_.decision_year -ge $TrainStartYear -and [int]$_.decision_year -le $TrainEndYear })
$validationRows = @($usableRows | Where-Object { [int]$_.decision_year -eq $ValidationYear })
$testRows = @($usableRows | Where-Object { [int]$_.decision_year -eq $TestYear })
$latestRows = @($usableRows | Where-Object { [int]$_.decision_year -eq $LatestCheckYear })

if ($trainRows.Count -lt 10) {
    Write-SkippedStatus -Reason "not_enough_training_rows_in_train_split" -Details @{
        total_rows = $modelArray.Count
        usable_target_rows = $usableRows.Count
        target_quality_counts = $qualityCounts
        outputs = $outputs
    }
    Write-Host "Skipped model: training split has insufficient rows."
    exit 0
}

$scaler = Get-Scaler -Rows $trainRows -Names $FeatureNames
$ridgeWeights = Train-LinearModel -Rows $trainRows -Names $FeatureNames -Means $scaler.Means -Stds $scaler.Stds -ModelKind "ridge" -Lambda $RidgeLambda -Iterations $Iterations -LearningRate $LearningRate
$lassoWeights = Train-LinearModel -Rows $trainRows -Names $FeatureNames -Means $scaler.Means -Stds $scaler.Stds -ModelKind "lasso" -Lambda $LassoLambda -Iterations $Iterations -LearningRate $LearningRate
$meanRemainingRatio = ($trainRows | ForEach-Object { [double]$_.remaining_ratio } | Measure-Object -Average).Average

$predictions = New-Object System.Collections.Generic.List[object]
$trainSplitName = "train_{0}_{1}" -f $TrainStartYear, $TrainEndYear
$validationSplitName = "validation_{0}" -f $ValidationYear
$testSplitName = "test_{0}" -f $TestYear
$latestSplitName = "latest_{0}" -f $LatestCheckYear
$splits = @(
    [pscustomobject]@{ Name = $trainSplitName; Rows = $trainRows },
    [pscustomobject]@{ Name = $validationSplitName; Rows = $validationRows },
    [pscustomobject]@{ Name = $testSplitName; Rows = $testRows },
    [pscustomobject]@{ Name = $latestSplitName; Rows = $latestRows }
)

foreach ($split in $splits) {
    Add-PredictionRows -Output $predictions -Rows $split.Rows -SplitName $split.Name -ModelName "mean_baseline" -PredictionScript {
        param($r)
        return $meanRemainingRatio
    }
    Add-PredictionRows -Output $predictions -Rows $split.Rows -SplitName $split.Name -ModelName "ridge_regression_l2" -PredictionScript {
        param($r)
        return Get-LinearPrediction -Row $r -Names $FeatureNames -Means $scaler.Means -Stds $scaler.Stds -Weights $ridgeWeights
    }
    Add-PredictionRows -Output $predictions -Rows $split.Rows -SplitName $split.Name -ModelName "lasso_regression_l1" -PredictionScript {
        param($r)
        return Get-LinearPrediction -Row $r -Names $FeatureNames -Means $scaler.Means -Stds $scaler.Stds -Weights $lassoWeights
    }
}

$predictionRows = @($predictions | ForEach-Object { $_ })
$metrics = New-Object System.Collections.Generic.List[object]
foreach ($modelName in @("mean_baseline", "ridge_regression_l2", "lasso_regression_l1")) {
    foreach ($split in $splits) {
        $modelSplitPredictions = @($predictionRows | Where-Object { $_.model -eq $modelName -and $_.split -eq $split.Name })
        [void]$metrics.Add((Get-MetricsRow -ModelName $modelName -SplitName $split.Name -Predictions $modelSplitPredictions))
    }
}

$metricsPath = Join-Path $OutputDir "metrics.csv"
$predictionsPath = Join-Path $OutputDir "predictions.csv"
$coefficientsPath = Join-Path $OutputDir "model_coefficients.csv"
$metrics | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $metricsPath
$predictionRows | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $predictionsPath

$coefs = New-Object System.Collections.Generic.List[object]
foreach ($modelInfo in @(
    [pscustomobject]@{ Name = "ridge_regression_l2"; Weights = $ridgeWeights },
    [pscustomobject]@{ Name = "lasso_regression_l1"; Weights = $lassoWeights }
)) {
    [void]$coefs.Add([pscustomobject]@{
        model = $modelInfo.Name
        feature = "intercept"
        mean = ""
        std = ""
        coefficient_scaled = [Math]::Round($modelInfo.Weights[0], 8)
        direction = if ($modelInfo.Weights[0] -gt 0) { "准許比例同向" } elseif ($modelInfo.Weights[0] -lt 0) { "酌減比例同向" } else { "接近零" }
    })
    for ($i = 0; $i -lt $FeatureNames.Count; $i += 1) {
        $coef = $modelInfo.Weights[$i + 1]
        [void]$coefs.Add([pscustomobject]@{
            model = $modelInfo.Name
            feature = $FeatureNames[$i]
            mean = [Math]::Round($scaler.Means[$FeatureNames[$i]], 8)
            std = [Math]::Round($scaler.Stds[$FeatureNames[$i]], 8)
            coefficient_scaled = [Math]::Round($coef, 8)
            direction = if ($coef -gt 0) { "准許比例同向" } elseif ($coef -lt 0) { "酌減比例同向" } else { "接近零" }
        })
    }
}
$coefs | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $coefficientsPath

$status = [ordered]@{
    status = "ok"
    input_csv = $InputCsv
    total_rows = $modelArray.Count
    usable_target_rows = $usableRows.Count
    training_split = "$TrainStartYear-$TrainEndYear"
    validation_split = "$ValidationYear"
    test_split = "$TestYear"
    latest_check_split = "$LatestCheckYear"
    target_quality_counts = $qualityCounts
    model_features = $FeatureNames
    screening_only_features = $ScreeningOnlyFeatures
    models = @("mean_baseline", "ridge_regression_l2", "lasso_regression_l1")
    planned_package_models = @("RandomForestRegressor", "LightGBMRegressor")
    outputs = [ordered]@{
        ratio_model_frame = $modelFramePath
        usable_ratio_model_frame = $usableFramePath
        target_quality_summary = $qualityPath
        metrics = $metricsPath
        predictions = $predictionsPath
        model_coefficients = $coefficientsPath
    }
    note = "Models predict remaining_ratio. reduction_rate is derived as 1 - remaining_ratio. Raw amount is not used as target."
}
$status | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $OutputDir "model_status.json")

$lines = @(
    "# 酌減比例迴歸模型回測結果",
    "",
    "## 設計",
    "",
    '- 目標變數：`remaining_ratio = allowed_penalty / claimed_penalty`。',
    '- 衍生變數：`reduction_rate = 1 - remaining_ratio`。',
    "- 時間切分：$TrainStartYear-$TrainEndYear 訓練、$ValidationYear 驗證、$TestYear 測試、$LatestCheckYear 最新年度檢查。",
    "- 模型：mean baseline、Ridge Regression、Lasso Regression。",
    "- 指標：MAE、RMSE、R2、分段命中率。",
    "",
    "## 輸出",
    "",
    '- `ratio_model_frame.csv`',
    '- `usable_ratio_model_frame.csv`',
    '- `target_quality_summary.csv`',
    '- `metrics.csv`',
    '- `predictions.csv`',
    '- `model_coefficients.csv`',
    '- `model_status.json`',
    "",
    "## 注意",
    "",
    '- 本模型不直接預測原始判賠金額，而是預測 `remaining_ratio`。',
    '- `allowed_penalty` 只用於建立目標變數，不會放入模型特徵。',
    "- Random Forest 與 LightGBM 需等 Python 套件環境穩定後再接入。"
)
$lines -join "`n" | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $OutputDir "迴歸模型狀態.md")

Write-Host "Wrote ratio model frame: $modelFramePath"
Write-Host "Wrote metrics: $metricsPath"
