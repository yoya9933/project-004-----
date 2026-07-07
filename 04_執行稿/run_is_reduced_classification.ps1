param(
    [string]$InputCsv,
    [string]$OutputDir,
    [int]$MinLabeledRows = 30,
    [int]$Iterations = 2500,
    [double]$LearningRate = 0.05,
    [double]$L2 = 0.01,
    [int]$TrainStartYear = 2021,
    [int]$TrainEndYear = 2023,
    [int]$ValidationYear = 2024,
    [int]$TestYear = 2025,
    [int]$LatestCheckYear = 2026,
    [switch]$UseDerivedLabelFromAmounts
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
if (-not $InputCsv) {
    $InputCsv = Join-Path $ProjectRoot "06_交付物\ai_rag_annotation\annotation_workbook.csv"
}
if (-not $OutputDir) {
    $OutputDir = Join-Path $ProjectRoot "06_交付物\is_reduced_classification"
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

function Get-ClassCounts {
    param([object[]]$Rows)

    [ordered]@{
        "0" = @($Rows | Where-Object { [int]$_.is_reduced_label -eq 0 }).Count
        "1" = @($Rows | Where-Object { [int]$_.is_reduced_label -eq 1 }).Count
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

function Get-Sigmoid {
    param([double]$Z)

    if ($Z -gt 30) {
        return 1.0
    }
    if ($Z -lt -30) {
        return 0.0
    }
    return 1.0 / (1.0 + [Math]::Exp(-$Z))
}

function Get-LogisticProbability {
    param(
        [object]$Row,
        [string[]]$Names,
        [hashtable]$Means,
        [hashtable]$Stds,
        [double[]]$Weights
    )

    $x = Get-ScaledVector -Row $Row -Names $Names -Means $Means -Stds $Stds
    $z = $Weights[0]
    for ($i = 0; $i -lt $x.Count; $i += 1) {
        $z += $Weights[$i + 1] * $x[$i]
    }
    return Get-Sigmoid $z
}

function Train-LogisticRegression {
    param(
        [object[]]$Rows,
        [string[]]$Names,
        [hashtable]$Means,
        [hashtable]$Stds,
        [int]$Iterations,
        [double]$LearningRate,
        [double]$L2
    )

    $weights = New-Object double[] ($Names.Count + 1)
    $n = [double]$Rows.Count
    for ($iter = 0; $iter -lt $Iterations; $iter += 1) {
        $grad = New-Object double[] ($Names.Count + 1)
        foreach ($row in $Rows) {
            $x = Get-ScaledVector -Row $row -Names $Names -Means $Means -Stds $Stds
            $z = $weights[0]
            for ($i = 0; $i -lt $x.Count; $i += 1) {
                $z += $weights[$i + 1] * $x[$i]
            }
            $p = Get-Sigmoid $z
            $error = $p - [double]$row.is_reduced_label
            $grad[0] += $error
            for ($i = 0; $i -lt $x.Count; $i += 1) {
                $grad[$i + 1] += $error * $x[$i]
            }
        }
        $weights[0] -= $LearningRate * ($grad[0] / $n)
        for ($j = 1; $j -lt $weights.Count; $j += 1) {
            $regularized = ($grad[$j] / $n) + ($L2 * $weights[$j])
            $weights[$j] -= $LearningRate * $regularized
        }
    }
    return $weights
}

function Get-Auc {
    param([object[]]$Predictions)

    $pos = @($Predictions | Where-Object { [int]$_.actual -eq 1 })
    $neg = @($Predictions | Where-Object { [int]$_.actual -eq 0 })
    if ($pos.Count -eq 0 -or $neg.Count -eq 0) {
        return $null
    }

    $score = 0.0
    foreach ($p in $pos) {
        foreach ($n in $neg) {
            if ([double]$p.probability -gt [double]$n.probability) {
                $score += 1.0
            }
            elseif ([double]$p.probability -eq [double]$n.probability) {
                $score += 0.5
            }
        }
    }
    return $score / ([double]$pos.Count * [double]$neg.Count)
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
            positives = 0
            negatives = 0
            accuracy = ""
            precision = ""
            recall = ""
            f1 = ""
            roc_auc = ""
        }
    }

    $tp = @($Predictions | Where-Object { [int]$_.actual -eq 1 -and [int]$_.predicted -eq 1 }).Count
    $tn = @($Predictions | Where-Object { [int]$_.actual -eq 0 -and [int]$_.predicted -eq 0 }).Count
    $fp = @($Predictions | Where-Object { [int]$_.actual -eq 0 -and [int]$_.predicted -eq 1 }).Count
    $fn = @($Predictions | Where-Object { [int]$_.actual -eq 1 -and [int]$_.predicted -eq 0 }).Count
    $precision = $null
    if (($tp + $fp) -gt 0) {
        $precision = $tp / [double]($tp + $fp)
    }
    $recall = $null
    if (($tp + $fn) -gt 0) {
        $recall = $tp / [double]($tp + $fn)
    }
    $f1 = $null
    if ($null -ne $precision -and $null -ne $recall -and ($precision + $recall) -gt 0) {
        $f1 = 2 * $precision * $recall / ($precision + $recall)
    }

    [pscustomobject]@{
        model = $ModelName
        split = $SplitName
        n = $n
        positives = @($Predictions | Where-Object { [int]$_.actual -eq 1 }).Count
        negatives = @($Predictions | Where-Object { [int]$_.actual -eq 0 }).Count
        accuracy = [Math]::Round(($tp + $tn) / [double]$n, 4)
        precision = Format-NullableMetric $precision
        recall = Format-NullableMetric $recall
        f1 = Format-NullableMetric $f1
        roc_auc = Format-NullableMetric (Get-Auc $Predictions)
    }
}

function Add-PredictionRows {
    param(
        [System.Collections.Generic.List[object]]$Output,
        [object[]]$Rows,
        [string]$SplitName,
        [string]$ModelName,
        [scriptblock]$ProbabilityScript
    )

    foreach ($row in $Rows) {
        $prob = [double](& $ProbabilityScript $row)
        $predicted = [int]($prob -ge 0.5)
        [void]$Output.Add([pscustomobject]@{
            model = $ModelName
            split = $SplitName
            JID = $row.JID
            decision_year = $row.decision_year
            actual = [int]$row.is_reduced_label
            predicted = $predicted
            probability = [Math]::Round($prob, 6)
        })
    }
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
        labeled_rows = $Details.labeled_rows
        min_labeled_rows = $MinLabeledRows
        class_counts = $Details.class_counts
        model_features = $FeatureNames
        screening_only_features = $ScreeningOnlyFeatures
        split_policy = [ordered]@{
            train = "$TrainStartYear-$TrainEndYear"
            validation = "$ValidationYear"
            test = "$TestYear"
            latest_check = "$LatestCheckYear"
        }
        outputs = $Details.outputs
        note = "Only official/manual is_reduced labels are used by default. AI suggestions are features, not labels."
    }
    $statusPath = Join-Path $OutputDir "model_status.json"
    $status | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 -LiteralPath $statusPath

    $lines = @(
        "# is_reduced 分類模型狀態",
        "",
        "- 狀態：skipped",
        "- 原因：$Reason",
        ("- 輸入資料：" + $InputCsv),
        "- 全部案件：$($Details.total_rows)",
        ('- 已有人工作業標籤 `is_reduced`：' + $Details.labeled_rows),
        "- 最低訓練門檻：$MinLabeledRows",
        "",
        "## 已產出",
        "",
        '- `feature_matrix.csv`：全部待精標案件的模型特徵矩陣。',
        '- `labeled_feature_matrix.csv`：目前已有正式 `is_reduced` 標籤的子表。',
        '- `model_status.json`：機器可讀狀態。',
        "",
        "## 後續補標",
        "",
        '- 先在 `annotation_workbook.csv` 補 `is_reduced`，0 表示未酌減，1 表示酌減。',
        '- 建議同步補 `contract_price`、`claimed_penalty`、`delay_days` 與 issue 欄位。',
        "- 補標後重新執行本腳本，即可依 $TrainStartYear-$TrainEndYear 訓練、$ValidationYear 驗證、$TestYear 測試、$LatestCheckYear 最新年度檢查輸出回測指標。",
        "",
        "## 防呆說明",
        "",
        '- 本腳本預設只使用正式 `is_reduced` 欄位當目標，不把 AI 初標或關鍵詞命中當作真實標籤。',
        '- `x_has_discretion`、`x_has_mfa_252`、`x_has_over_high` 只保留於特徵矩陣與關鍵詞規則 baseline，預測未判決案件時應避免把判決結論語句當作輸入。'
    )
    $lines -join "`n" | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $OutputDir "分類模型狀態.md")
}

Write-Host "Reading annotation workbook: $InputCsv"
$rows = Import-Csv -Encoding UTF8 -LiteralPath $InputCsv
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$featureRows = New-Object System.Collections.Generic.List[object]
foreach ($row in $rows) {
    $decisionYear = Get-NumberOrNull (Get-Field $row "decision_year")
    $contractPrice = Get-NumberOrNull (Get-Field $row "contract_price")
    $delayDays = Get-NumberOrNull (Get-Field $row "delay_days")
    $claimedPenalty = Get-NumberOrNull (Get-Field $row "claimed_penalty")
    $allowedPenalty = Get-NumberOrNull (Get-Field $row "allowed_penalty")
    $label = Get-NullableLabel (Get-Field $row "is_reduced")
    $labelSource = ""
    if ($null -ne $label) {
        $labelSource = "manual_is_reduced"
    }

    $derivedLabel = $null
    if ($null -ne $claimedPenalty -and $null -ne $allowedPenalty -and $claimedPenalty -gt 0 -and $allowedPenalty -ge 0) {
        $derivedLabel = [int]($allowedPenalty -lt $claimedPenalty)
        if ($UseDerivedLabelFromAmounts -and $null -eq $label) {
            $label = $derivedLabel
            $labelSource = "derived_from_manual_amounts"
        }
    }

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
    $keywordRule = [int](($hasMfa252 -eq 1) -or ($hasDiscretion -eq 1) -or ($hasOverHigh -eq 1))

    [void]$featureRows.Add([pscustomobject][ordered]@{
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
        is_reduced = if ($null -eq $label) { "" } else { [int]$label }
        is_reduced_label = if ($null -eq $label) { "" } else { [int]$label }
        label_source = $labelSource
        derived_is_reduced_from_amounts = if ($null -eq $derivedLabel) { "" } else { [int]$derivedLabel }
        contract_price = if ($null -eq $contractPrice) { "" } else { [double]$contractPrice }
        delay_days = if ($null -eq $delayDays) { "" } else { [double]$delayDays }
        claimed_penalty = if ($null -eq $claimedPenalty) { "" } else { [double]$claimedPenalty }
        allowed_penalty = if ($null -eq $allowedPenalty) { "" } else { [double]$allowedPenalty }
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
        x_keyword_rule_predict_reduced = $keywordRule
    })
}

$featureMatrixPath = Join-Path $OutputDir "feature_matrix.csv"
$labeledMatrixPath = Join-Path $OutputDir "labeled_feature_matrix.csv"
$headers = @($featureRows[0].PSObject.Properties.Name)
$featureArray = @($featureRows | ForEach-Object { $_ })
$labeledRows = @($featureArray | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_.is_reduced_label) })
Export-ObjectCsv -Rows $featureArray -Path $featureMatrixPath -Headers $headers
Export-ObjectCsv -Rows $labeledRows -Path $labeledMatrixPath -Headers $headers

$classCounts = Get-ClassCounts $labeledRows
$outputs = [ordered]@{
    feature_matrix = $featureMatrixPath
    labeled_feature_matrix = $labeledMatrixPath
}

if ($labeledRows.Count -lt $MinLabeledRows) {
    Write-SkippedStatus -Reason "not_enough_official_is_reduced_labels" -Details @{
        total_rows = $featureArray.Count
        labeled_rows = $labeledRows.Count
        class_counts = $classCounts
        outputs = $outputs
    }
    Write-Host "Skipped model: only $($labeledRows.Count) official labels. Feature matrix written to $featureMatrixPath"
    exit 0
}

if ($classCounts["0"] -eq 0 -or $classCounts["1"] -eq 0) {
    Write-SkippedStatus -Reason "target_has_single_class" -Details @{
        total_rows = $featureArray.Count
        labeled_rows = $labeledRows.Count
        class_counts = $classCounts
        outputs = $outputs
    }
    Write-Host "Skipped model: is_reduced has a single class."
    exit 0
}

$trainRows = @($labeledRows | Where-Object { [int]$_.decision_year -ge $TrainStartYear -and [int]$_.decision_year -le $TrainEndYear })
$validationRows = @($labeledRows | Where-Object { [int]$_.decision_year -eq $ValidationYear })
$testRows = @($labeledRows | Where-Object { [int]$_.decision_year -eq $TestYear })
$latestRows = @($labeledRows | Where-Object { [int]$_.decision_year -eq $LatestCheckYear })
$trainClassCounts = Get-ClassCounts $trainRows

if ($trainRows.Count -lt 10 -or $trainClassCounts["0"] -eq 0 -or $trainClassCounts["1"] -eq 0) {
    Write-SkippedStatus -Reason "not_enough_two_class_training_rows_in_train_split" -Details @{
        total_rows = $featureArray.Count
        labeled_rows = $labeledRows.Count
        class_counts = $classCounts
        outputs = $outputs
    }
    Write-Host "Skipped model: training split has insufficient two-class labels."
    exit 0
}

$scaler = Get-Scaler -Rows $trainRows -Names $FeatureNames
$weights = Train-LogisticRegression -Rows $trainRows -Names $FeatureNames -Means $scaler.Means -Stds $scaler.Stds -Iterations $Iterations -LearningRate $LearningRate -L2 $L2
$majorityProb = $trainClassCounts["1"] / [double]($trainClassCounts["0"] + $trainClassCounts["1"])

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
    Add-PredictionRows -Output $predictions -Rows $split.Rows -SplitName $split.Name -ModelName "majority_baseline" -ProbabilityScript {
        param($r)
        return $majorityProb
    }
    Add-PredictionRows -Output $predictions -Rows $split.Rows -SplitName $split.Name -ModelName "keyword_rule_baseline" -ProbabilityScript {
        param($r)
        if ([int]$r.x_keyword_rule_predict_reduced -eq 1) {
            return 1.0
        }
        return 0.0
    }
    Add-PredictionRows -Output $predictions -Rows $split.Rows -SplitName $split.Name -ModelName "logistic_regression_l2" -ProbabilityScript {
        param($r)
        return Get-LogisticProbability -Row $r -Names $FeatureNames -Means $scaler.Means -Stds $scaler.Stds -Weights $weights
    }
}

$predictionRows = @($predictions | ForEach-Object { $_ })
$metrics = New-Object System.Collections.Generic.List[object]
foreach ($modelName in @("majority_baseline", "keyword_rule_baseline", "logistic_regression_l2")) {
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
[void]$coefs.Add([pscustomobject]@{
    feature = "intercept"
    mean = ""
    std = ""
    coefficient_scaled = [Math]::Round($weights[0], 8)
    direction = if ($weights[0] -gt 0) { "酌減同向" } elseif ($weights[0] -lt 0) { "未酌減同向" } else { "接近零" }
})
for ($i = 0; $i -lt $FeatureNames.Count; $i += 1) {
    $coef = $weights[$i + 1]
    [void]$coefs.Add([pscustomobject]@{
        feature = $FeatureNames[$i]
        mean = [Math]::Round($scaler.Means[$FeatureNames[$i]], 8)
        std = [Math]::Round($scaler.Stds[$FeatureNames[$i]], 8)
        coefficient_scaled = [Math]::Round($coef, 8)
        direction = if ($coef -gt 0) { "酌減同向" } elseif ($coef -lt 0) { "未酌減同向" } else { "接近零" }
    })
}
$coefs | Export-Csv -NoTypeInformation -Encoding UTF8 -LiteralPath $coefficientsPath

$status = [ordered]@{
    status = "ok"
    input_csv = $InputCsv
    total_rows = $featureArray.Count
    labeled_rows = $labeledRows.Count
    training_split = "$TrainStartYear-$TrainEndYear"
    validation_split = "$ValidationYear"
    test_split = "$TestYear"
    latest_check_split = "$LatestCheckYear"
    class_counts = $classCounts
    train_class_counts = $trainClassCounts
    model_features = $FeatureNames
    screening_only_features = $ScreeningOnlyFeatures
    outputs = [ordered]@{
        feature_matrix = $featureMatrixPath
        labeled_feature_matrix = $labeledMatrixPath
        metrics = $metricsPath
        predictions = $predictionsPath
        model_coefficients = $coefficientsPath
    }
    note = "The default logistic model excludes direct reduction keyword features; keyword terms are evaluated separately as a rule baseline."
}
$status | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $OutputDir "model_status.json")

$lines = @(
    "# is_reduced 分類模型回測結果",
    "",
    "## 設計",
    "",
    '- 目標變數：`is_reduced`。',
    "- 時間切分：$TrainStartYear-$TrainEndYear 訓練、$ValidationYear 驗證、$TestYear 測試、$LatestCheckYear 最新年度檢查。",
    "- 模型：多數類別 baseline、關鍵詞規則 baseline、L2 Logistic Regression。",
    "- 指標：Accuracy、Precision、Recall、F1-score、ROC-AUC。",
    "",
    "## 輸出",
    "",
    '- `feature_matrix.csv`',
    '- `labeled_feature_matrix.csv`',
    '- `metrics.csv`',
    '- `predictions.csv`',
    '- `model_coefficients.csv`',
    '- `model_status.json`',
    "",
    "## 注意",
    "",
    '- Logistic Regression 預設不使用 `x_has_discretion`、`x_has_mfa_252`、`x_has_over_high` 等較可能帶有判決結論的關鍵詞特徵。',
    "- 關鍵詞規則 baseline 只用來比較，不應被包裝成可直接預測未判決案件的模型。"
)
$lines -join "`n" | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $OutputDir "分類模型狀態.md")

Write-Host "Wrote feature matrix: $featureMatrixPath"
Write-Host "Wrote metrics: $metricsPath"
