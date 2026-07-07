$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $PSCommandPath
$ProjectRoot = Split-Path -Parent $ScriptDir
$InputDir = Join-Path $ProjectRoot '05_測試與驗證\test'
$AnalysisDir = Join-Path $ProjectRoot '03_研究與分析'

$featurePatterns = [ordered]@{
  issue_payment = '工程款|承攬報酬|承攬費用|工程費|價金'
  issue_additional_or_change_work = '追加工程|追加工項|追加款|追加費用|追加施作|變更設計|變更施工|變更工項|加作|新增工程'
  issue_acceptance_or_settlement = '驗收|結算|估驗|計價'
  issue_delay_or_penalty = '逾期|遲延|違約金|罰款'
  issue_defect_or_repair = '瑕疵|修補|修繕|改善|缺失'
  issue_retention_or_warranty = '保留款|保固款|保固|保證金'
  issue_setoff_or_deduction = '抵銷|扣款|扣除|扣抵|扣留'
  issue_termination = '停工|終止|解除'
  issue_unjust_enrichment = '不當得利'
  issue_promissory_note = '本票'
  issue_appraisal = '鑑定|估價'
}

$featureDefinitions = [ordered]@{
  issue_payment = '工程款、承攬報酬爭議'
  issue_additional_or_change_work = '追加工程、變更設計'
  issue_acceptance_or_settlement = '驗收、結算、估驗'
  issue_delay_or_penalty = '逾期、遲延、違約金'
  issue_defect_or_repair = '瑕疵、修補'
  issue_retention_or_warranty = '保留款、保固款、保固責任'
  issue_setoff_or_deduction = '抵銷、扣款'
  issue_termination = '停工、終止、解除契約'
  issue_unjust_enrichment = '不當得利'
  issue_promissory_note = '本票相關'
  issue_appraisal = '鑑定金額或估價'
}

function Convert-ChineseMoneyToNumber {
  param([Parameter(Mandatory = $true)][string]$RawText)

  $s = $RawText
  $s = $s -replace '新臺幣|新台幣|臺幣|台幣|新幣|下同|整|元|圓|餘|約|計|共|合計|總計|金額|為|：|:|，|,|\s|\(|\)|（|）', ''
  if ([string]::IsNullOrWhiteSpace($s)) { return $null }

  if ($s -match '^[\d萬億]+$') {
    $remaining = $s
    $result = [int64]0
    if ($remaining -match '^(\d+)億') {
      $result += [int64]$Matches[1] * 100000000
      $remaining = $remaining -replace '^\d+億', ''
    }
    if ($remaining -match '^(\d+)萬') {
      $result += [int64]$Matches[1] * 10000
      $remaining = $remaining -replace '^\d+萬', ''
    }
    if ($remaining -match '^\d+$') {
      $result += [int64]$remaining
    }
    return $result
  }

  $digitMap = @{
    '零' = 0; '〇' = 0; '○' = 0; 'Ｏ' = 0; 'O' = 0
    '一' = 1; '壹' = 1; '二' = 2; '貳' = 2; '兩' = 2
    '三' = 3; '參' = 3; '四' = 4; '肆' = 4; '五' = 5; '伍' = 5
    '六' = 6; '陸' = 6; '七' = 7; '柒' = 7; '八' = 8; '捌' = 8
    '九' = 9; '玖' = 9
  }
  $smallUnitMap = @{
    '十' = 10; '拾' = 10; '百' = 100; '佰' = 100; '千' = 1000; '仟' = 1000
  }
  $largeUnitMap = @{
    '萬' = 10000; '亿' = 100000000; '億' = 100000000
  }

  $total = [int64]0
  $section = [int64]0
  $number = [int64]0
  $chars = $s.ToCharArray()

  foreach ($ch in $chars) {
    $c = [string]$ch
    if ($digitMap.ContainsKey($c)) {
      $number = [int64]$digitMap[$c]
    } elseif ($smallUnitMap.ContainsKey($c)) {
      if ($number -eq 0) { $number = 1 }
      $section += $number * [int64]$smallUnitMap[$c]
      $number = 0
    } elseif ($largeUnitMap.ContainsKey($c)) {
      $section += $number
      $total += $section * [int64]$largeUnitMap[$c]
      $section = 0
      $number = 0
    } elseif ($c -match '\d') {
      $number = [int64]$c
    }
  }

  $result = $total + $section + $number
  if ($result -eq 0) { return $null }
  return $result
}

function Get-Context {
  param(
    [string]$Text,
    [int]$Index,
    [int]$Length,
    [int]$Window = 90
  )

  $start = [Math]::Max(0, $Index - $Window)
  $end = [Math]::Min($Text.Length, $Index + $Length + $Window)
  return (($Text.Substring($start, $end - $start)) -replace "`r|`n", ' ' -replace '\s+', ' ').Trim()
}

$cnChars = '壹貳參肆伍陸柒捌玖零〇○ＯO一二三四五六七八九十拾百佰千仟萬億亿兩'
$amountRegex = [regex]"(?:新[臺台]幣|臺幣|台幣)?\s*(?:(?:\d+億)?\d+萬(?:\d{1,3}(?:,\d{3})*|\d+)?|\d{1,3}(?:,\d{3})+|\d{4,}|[$cnChars]{2,})\s*(?:元|圓)"
$longRows = New-Object System.Collections.Generic.List[object]
$seen = @{}

foreach ($file in Get-ChildItem -LiteralPath $InputDir -Filter *.json | Sort-Object Name) {
  $data = Get-Content -Raw -Encoding UTF8 -LiteralPath $file.FullName | ConvertFrom-Json
  $text = [string]$data.JFULL
  $text = $text -replace '(?<=[\d,])\s+(?=[\d,])', ''
  $text = $text -replace '(?<=\d)\s+(?=[萬億元圓])', ''
  $text = $text -replace '(?<=[萬億])\s+(?=\d)', ''
  $text = [regex]::Replace($text, "(?<=[${cnChars}])\s+(?=[${cnChars}元圓])", '')
  $matches = $amountRegex.Matches($text)

  foreach ($match in $matches) {
    $amountText = ($match.Value -replace '\s+', '')
    $amountValue = Convert-ChineseMoneyToNumber -RawText $amountText
    if ($null -eq $amountValue) { continue }
    if ($amountValue -lt 100) { continue }

    $context = Get-Context -Text $text -Index $match.Index -Length $match.Length
    foreach ($feature in $featurePatterns.Keys) {
      if ($context -match $featurePatterns[$feature]) {
        $key = "$($file.Name)|$feature|$amountValue|$amountText"
        if ($seen.ContainsKey($key)) { continue }
        $seen[$key] = $true
        $longRows.Add([pscustomobject][ordered]@{
          file = $file.Name
          jid = $data.JID
          title = $data.JTITLE
          feature = $feature
          definition = $featureDefinitions[$feature]
          amount_value = $amountValue
          amount_text = $amountText
          context = $context
        })
      }
    }
  }
}

$longPath = Join-Path $AnalysisDir 'money_feature_amounts_long.csv'
$widePath = Join-Path $AnalysisDir 'money_feature_amounts_wide.csv'
$summaryPath = Join-Path $AnalysisDir 'money_feature_amounts_summary.csv'

$longRows | Sort-Object file, feature, amount_value | Export-Csv -LiteralPath $longPath -NoTypeInformation -Encoding UTF8

$wideRows = foreach ($group in ($longRows | Group-Object file)) {
  $first = $group.Group[0]
  $obj = [ordered]@{
    file = $first.file
    jid = $first.jid
    title = $first.title
  }
  foreach ($feature in $featurePatterns.Keys) {
    $values = $group.Group |
      Where-Object { $_.feature -eq $feature } |
      Sort-Object amount_value -Unique |
      ForEach-Object { $_.amount_value }
    $obj[$feature] = ($values -join ';')
  }
  [pscustomobject]$obj
}
$wideRows | Sort-Object file | Export-Csv -LiteralPath $widePath -NoTypeInformation -Encoding UTF8

$summaryRows = foreach ($feature in $featurePatterns.Keys) {
  $featureRows = $longRows | Where-Object { $_.feature -eq $feature }
  $docs = @($featureRows | Select-Object -ExpandProperty file -Unique)
  $amounts = @($featureRows | Sort-Object amount_value -Unique | Select-Object -ExpandProperty amount_value)
  [pscustomobject][ordered]@{
    feature = $feature
    definition = $featureDefinitions[$feature]
    docs_with_amount = $docs.Count
    unique_amount_count = $amounts.Count
    min_amount = if ($amounts.Count) { ($amounts | Measure-Object -Minimum).Minimum } else { '' }
    max_amount = if ($amounts.Count) { ($amounts | Measure-Object -Maximum).Maximum } else { '' }
    amounts = ($amounts -join ';')
  }
}
$summaryRows | Export-Csv -LiteralPath $summaryPath -NoTypeInformation -Encoding UTF8

"Wrote $longPath"
"Wrote $widePath"
"Wrote $summaryPath"
$summaryRows | Format-Table -AutoSize | Out-String -Width 260



