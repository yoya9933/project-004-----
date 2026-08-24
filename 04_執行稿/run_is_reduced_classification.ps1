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
$PythonScript = Join-Path $ProjectRoot "04_執行稿\run_is_reduced_classification.py"
$Args = @(
    $PythonScript,
    "--min-labeled-rows", "$MinLabeledRows",
    "--l2", "$L2",
    "--train-start-year", "$TrainStartYear",
    "--train-end-year", "$TrainEndYear",
    "--validation-year", "$ValidationYear",
    "--test-year", "$TestYear",
    "--latest-check-year", "$LatestCheckYear"
)
if ($InputCsv) { $Args += @("--input-csv", $InputCsv) }
if ($OutputDir) { $Args += @("--output-dir", $OutputDir) }
if ($UseDerivedLabelFromAmounts) { $Args += "--use-derived-label-from-amounts" }
if ($Iterations -ne 2500 -or $LearningRate -ne 0.05) {
    Write-Warning "Iterations/LearningRate are legacy parameters; sklearn training is governed in the Python package."
}
& python @Args
exit $LASTEXITCODE
