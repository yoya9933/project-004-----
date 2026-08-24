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
$PythonScript = Join-Path $ProjectRoot "04_執行稿\run_reduction_ratio_model_sklearn.py"
$Args = @($PythonScript)
if ($InputCsv) { $Args += @("--usable-frame", $InputCsv) }
if ($OutputDir) { $Args += @("--output-dir", $OutputDir) }
if (
    $MinTargetRows -ne 30 -or $TrainStartYear -ne 2021 -or $TrainEndYear -ne 2023 -or
    $ValidationYear -ne 2024 -or $TestYear -ne 2025 -or $LatestCheckYear -ne 2026 -or
    $Iterations -ne 3000 -or $LearningRate -ne 0.03 -or $RidgeLambda -ne 0.05 -or
    $LassoLambda -ne 0.01
) {
    Write-Warning "Legacy training hyperparameters are ignored; model/search definitions are governed by legal_risk_modeling.models."
}
& python @Args
exit $LASTEXITCODE
