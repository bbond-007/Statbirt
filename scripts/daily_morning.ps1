Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = "X:\Coding\Statbirt"
Set-Location $ProjectRoot

New-Item -ItemType Directory -Force logs | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogPath = Join-Path $ProjectRoot "logs\daily-morning-$Stamp.log"

function Invoke-StatbirtStep {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Name,

        [Parameter(Mandatory = $true)]
        [string[]] $Arguments
    )

    Write-Host ""
    Write-Host "[$(Get-Date -Format o)] Starting: $Name"
    & py -3 -m @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE."
    }
    Write-Host "[$(Get-Date -Format o)] Finished: $Name"
}

Start-Transcript -Path $LogPath

try {
    $Today = Get-Date -Format "yyyy-MM-dd"

    Invoke-StatbirtStep `
        -Name "Primary daily model" `
        -Arguments @("statbirt.cli", "--date", $Today, "--top", "25", "--skip-fangraphs-fetch")

    Invoke-StatbirtStep `
        -Name "Frozen production learned model" `
        -Arguments @(
            "statbirt.learned_model",
            "score",
            "--model",
            "models\learned-logistic-v2-20260717T120305Z.json",
            "--date",
            "latest",
            "--top",
            "25"
        )

    Invoke-StatbirtStep `
        -Name "Learned shadow model" `
        -Arguments @("statbirt.learned_shadow", "run", "--date", "latest")

    Invoke-StatbirtStep `
        -Name "Immutable pregame decision snapshot" `
        -Arguments @(
            "statbirt.prediction_ledger",
            "snapshot",
            "--run-id",
            "daily-morning-$Stamp",
            "--target-date",
            "latest",
            "--shadow-predictions",
            "data\learned_shadow_predictions.csv"
        )

    Invoke-StatbirtStep `
        -Name "Decision ledger audit" `
        -Arguments @("statbirt.prediction_ledger", "audit")

    Invoke-StatbirtStep `
        -Name "Dashboard export" `
        -Arguments @("statbirt.export_web", "--all-dates", "--limit", "10")

    Invoke-StatbirtStep `
        -Name "Learned shortlist dashboard export" `
        -Arguments @("statbirt.export_learned_web", "--all-dates", "--limit", "5")

    Write-Host ""
    Write-Host "[$(Get-Date -Format o)] Daily morning run complete."
}
finally {
    Stop-Transcript
}
