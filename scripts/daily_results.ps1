[CmdletBinding()]
param(
    [switch] $RefreshFilled,
    [switch] $MainAllDates
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

New-Item -ItemType Directory -Force logs | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogPath = Join-Path $ProjectRoot "logs\daily-results-$Stamp.log"

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
    $ResultArguments = @("statbirt.update_results")
    if ($RefreshFilled) {
        $ResultArguments += "--refresh-filled"
    }

    Invoke-StatbirtStep `
        -Name "Postgame result update" `
        -Arguments $ResultArguments

    Invoke-StatbirtStep `
        -Name "Learned prediction result refresh" `
        -Arguments @("statbirt.learned_model", "score", "--date", "latest", "--top", "25")

    if ($MainAllDates) {
        $DashboardArguments = @("statbirt.export_web", "--all-dates", "--limit", "10")
    } else {
        $DashboardArguments = @("statbirt.export_web", "--date", "latest", "--limit", "10")
    }

    Invoke-StatbirtStep `
        -Name "Dashboard export" `
        -Arguments $DashboardArguments

    Invoke-StatbirtStep `
        -Name "Learned shortlist dashboard export" `
        -Arguments @("statbirt.export_learned_web", "--all-dates", "--limit", "5")

    Write-Host ""
    Write-Host "[$(Get-Date -Format o)] Daily results run complete."
}
finally {
    Stop-Transcript
}
