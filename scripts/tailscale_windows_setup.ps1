[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$TailscalePath,

    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 65535)]
    [int]$Port,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$UserId
)

$ErrorActionPreference = "Stop"
$taskName = "Maestro Tailscale Serve"
$target = "http://127.0.0.1:$Port"
$serveArguments = "serve --bg --yes --https=443 $target"

if (-not (Test-Path -LiteralPath $TailscalePath -PathType Leaf)) {
    throw "Tailscale executable was not found at '$TailscalePath'."
}

# Configure the route immediately while this one-time setup process has the
# user's explicit administrator approval. This also surfaces any Tailscale
# HTTPS/tailnet consent failure before Maestro records setup as complete.
& $TailscalePath serve --bg --yes --https=443 $target
if ($LASTEXITCODE -ne 0) {
    throw "Tailscale Serve setup failed with exit code $LASTEXITCODE."
}

# Windows requires an elevated token for Tailscale Serve configuration. Store
# one tightly scoped, on-demand task whose action is the protected Tailscale
# executable plus Maestro's fixed loopback target. Maestro can request this
# task after later restarts without showing another UAC prompt. The task has no
# timer or login trigger and runs only when an opted-in Maestro start requests
# it.
$action = New-ScheduledTaskAction `
    -Execute $TailscalePath `
    -Argument $serveArguments
$principal = New-ScheduledTaskPrincipal `
    -UserId $UserId `
    -LogonType Interactive `
    -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 2) `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Principal $principal `
    -Settings $settings `
    -Description "Restores Maestro's user-enabled private Tailscale Serve route when Maestro starts." `
    -Force | Out-Null

# Confirm the stored action can be dispatched. It is harmless to reapply the
# same background Serve route that was configured above.
Start-ScheduledTask -TaskName $taskName
Write-Output "Maestro private access is configured for $target and restart-safe restoration is ready."
