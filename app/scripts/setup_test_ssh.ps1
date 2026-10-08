param(
    [Parameter(Mandatory=$true)][string]$PublicKeyFile,
    [Parameter(Mandatory=$true)][string]$ClientAddress,
    [Parameter(Mandatory=$true)][string]$AccountName,
    [Parameter(Mandatory=$true)][string]$RunId,
    [switch]$VerifyOnly
)
$ErrorActionPreference = 'Stop'
$outputRoot = Join-Path $PSScriptRoot '..\outputs\Test-Bench'
$reportPath = Join-Path $outputRoot 'ssh-setup.json'
if ($VerifyOnly) {
    if (-not (Test-Path -LiteralPath $reportPath)) {
        throw 'Windows administrator setup did not publish a receipt. Check the UAC prompt or run the helper from an elevated session.'
    }
    $saved = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
    if ($saved.run_id -ne $RunId) { throw 'No receipt matches this SSH setup attempt. Check the Windows administrator prompt.' }
    if ($saved.status -eq 'running') { throw "Windows SSH setup is still running at '$($saved.stage)'; wait for it to finish before retrying." }
    if ($saved.status -ne 'completed') { throw "Windows SSH setup failed at '$($saved.stage)': $($saved.error)" }
    if ((Get-Service sshd).Status -ne 'Running') { throw 'The SSH service is no longer running.' }
    $saved | ConvertTo-Json
    exit 0
}
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$setupReport = @{version=1;run_id=$RunId;account_name=$AccountName;client_address=$ClientAddress;status='running';stage='validating';started_at=[DateTime]::UtcNow.ToString('o')}
function Save-Stage([string]$Stage, [string]$Status='running', [string]$ErrorMessage='') {
    $script:setupReport.stage = $Stage
    $script:setupReport.status = $Status
    $script:setupReport.updated_at = [DateTime]::UtcNow.ToString('o')
    if ($ErrorMessage) { $script:setupReport.error = $ErrorMessage }
    $script:setupReport | ConvertTo-Json | Set-Content -LiteralPath $reportPath -Encoding utf8
}
Save-Stage 'validating'
try {
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'OpenSSH setup requires one elevated Windows session.'
}
$address = $null
if (-not [Net.IPAddress]::TryParse($ClientAddress, [ref]$address)) {
    throw 'ClientAddress must be a single explicit IP address.'
}
if ($address.AddressFamily -ne [Net.Sockets.AddressFamily]::InterNetwork) {
    throw 'This setup helper currently accepts an IPv4 private LAN or Tailscale client.'
}
$bytes = $address.GetAddressBytes()
$private = $bytes[0] -eq 10 -or ($bytes[0] -eq 192 -and $bytes[1] -eq 168) -or
    ($bytes[0] -eq 172 -and $bytes[1] -ge 16 -and $bytes[1] -le 31) -or
    ($bytes[0] -eq 100 -and $bytes[1] -ge 64 -and $bytes[1] -le 127)
if (-not $private) { throw 'SSH setup is limited to private LAN or Tailscale clients.' }
if ($AccountName -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Use an existing local Windows account name.' }
$account = Get-LocalUser -Name $AccountName
if (-not $account.Enabled) { throw 'The selected local account is disabled.' }
$publicKey = (Get-Content -LiteralPath $PublicKeyFile -Raw).Trim()
if ($publicKey -notmatch '^ssh-ed25519 [A-Za-z0-9+/]+={0,2}( [^\r\n]+)?$') {
    throw 'Supply an Ed25519 PUBLIC key file containing exactly one key.'
}
Save-Stage 'installing'
$capability = Get-WindowsCapability -Online -Name 'OpenSSH.Server~~~~0.0.1.0'
$newInstall = $capability.State -ne 'Installed'
if ($newInstall) {
    $install = Add-WindowsCapability -Online -Name 'OpenSSH.Server~~~~0.0.1.0'
    if ($install.RestartNeeded) { throw 'Windows must restart to complete OpenSSH installation; run setup again afterwards.' }
    # Windows installation creates a broad rule. Replace that newly created
    # default with a rule for only the explicit controller address.
    Get-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -ErrorAction SilentlyContinue | Disable-NetFirewallRule
}
Save-Stage 'firewall'
$ruleName = 'Maestro-Test-SSH'
if (Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue) {
    Set-NetFirewallRule -Name $ruleName -Enabled True -Profile Private,Domain -RemoteAddress $ClientAddress
} else {
    New-NetFirewallRule -Name $ruleName -DisplayName 'Maestro test diagnostics SSH' -Enabled True -Direction Inbound -Action Allow -Protocol TCP -LocalPort 22 -RemoteAddress $ClientAddress -Profile Private,Domain | Out-Null
}
Save-Stage 'authorizing_key'
$adminSid = [Security.Principal.SecurityIdentifier]'S-1-5-32-544'
$admins = Get-LocalGroup -SID $adminSid
$adminMember = Get-LocalGroupMember -Group $admins | Where-Object { $_.SID -eq $account.SID }
if ($adminMember) {
    $keyPath = Join-Path $env:ProgramData 'ssh\administrators_authorized_keys'
} else {
    $profile = Get-CimInstance Win32_UserProfile | Where-Object { $_.SID -eq $account.SID.Value } | Select-Object -First 1
    if (-not $profile) { throw 'Sign in to the local account once to create its Windows profile, then retry.' }
    $keyPath = Join-Path $profile.LocalPath '.ssh\authorized_keys'
}
$keyDir = Split-Path -Parent $keyPath
New-Item -ItemType Directory -Path $keyDir -Force | Out-Null
$existing = if (Test-Path -LiteralPath $keyPath) { @(Get-Content -LiteralPath $keyPath) } else { @() }
if ($publicKey -notin $existing) { Add-Content -LiteralPath $keyPath -Value $publicKey -Encoding ascii }
$acl = New-Object Security.AccessControl.FileSecurity
$acl.SetAccessRuleProtection($true, $false)
$owners = if ($adminMember) { @($adminSid, [Security.Principal.SecurityIdentifier]'S-1-5-18') } else { @($account.SID, [Security.Principal.SecurityIdentifier]'S-1-5-18') }
foreach ($sid in $owners) {
    $acl.AddAccessRule((New-Object Security.AccessControl.FileSystemAccessRule($sid, 'FullControl', 'Allow')))
}
$acl.SetOwner($(if ($adminMember) { $adminSid } else { $account.SID }))
Set-Acl -LiteralPath $keyPath -AclObject $acl
Save-Stage 'starting_service'
Set-Service -Name sshd -StartupType Automatic
Start-Service sshd
$hostKeyPath = Join-Path $env:ProgramData 'ssh\ssh_host_ed25519_key.pub'
$report = @{
    version = 1; account_name = $AccountName; client_address = $ClientAddress
    sshd_running = (Get-Service sshd).Status -eq 'Running'
    authorized_keys_path = $keyPath; new_install = $newInstall
    host_public_key = (Get-Content -LiteralPath $hostKeyPath -Raw).Trim()
    firewall_rule = $ruleName; existing_firewall_rules_preserved = -not $newInstall
}
foreach ($key in $report.Keys) { $setupReport[$key] = $report[$key] }
Save-Stage 'completed' 'completed'
$setupReport | ConvertTo-Json
} catch {
    Save-Stage $setupReport.stage 'failed' $_.Exception.Message
    throw
}
