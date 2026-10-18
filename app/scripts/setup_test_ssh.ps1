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
function Resolve-TargetAccount([string]$Name) {
    $localPattern = '^[A-Za-z0-9_.-]+$'
    $domainPattern = '^(?<domain>[A-Za-z0-9_.-]+)\\(?<account>[A-Za-z0-9_.-]+)$'
    if ($Name -match $localPattern) {
        try {
            $localAccount = Get-LocalUser -Name $Name -ErrorAction Stop
        } catch {
            throw "Could not find the existing local Windows account '$Name'. Confirm its name and try again."
        }
        if (-not $localAccount.Enabled) { throw 'The selected local account is disabled.' }
        $accountSid = $localAccount.SID
        $resolvedName = "$env:COMPUTERNAME\$($localAccount.Name)"
        return [pscustomobject]@{
            SID = $accountSid
            Name = $resolvedName
            Kind = 'local'
            Enabled = [bool]$localAccount.Enabled
            Profile = $null
        }
    }
    if ($Name -notmatch $domainPattern) {
        throw 'Use an existing local account name or one strict domain\account name with letters, digits, dot, underscore, or hyphen.'
    }

    $domain = $Matches['domain']
    $user = $Matches['account']
    $qualifiedName = "$domain\$user"
    if ($domain -ieq 'AzureAD') {
        throw 'Microsoft Entra ID (AzureAD) accounts are not supported by Windows OpenSSH key authentication. Use an Active Directory domain account or an existing local account.'
    }
    try {
        $accountSid = ([Security.Principal.NTAccount]::new($qualifiedName)).Translate([Security.Principal.SecurityIdentifier])
    } catch {
        throw "Could not resolve the existing domain account '$qualifiedName' to a Windows SID. Confirm the laptop can reach its domain and that the account exists."
    }
    if ($accountSid.Value -match '^S-1-12-1(?:-|$)') {
        throw 'Microsoft Entra ID accounts are not supported by Windows OpenSSH key authentication. Use an Active Directory domain account or an existing local account.'
    }

    try {
        $localAccounts = @(Get-LocalUser -ErrorAction Stop)
    } catch {
        throw "Could not inspect local Windows accounts while resolving '$qualifiedName': $($_.Exception.Message)"
    }
    $localMatch = $localAccounts | Where-Object { $_.SID.Value -eq $accountSid.Value } | Select-Object -First 1
    if ($localMatch) {
        if (-not $localMatch.Enabled) { throw 'The selected local account is disabled.' }
        $resolvedName = "$env:COMPUTERNAME\$($localMatch.Name)"
        $kind = 'local'
        $enabled = [bool]$localMatch.Enabled
    } else {
        try {
            $resolvedName = $accountSid.Translate([Security.Principal.NTAccount]).Value
        } catch {
            $resolvedName = $qualifiedName
        }
        $kind = 'domain'
        $enabled = $null
    }

    return [pscustomobject]@{
        SID = $accountSid
        Name = $resolvedName
        Kind = $kind
        Enabled = $enabled
        Profile = $null
    }
}
function Resolve-ExistingProfile([Security.Principal.SecurityIdentifier]$SID, [string]$Name) {
    try {
        $profiles = @(Get-CimInstance -ClassName Win32_UserProfile -ErrorAction Stop | Where-Object { $_.SID -eq $SID.Value })
    } catch {
        throw "Could not inspect Windows profiles for '$Name': $($_.Exception.Message)"
    }
    if ($profiles.Count -eq 0) {
        throw "No existing Windows profile was found for '$Name'. Sign in to that account once on this laptop, then retry."
    }
    if ($profiles.Count -gt 1) {
        throw "More than one Windows profile was found for '$Name'. Resolve the duplicate profile entries before retrying."
    }
    if ([string]::IsNullOrWhiteSpace([string]$profiles[0].LocalPath)) {
        throw "The existing Windows profile for '$Name' has no usable local path. Resolve the profile before retrying."
    }
    return $profiles[0]
}
function Resolve-AdministratorMembership(
    [Security.Principal.SecurityIdentifier]$SID,
    [Security.Principal.WindowsIdentity]$Identity,
    [Security.Principal.WindowsPrincipal]$Principal,
    [Security.Principal.SecurityIdentifier]$AdministratorsSID,
    [object]$AdministratorsGroup,
    [string]$Name
) {
    if ($Identity.User -and $Identity.User.Value -eq $SID.Value) {
        return [pscustomobject]@{
            IsAdministrator = [bool]$Principal.IsInRole($AdministratorsSID)
            Source = 'current_token'
        }
    }

    try {
        $members = @(Get-LocalGroupMember -Group $AdministratorsGroup -ErrorAction Stop)
    } catch {
        throw "Could not inspect local Administrators membership for '$Name': $($_.Exception.Message)"
    }
    $directMatch = @($members | Where-Object { $_.SID -and $_.SID.Value -eq $SID.Value })
    if ($directMatch.Count -gt 0) {
        return [pscustomobject]@{ IsAdministrator = $true; Source = 'direct_local_group' }
    }

    $nestedGroups = @($members | Where-Object { [string]$_.ObjectClass -eq 'Group' } | ForEach-Object { [string]$_.Name })
    if ($nestedGroups.Count -gt 0) {
        $groupList = $nestedGroups -join ', '
        throw "Could not safely determine whether '$Name' is an Administrator through nested group membership ($groupList). Run setup as that same elevated account, or resolve the listed group memberships first."
    }
    return [pscustomobject]@{ IsAdministrator = $false; Source = 'direct_local_group' }
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
$account = Resolve-TargetAccount $AccountName
$accountSid = $account.SID
$resolvedAccountName = $account.Name
$accountKind = $account.Kind
$sshLoginName = $resolvedAccountName.ToLowerInvariant()
$setupReport.resolved_account_name = $resolvedAccountName
$setupReport.account_sid = $accountSid.Value
$setupReport.account_kind = $accountKind
$setupReport.ssh_login_name = $sshLoginName
$setupReport.account_enabled = $account.Enabled
$setupReport.account_enabled_check = if ($accountKind -eq 'local') { 'checked' } else { 'not_checked_for_domain_account' }
Save-Stage 'validating'
$publicKey = (Get-Content -LiteralPath $PublicKeyFile -Raw).Trim()
if ($publicKey -notmatch '^ssh-ed25519 [A-Za-z0-9+/]+={0,2}( [^\r\n]+)?$') {
    throw 'Supply an Ed25519 PUBLIC key file containing exactly one key.'
}
$adminSid = [Security.Principal.SecurityIdentifier]'S-1-5-32-544'
$admins = Get-LocalGroup -SID $adminSid -ErrorAction Stop
$adminMembership = Resolve-AdministratorMembership $accountSid $identity $principal $adminSid $admins $resolvedAccountName
$adminMember = $adminMembership.IsAdministrator
$profile = $null
if ($accountKind -eq 'domain' -or -not $adminMember) {
    $profile = Resolve-ExistingProfile $accountSid $resolvedAccountName
}
if ($adminMember) {
    $keyPath = Join-Path $env:ProgramData 'ssh\administrators_authorized_keys'
} else {
    $keyPath = Join-Path $profile.LocalPath '.ssh\authorized_keys'
}
$setupReport.administrator = [bool]$adminMember
$setupReport.administrator_membership_source = $adminMembership.Source
$setupReport.account_profile_path = $(if ($profile) { $profile.LocalPath } else { $null })
$setupReport.authorized_keys_path = $keyPath
Save-Stage 'validating'
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
$keyDir = Split-Path -Parent $keyPath
New-Item -ItemType Directory -Path $keyDir -Force | Out-Null
$existing = if (Test-Path -LiteralPath $keyPath) { @(Get-Content -LiteralPath $keyPath) } else { @() }
if ($publicKey -notin $existing) { Add-Content -LiteralPath $keyPath -Value $publicKey -Encoding ascii }
$acl = New-Object Security.AccessControl.FileSecurity
$acl.SetAccessRuleProtection($true, $false)
$owners = if ($adminMember) { @($adminSid, [Security.Principal.SecurityIdentifier]'S-1-5-18') } else { @($accountSid, [Security.Principal.SecurityIdentifier]'S-1-5-18') }
foreach ($sid in $owners) {
    $acl.AddAccessRule((New-Object Security.AccessControl.FileSystemAccessRule($sid, 'FullControl', 'Allow')))
}
$acl.SetOwner($(if ($adminMember) { $adminSid } else { $accountSid }))
Set-Acl -LiteralPath $keyPath -AclObject $acl
Save-Stage 'starting_service'
Set-Service -Name sshd -StartupType Automatic
Start-Service sshd
$hostKeyPath = Join-Path $env:ProgramData 'ssh\ssh_host_ed25519_key.pub'
$report = @{
    version = 1; account_name = $AccountName; resolved_account_name = $resolvedAccountName
    account_sid = $accountSid.Value; account_kind = $accountKind; ssh_login_name = $sshLoginName
    account_enabled = $account.Enabled
    account_enabled_check = $(if ($accountKind -eq 'local') { 'checked' } else { 'not_checked_for_domain_account' })
    account_profile_path = $(if ($profile) { $profile.LocalPath } else { $null })
    administrator = [bool]$adminMember; administrator_membership_source = $adminMembership.Source
    client_address = $ClientAddress
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
