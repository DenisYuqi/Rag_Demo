[CmdletBinding()]
param(
    [string]$ResourceGroup,
    [string]$Location = "southeastasia",
    [string]$VmName = "vm-rag-mvp",
    [string]$VmSize = "Standard_D2s_v5",
    [string]$AdminUser = "azureuser",
    [string]$DnsLabel,
    [string]$ProviderEnvFile = ".env"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

function Invoke-NativeTwice {
    param([string]$Description, [string]$File, [string[]]$Arguments)
    for ($attempt = 1; $attempt -le 2; $attempt++) {
        & $File @Arguments
        if ($LASTEXITCODE -eq 0) { return }
        if ($attempt -eq 2) { throw "$Description failed twice; stopping." }
        Write-Warning "$Description failed once; retrying."
        Start-Sleep -Seconds 5
    }
}

function Invoke-SshTwice {
    param([string]$Description, [string[]]$Arguments)
    Invoke-NativeTwice -Description $Description -File "ssh" -Arguments $Arguments
}

if (-not $ResourceGroup) {
    $suffix = (Get-Date).ToUniversalTime().ToString("yyyyMMddHHmm")
    $ResourceGroup = "rg-rag-mvp-$suffix"
}
if (-not $DnsLabel) {
    $compact = ($ResourceGroup.ToLowerInvariant() -replace '[^a-z0-9-]', '-')
    $DnsLabel = "$compact-app"
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$sourceRevision = (git -C $repoRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $sourceRevision -notmatch '^[0-9a-f]{40}$') {
    throw "Unable to resolve the source revision."
}

$account = az account show --output json | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or $account.state -ne "Enabled") {
    throw "Azure subscription is unavailable or disabled."
}
$cloud = az cloud show --output json | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or $cloud.name -ne "AzureCloud") {
    throw "Azure CLI must target AzureCloud."
}

$exists = az group exists --name $ResourceGroup --output tsv
if ($LASTEXITCODE -ne 0) { throw "Unable to check resource-group availability." }
if ($exists.Trim().ToLowerInvariant() -eq "true") {
    throw "Resource group '$ResourceGroup' already exists; refusing to mutate it."
}

$sku = az vm list-sizes --location $Location --query "[?name=='$VmSize']" --output json |
    ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or @($sku).Count -eq 0) {
    throw "VM size '$VmSize' is not listed in '$Location'."
}

$providerLine = Get-Content -LiteralPath (Join-Path $repoRoot $ProviderEnvFile) |
    Where-Object { $_ -match '^\s*RAG_MVP_OPENAI_API_KEY\s*=' } |
    Select-Object -Last 1
if (-not $providerLine) { throw "RAG_MVP_OPENAI_API_KEY is missing from $ProviderEnvFile." }
$providerKey = ($providerLine -split '=', 2)[1].Trim()
if (($providerKey.StartsWith('"') -and $providerKey.EndsWith('"')) -or
    ($providerKey.StartsWith("'") -and $providerKey.EndsWith("'"))) {
    $providerKey = $providerKey.Substring(1, $providerKey.Length - 2)
}
if ([string]::IsNullOrWhiteSpace($providerKey)) { throw "Provider key is empty." }

$deploymentRoot = Join-Path $env:LOCALAPPDATA "rag-mvp\azure-vm\$ResourceGroup"
New-Item -ItemType Directory -Force -Path $deploymentRoot | Out-Null
$keyPath = Join-Path $deploymentRoot "id_ed25519"
if (-not (Test-Path -LiteralPath $keyPath)) {
    Invoke-NativeTwice -Description "SSH key generation" -File "ssh-keygen" -Arguments @(
        "-q", "-t", "ed25519", "-N", "", "-C", $ResourceGroup, "-f", $keyPath
    )
}

$clientIp = (Invoke-RestMethod -Uri "https://api.ipify.org").Trim()
if ($clientIp -notmatch '^\d{1,3}(\.\d{1,3}){3}$') {
    throw "Unable to determine a safe IPv4 SSH source."
}

$publicIpName = "pip-rag-mvp"
$nsgName = "nsg-rag-mvp"
$vnetName = "vnet-rag-mvp"
$subnetName = "default"
$nicName = "nic-rag-mvp"

foreach ($provider in @("Microsoft.Compute", "Microsoft.Network")) {
    $providerState = az provider show --namespace $provider --query registrationState --output tsv
    if ($LASTEXITCODE -ne 0) { throw "Unable to inspect provider '$provider'." }
    if ($providerState -ne "Registered") {
        Invoke-NativeTwice "$provider registration" "az" @(
            "provider", "register", "--namespace", $provider, "--wait", "--output", "none"
        )
    }
}

Invoke-NativeTwice "resource group creation" "az" @(
    "group", "create", "--name", $ResourceGroup, "--location", $Location, "--output", "none"
)
Invoke-NativeTwice "public IP creation" "az" @(
    "network", "public-ip", "create", "--resource-group", $ResourceGroup,
    "--name", $publicIpName, "--sku", "Standard", "--allocation-method", "Static",
    "--dns-name", $DnsLabel, "--output", "none"
)
Invoke-NativeTwice "network security group creation" "az" @(
    "network", "nsg", "create", "--resource-group", $ResourceGroup,
    "--name", $nsgName, "--output", "none"
)
Invoke-NativeTwice "restricted SSH rule creation" "az" @(
    "network", "nsg", "rule", "create", "--resource-group", $ResourceGroup,
    "--nsg-name", $nsgName, "--name", "AllowSshFromOperator", "--priority", "100",
    "--source-address-prefixes", "$clientIp/32", "--destination-port-ranges", "22",
    "--access", "Allow", "--protocol", "Tcp", "--direction", "Inbound", "--output", "none"
)
foreach ($webRule in @(@("AllowHttp", "110", "80"), @("AllowHttps", "120", "443"))) {
    Invoke-NativeTwice "$($webRule[0]) rule creation" "az" @(
        "network", "nsg", "rule", "create", "--resource-group", $ResourceGroup,
        "--nsg-name", $nsgName, "--name", $webRule[0], "--priority", $webRule[1],
        "--source-address-prefixes", "Internet", "--destination-port-ranges", $webRule[2],
        "--access", "Allow", "--protocol", "Tcp", "--direction", "Inbound", "--output", "none"
    )
}
Invoke-NativeTwice "virtual network creation" "az" @(
    "network", "vnet", "create", "--resource-group", $ResourceGroup, "--name", $vnetName,
    "--address-prefixes", "10.42.0.0/16", "--subnet-name", $subnetName,
    "--subnet-prefixes", "10.42.1.0/24", "--output", "none"
)
Invoke-NativeTwice "network interface creation" "az" @(
    "network", "nic", "create", "--resource-group", $ResourceGroup, "--name", $nicName,
    "--vnet-name", $vnetName, "--subnet", $subnetName, "--network-security-group", $nsgName,
    "--public-ip-address", $publicIpName, "--output", "none"
)
Invoke-NativeTwice "virtual machine creation" "az" @(
    "vm", "create", "--resource-group", $ResourceGroup, "--name", $VmName,
    "--nics", $nicName, "--image", "Ubuntu2204", "--size", $VmSize,
    "--admin-username", $AdminUser, "--ssh-key-values", "$keyPath.pub",
    "--os-disk-size-gb", "128", "--storage-sku", "Premium_LRS", "--output", "none"
)

$publicHost = az network public-ip show --resource-group $ResourceGroup --name $publicIpName `
    --query dnsSettings.fqdn --output tsv
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($publicHost)) {
    throw "Unable to resolve the VM public hostname."
}

$knownHosts = Join-Path $deploymentRoot "known_hosts"
$sshBase = @(
    "-i", $keyPath, "-o", "StrictHostKeyChecking=accept-new",
    "-o", "UserKnownHostsFile=$knownHosts", "$AdminUser@$publicHost"
)
Invoke-SshTwice "SSH reachability" ($sshBase + @("true"))

$archivePath = Join-Path $deploymentRoot "rag-mvp-source.tgz"
Push-Location $repoRoot
try {
    Invoke-NativeTwice "source archive creation" "tar" @(
        "-czf", $archivePath, ".dockerignore", "Dockerfile", "compose.yaml",
        "pyproject.toml", "uv.lock", "src", "deploy/azure-vm"
    )
} finally {
    Pop-Location
}

Invoke-NativeTwice "bootstrap upload" "scp" @(
    "-i", $keyPath, "-o", "StrictHostKeyChecking=accept-new",
    "-o", "UserKnownHostsFile=$knownHosts", (Join-Path $repoRoot "deploy\azure-vm\bootstrap.sh"),
    (Join-Path $repoRoot "deploy\azure-vm\configure.sh"),
    (Join-Path $repoRoot "deploy\azure-vm\verify.sh"), $archivePath,
    "${AdminUser}@${publicHost}:/tmp/"
)
Invoke-SshTwice "Docker bootstrap" ($sshBase + @("sudo bash /tmp/bootstrap.sh $AdminUser"))
Invoke-SshTwice "source extraction" ($sshBase + @(
    "sudo tar -xzf /tmp/rag-mvp-source.tgz -C /opt/rag-mvp/app && " +
    "sudo chown -R ${AdminUser}:${AdminUser} /opt/rag-mvp/app"
))

$sshInfo = [System.Diagnostics.ProcessStartInfo]::new()
$sshInfo.FileName = "ssh"
$sshInfo.UseShellExecute = $false
$sshInfo.RedirectStandardInput = $true
foreach ($item in ($sshBase + @(
    "sudo tee /opt/rag-mvp/secrets/provider-key >/dev/null && " +
    "sudo chmod 0400 /opt/rag-mvp/secrets/provider-key"
))) { [void]$sshInfo.ArgumentList.Add($item) }
$sshProcess = [System.Diagnostics.Process]::Start($sshInfo)
$sshProcess.StandardInput.WriteLine($providerKey)
$sshProcess.StandardInput.Close()
$sshProcess.WaitForExit()
$providerKey = $null
if ($sshProcess.ExitCode -ne 0) { throw "Provider-secret transfer failed." }

Invoke-SshTwice "runtime configuration" ($sshBase + @(
    "sudo bash /tmp/configure.sh $publicHost $sourceRevision && " +
    "sudo chown -R ${AdminUser}:${AdminUser} /opt/rag-mvp/app"
))
$remoteCompose = "cd /opt/rag-mvp/app && docker compose --env-file deployment.env " +
    "-f compose.yaml -f deploy/azure-vm/compose.azure.yaml"
Invoke-SshTwice "container build" ($sshBase + @("$remoteCompose build"))
Invoke-SshTwice "container startup" ($sshBase + @("$remoteCompose up -d"))
Invoke-SshTwice "deployment verification" ($sshBase + @(
    "sudo bash /tmp/verify.sh $publicHost"
))

[ordered]@{
    resource_group = $ResourceGroup
    location = $Location
    vm_name = $VmName
    vm_size = $VmSize
    public_url = "https://$publicHost"
    ssh_private_key = $keyPath
    basic_auth_remote_file = "/opt/rag-mvp/secrets/basic-auth.txt"
    source_revision = $sourceRevision
} | ConvertTo-Json
