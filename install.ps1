# Xencode Windows Installer (Rust)
# ================================

param(
    [switch]$Release,       # download a published cargo-dist archive instead of building
    [switch]$Source,        # build from source (the default)
    [string]$Version = $env:XENCODE_RELEASE_VERSION,
    [switch]$DryRun,        # print the release plan and stop before any network request
    [switch]$Help
)

$ErrorActionPreference = "Stop"

if ($Help) {
    Write-Host @'
Usage: .\install.ps1 [options]

  -Release              Download a published release archive from GitHub
                        Releases instead of building from source.
  -Source               Build from source (the default).
  -Version VERSION      Release version to download (default: 0.1.0).
                        Also settable with XENCODE_RELEASE_VERSION.
  -DryRun               Show the derived asset name and URLs, then stop before
                        any network request. Implies -Release.
  -Help                 Show this help.

No GitHub Release exists yet: the v0.1.0 tag has not been pushed, so release
mode currently fails with HTTP 404 until that release is published.
'@
    exit 0
}

Write-Host "🚀 Installing Xencode for Windows" -ForegroundColor Cyan
Write-Host "============================"

# --- Configuration ---
$AppName = "xencode"
$InstallDir = "$env:LOCALAPPDATA\xencode"
$RepoUrl = "https://github.com/sreevarshan-xenoz/xencode.git"
$ReleaseOwner = "sreevarshan-xenoz"
$ReleaseRepo = "xencode"
$ReleaseWorkDir = $null

if ([string]::IsNullOrWhiteSpace($Version)) { $Version = "0.1.0" }

$InstallMode = "source"
if ($Release -or $DryRun -or $env:XENCODE_INSTALL_MODE -eq "release") { $InstallMode = "release" }
if ($Source) { $InstallMode = "source" }

Write-Host "   📦 Mode: $InstallMode" -ForegroundColor Gray
if ($InstallMode -eq "release") { Write-Host "   🏷️  Release version: v$Version" -ForegroundColor Gray }

# --- Release asset derivation (mirrors scripts/lib/release-assets.sh) ---
function Get-XencodeReleaseTarget {
    param([string]$OS, [string]$Arch)
    $osLower = $OS.ToLowerInvariant()
    $archLower = $Arch.ToLowerInvariant()
    if ($osLower -notlike 'mingw*' -and $osLower -notlike 'msys*' -and
        $osLower -notlike 'cygwin*' -and $osLower -notlike 'windows*') {
        return $null
    }
    switch ($archLower) {
        'x86_64' { return 'x86_64-pc-windows-msvc' }
        'amd64'  { return 'x86_64-pc-windows-msvc' }
        default  { return $null }
    }
}

function Get-XencodeReleaseInfo {
    param([string]$OS, [string]$Arch, [string]$RelVersion)
    $target = Get-XencodeReleaseTarget -OS $OS -Arch $Arch
    if (-not $target) {
        throw "No prebuilt release archive for '$OS / $Arch'. Re-run without -Release to build from source."
    }
    $asset = "xencode-cli-$target.zip"
    $base = "https://github.com/$ReleaseOwner/$ReleaseRepo/releases/download/v$RelVersion"
    return [pscustomobject]@{
        Target      = $target
        Asset       = $asset
        AssetUrl    = "$base/$asset"
        ChecksumUrl = "$base/$asset.sha256"
    }
}

# --- 1. System Checks ---
Write-Host "`n1. 🔍 System Checks" -ForegroundColor Yellow

if ($InstallMode -eq "source") {
    # Check Rust toolchain (needed to build xencode)
    if (Get-Command "cargo" -ErrorAction SilentlyContinue) {
        $CargoVer = cargo --version 2>&1
        Write-Host "   ✅ Rust found: $CargoVer" -ForegroundColor Green
    } else {
        Write-Host "   ❌ cargo not found. Install Rust from https://rustup.rs first." -ForegroundColor Red
        Write-Host "      Then re-run this installer (rustup installs cargo + rustc)." -ForegroundColor Gray
        exit 1
    }

    # Check Git
    if (Get-Command "git" -ErrorAction SilentlyContinue) {
        Write-Host "   ✅ Git found" -ForegroundColor Green
    } else {
        Write-Host "   ❌ Git not found. Please install Git for Windows." -ForegroundColor Red
        exit 1
    }
}

# Check Ollama (Warning only)
if (Get-Command "ollama" -ErrorAction SilentlyContinue) {
    Write-Host "   ✅ Ollama found" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  Ollama not found. Xencode needs Ollama for AI features." -ForegroundColor DarkYellow
    Write-Host "      You can install it later from https://ollama.ai" -ForegroundColor Gray
}

# --- 2. Obtain the binary ---
if ($InstallMode -eq "release") {
    Write-Host "`n2. 📥 Downloading Xencode (release v$Version)" -ForegroundColor Yellow
    try {
        $info = Get-XencodeReleaseInfo -OS $env:OS -Arch $env:PROCESSOR_ARCHITECTURE -RelVersion $Version
        Write-Host "   🖥️  Host: $($env:OS) / $($env:PROCESSOR_ARCHITECTURE)" -ForegroundColor Gray
        Write-Host "   📄 Archive:  $($info.Asset)" -ForegroundColor Gray
        Write-Host "   🔗 Download: $($info.AssetUrl)" -ForegroundColor Gray
        Write-Host "   🔐 Checksum: $($info.ChecksumUrl)" -ForegroundColor Gray
        Write-Host ""
        Write-Host "   ⚠️  There is no GitHub Release yet: the v$Version tag has not been" -ForegroundColor DarkYellow
        Write-Host "      pushed, so these URLs return 404 until the release is published." -ForegroundColor DarkYellow
        Write-Host "      Nothing is fetched from a mirror or a fabricated release." -ForegroundColor DarkYellow
        Write-Host ""
        if ($DryRun) {
            Write-Host "   🔎 Dry run: no network request was made." -ForegroundColor Green
            exit 0
        }

        $ReleaseWorkDir = Join-Path ([System.IO.Path]::GetTempPath()) ("xencode-release-" + [guid]::NewGuid().ToString('N'))
        New-Item -Path $ReleaseWorkDir -ItemType Directory -Force | Out-Null
        $archivePath = Join-Path $ReleaseWorkDir $info.Asset
        $checksumPath = "$archivePath.sha256"
        Write-Host "   ⏳ Fetching the archive and its .sha256 checksum..." -ForegroundColor Gray
        Invoke-WebRequest -Uri $info.AssetUrl -OutFile $archivePath -UseBasicParsing
        Invoke-WebRequest -Uri $info.ChecksumUrl -OutFile $checksumPath -UseBasicParsing

        $firstLine = (Get-Content -Path $checksumPath -TotalCount 1).Trim()
        $expected = ($firstLine -split '\s+')[0]
        if ($expected -notmatch '^[0-9a-fA-F]{64}$') {
            throw "Checksum file did not contain a 64-character SHA-256 digest."
        }
        Write-Host "   🔍 Verifying the SHA-256 checksum..." -ForegroundColor Gray
        $actual = (Get-FileHash -Algorithm SHA256 -Path $archivePath).Hash
        if ($actual -ine $expected) {
            throw "Checksum mismatch: expected $expected but the download hashed to $actual."
        }
        Write-Host "   ✅ Checksum verified" -ForegroundColor Green

        Write-Host "   📂 Extracting $($info.Asset)..." -ForegroundColor Gray
        Expand-Archive -Path $archivePath -DestinationPath $ReleaseWorkDir -Force
        $BuiltExe = Join-Path $ReleaseWorkDir "xencode.exe"
        if (-not (Test-Path $BuiltExe)) {
            throw "The archive did not contain an executable named xencode.exe."
        }
        Write-Host "   ✅ Verified and extracted" -ForegroundColor Green
    } catch {
        if ($ReleaseWorkDir) {
            Remove-Item -Path $ReleaseWorkDir -Recurse -Force -ErrorAction SilentlyContinue
        }
        Write-Host "   ❌ $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "`n2. 🏗️  Building Xencode (release)" -ForegroundColor Yellow

    if (-not (Test-Path "rust\Cargo.toml")) {
        Write-Host "   ❌ rust\Cargo.toml not found. Run install.ps1 from the repo root." -ForegroundColor Red
        exit 1
    }

    Write-Host "   ⏳ cargo build --release -p xencode-cli (first build takes a few minutes)..." -ForegroundColor Gray
    cargo build --release -p xencode-cli --manifest-path rust\Cargo.toml

    $BuiltExe = "rust\target\release\xencode.exe"
    if (-not (Test-Path $BuiltExe)) {
        Write-Host "   ❌ Build failed: $BuiltExe not found." -ForegroundColor Red
        exit 1
    }
    Write-Host "   ✅ Build succeeded" -ForegroundColor Green
}

# --- 3. Install ---
Write-Host "`n3. 📦 Installing" -ForegroundColor Yellow

if (Test-Path $InstallDir) {
    Write-Host "   📂 Cleaning existing installation directory..." -ForegroundColor Gray
    Remove-Item -Path $InstallDir -Recurse -Force
}
New-Item -Path $InstallDir -ItemType Directory -Force | Out-Null

Copy-Item $BuiltExe "$InstallDir\xencode.exe"
Write-Host "   ✅ Installed: $InstallDir\xencode.exe" -ForegroundColor Green

if ($ReleaseWorkDir) {
    Remove-Item -Path $ReleaseWorkDir -Recurse -Force -ErrorAction SilentlyContinue
    $ReleaseWorkDir = $null
}

# --- 4. CLI Integration ---
Write-Host "`n4. 🔌 CLI Integration" -ForegroundColor Yellow

$BatPath = "$InstallDir\xencode.bat"
Set-Content -Path $BatPath -Value "@echo off`r`n`"%~dp0xencode.exe`" %*"
Write-Host "   ✅ Created CLI shim: $BatPath" -ForegroundColor Green

# Add to PATH
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($UserPath -notlike "*$InstallDir*") {
    Write-Host "   🔗 Adding to User PATH..." -ForegroundColor Gray
    [Environment]::SetEnvironmentVariable("Path", "$UserPath;$InstallDir", "User")
    Write-Host "   ✅ Added to PATH (requires shell restart)" -ForegroundColor Green
} else {
    Write-Host "   ✅ Already in PATH" -ForegroundColor Green
}

# --- 5. Custom Shortcuts ---
Write-Host "`n5. 📎 Creating Shortcuts" -ForegroundColor Yellow

$WshShell = New-Object -comObject WScript.Shell
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = "$DesktopPath\Xencode AI.lnk"
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = "$InstallDir\xencode.exe"
$Shortcut.WorkingDirectory = "$InstallDir"
$Shortcut.Description = "Xencode AI Assistant"
$Shortcut.Save()
Write-Host "   ✅ Created Desktop Shortcut: $ShortcutPath" -ForegroundColor Green

# --- 6. Uninstaller Generation ---
Write-Host "`n6. 🧹 Generatng Uninstaller" -ForegroundColor Yellow

$UninstallScript = "$InstallDir\uninstall.ps1"
$UninstallContent = @"
Write-Host "🗑️  Uninstalling Xencode..." -ForegroundColor Cyan

# Remove Directory
if (Test-Path "$InstallDir") {
    Remove-Item -Path "$InstallDir" -Recurse -Force
    Write-Host "   ✅ Removed installation files" -ForegroundColor Green
}

# Remove Shortcut
if (Test-Path "$ShortcutPath") {
    Remove-Item -Path "$ShortcutPath" -Force
    Write-Host "   ✅ Removed Desktop shortcut" -ForegroundColor Green
}

# Remove from Path (Advanced)
`$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (`$UserPath -like "*$InstallDir*") {
    `$NewPath = `$UserPath.Replace(";$InstallDir", "").Replace("$InstallDir;", "").Replace("$InstallDir", "")
    [Environment]::SetEnvironmentVariable("Path", `$NewPath, "User")
    Write-Host "   ✅ Removed from PATH" -ForegroundColor Green
}

Write-Host "✨ Uninstallation Complete!" -ForegroundColor Green
Pause
"@

Set-Content -Path $UninstallScript -Value $UninstallContent
Write-Host "   ✅ Created uninstaller: $UninstallScript" -ForegroundColor Green

# --- Finish ---
Write-Host "`n🎉 Installation Complete!" -ForegroundColor Green
if ($InstallMode -eq "release") {
    Write-Host "   • Binary: release v$Version downloaded from GitHub Releases and checksum-verified"
} else {
    Write-Host "   • Binary: built from source"
}
Write-Host "   • Run 'xencode' in a new terminal (launches the TUI)"
Write-Host "   • xencode query `"hello`" for a one-shot answer"
Write-Host "   • Open 'Xencode AI' from your Desktop"
Write-Host "   • To uninstall, run '$UninstallScript'"
Read-Host -Prompt "Press Enter to exit"
