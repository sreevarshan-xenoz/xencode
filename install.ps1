# Xencode Windows Installer (Rust)
# ================================

$ErrorActionPreference = "Stop"
Write-Host "🚀 Installing Xencode for Windows" -ForegroundColor Cyan
Write-Host "============================"

# --- Configuration ---
$AppName = "xencode"
$InstallDir = "$env:LOCALAPPDATA\xencode"
$RepoUrl = "https://github.com/sreevarshan-xenoz/xencode.git"

# --- 1. System Checks ---
Write-Host "`n1. 🔍 System Checks" -ForegroundColor Yellow

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

# Check Ollama (Warning only)
if (Get-Command "ollama" -ErrorAction SilentlyContinue) {
    Write-Host "   ✅ Ollama found" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  Ollama not found. Xencode needs Ollama for AI features." -ForegroundColor DarkYellow
    Write-Host "      You can install it later from https://ollama.ai" -ForegroundColor Gray
}

# --- 2. Build ---
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

# --- 3. Install ---
Write-Host "`n3. 📦 Installing" -ForegroundColor Yellow

if (Test-Path $InstallDir) {
    Write-Host "   📂 Cleaning existing installation directory..." -ForegroundColor Gray
    Remove-Item -Path $InstallDir -Recurse -Force
}
New-Item -Path $InstallDir -ItemType Directory -Force | Out-Null

Copy-Item $BuiltExe "$InstallDir\xencode.exe"
Write-Host "   ✅ Installed: $InstallDir\xencode.exe" -ForegroundColor Green

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
Write-Host "   • Run 'xencode' in a new terminal (launches the TUI)"
Write-Host "   • xencode query `"hello`" for a one-shot answer"
Write-Host "   • Open 'Xencode AI' from your Desktop"
Write-Host "   • To uninstall, run '$UninstallScript'"
Read-Host -Prompt "Press Enter to exit"
