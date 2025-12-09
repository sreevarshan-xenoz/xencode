# Xencode Windows Installer
# =========================

$ErrorActionPreference = "Stop"
Write-Host "🚀 Installing Xencode for Windows" -ForegroundColor Cyan
Write-Host "============================"

# --- Configuration ---
$AppName = "xencode"
$InstallDir = "$env:LOCALAPPDATA\xencode"
$PythonMinVersion = "3.8"
$RepoUrl = "https://github.com/sreevarshan-xenoz/xencode.git"

# --- 1. System Checks ---
Write-Host "`n1. 🔍 System Checks" -ForegroundColor Yellow

# Check Python
if (Get-Command "python" -ErrorAction SilentlyContinue) {
    $PyVer = python --version 2>&1
    Write-Host "   ✅ Python found: $PyVer" -ForegroundColor Green
} else {
    Write-Host "   ❌ Python not found. Please install Python $PythonMinVersion+" -ForegroundColor Red
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

# --- 2. Environment Setup ---
Write-Host "`n2. 🛠️  Environment Setup" -ForegroundColor Yellow

if (Test-Path $InstallDir) {
    Write-Host "   📂 Cleaning existing installation directory..." -ForegroundColor Gray
    Remove-Item -Path $InstallDir -Recurse -Force
}
New-Item -Path $InstallDir -ItemType Directory -Force | Out-Null
Write-Host "   ✅ Created installation directory: $InstallDir" -ForegroundColor Green

# Create venv
Write-Host "   🐍 Creating virtual environment..." -ForegroundColor Gray
python -m venv "$InstallDir\venv"
if (-not (Test-Path "$InstallDir\venv\Scripts\python.exe")) {
    Write-Host "   ❌ Failed to create venv." -ForegroundColor Red
    exit 1
}
Write-Host "   ✅ Virtual environment ready" -ForegroundColor Green

# Install Dependencies
Write-Host "   📦 Installing dependencies (this may take a moment)..." -ForegroundColor Gray
& "$InstallDir\venv\Scripts\python.exe" -m pip install --upgrade pip
& "$InstallDir\venv\Scripts\python.exe" -m pip install -r requirements.txt
& "$InstallDir\venv\Scripts\python.exe" -m pip install pyinstaller>=6.3.0
Write-Host "   ✅ Dependencies installed" -ForegroundColor Green

# --- 3. Build Executable ---
Write-Host "`n3. 🏗️  Building Standalone Application" -ForegroundColor Yellow
Write-Host "   ⏳ Running PyInstaller (please wait)..." -ForegroundColor Gray

# Call build_exe.py using the venv python to ensure all deps are found
# We assumes build_exe.py is in the current directory (repo root)
if (Test-Path "build_exe.py") {
    & "$InstallDir\venv\Scripts\python.exe" "build_exe.py"
    
    if (Test-Path "dist\xencode.exe") {
        Copy-Item "dist\xencode.exe" "$InstallDir\xencode.exe"
        Write-Host "   ✅ Standalone executable built and moved to install dir" -ForegroundColor Green
    } else {
        Write-Host "   ❌ Build failed: dist\xencode.exe not found." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "   ❌ build_exe.py not found in current directory." -ForegroundColor Red
    exit 1
}

# --- 4. CLI Integration ---
Write-Host "`n4. 🔌 CLI Integration" -ForegroundColor Yellow

$BatPath = "$InstallDir\xencode.bat"
# Create shim that prefers the exe if it exists, roughly mimicking the shell script logic?
# Actually simpler: The batch file just runs the exe or the python script? 
# Use the python script for CLI to allow faster updates without rebuilding EXE every time?
# User wants "CLI tool that runs terminal". The EXE does exactly that.
# Let's make the shim point to the exe for consistency.
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
Write-Host "   • Run 'xencode' in a new terminal"
Write-Host "   • Open 'Xencode AI' from your Desktop"
Write-Host "   • To uninstall, run '$UninstallScript'"
Read-Host -Prompt "Press Enter to exit"
