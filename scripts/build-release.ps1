# Build release binary for Windows
param(
    [string]$Version = (git describe --tags --always 2>$null)
)

if (-not $Version) {
    $Version = "2.1.0"
}

Write-Host "Building xencode v$Version..." -ForegroundColor Cyan

# Build release
Set-Location rust
cargo build --release -p xencode-cli
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

# Copy binary with version
$BinaryName = "xencode-v$Version-x86_64-windows.exe"
Copy-Item "target/release/xencode.exe" "target/release/$BinaryName"

# Build checksum
$hash = Get-FileHash "target/release/$BinaryName" -Algorithm SHA256
"$($hash.Hash)  $BinaryName" | Out-File -FilePath "target/release/$BinaryName.sha256"

Write-Host "Done: target/release/$BinaryName" -ForegroundColor Green
Write-Host "SHA256: $($hash.Hash)" -ForegroundColor Gray
