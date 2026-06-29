$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$LogDir = Join-Path $Root "logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "test_run_$Timestamp.log"

$Python = "C:\Users\nisar\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

Write-Output "=== Finance Agent Test Run $Timestamp ===" | Tee-Object -FilePath $LogFile
Write-Output "Working directory: $Root" | Tee-Object -FilePath $LogFile -Append

if (-not (Test-Path (Join-Path $Root ".env"))) {
    Write-Output "WARNING: .env not found — copy .env.example and add API keys" | Tee-Object -FilePath $LogFile -Append
} else {
    Write-Output ".env present (gitignored)" | Tee-Object -FilePath $LogFile -Append
}

$env:PYTHONPATH = $Root
& $Python -m pip install pytest pytest-asyncio httpx -q 2>&1 | Tee-Object -FilePath $LogFile -Append

Write-Output "`n--- pytest ---" | Tee-Object -FilePath $LogFile -Append
& $Python -m pytest tests/ -v --tb=short 2>&1 | Tee-Object -FilePath $LogFile -Append
$PytestExit = $LASTEXITCODE

Write-Output "`n--- git check-ignore .env ---" | Tee-Object -FilePath $LogFile -Append
git check-ignore -v .env 2>&1 | Tee-Object -FilePath $LogFile -Append

Write-Output "`n--- env key status (not values) ---" | Tee-Object -FilePath $LogFile -Append
@("GEMINI_API_KEY","GEMINI_API_KEY_FALLBACK","NEWSAPI_KEY","CARTESIA_API_KEY") | ForEach-Object {
    $val = [Environment]::GetEnvironmentVariable($_)
    if (-not $val) {
        Get-Content (Join-Path $Root ".env") -ErrorAction SilentlyContinue | ForEach-Object {
            if ($_ -match "^$_=(.+)$" -and $matches[1].Trim()) { $val = "set_in_dotenv" }
        }
    }
    $status = if ($val) { "SET" } else { "MISSING" }
    "$($_)=$status" | Tee-Object -FilePath $LogFile -Append
}

Write-Output "`nLog saved to: $LogFile" | Tee-Object -FilePath $LogFile -Append
exit $PytestExit