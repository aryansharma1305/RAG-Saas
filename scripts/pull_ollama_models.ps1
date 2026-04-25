param(
    [string]$EnvPath = ".env",
    [string[]]$Models = @()
)

$ErrorActionPreference = "Stop"

function Read-DotEnvValue {
    param(
        [string]$Path,
        [string]$Name,
        [string]$Default
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return $Default
    }

    $line = Get-Content -LiteralPath $Path |
        Where-Object { $_ -match "^\s*$([regex]::Escape($Name))\s*=" } |
        Select-Object -Last 1

    if (-not $line) {
        return $Default
    }

    return ($line -replace "^\s*$([regex]::Escape($Name))\s*=\s*", "").Trim().Trim('"').Trim("'")
}

$ollamaCommand = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollamaCommand) {
    $standardPath = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    if (Test-Path -LiteralPath $standardPath) {
        $env:Path = "$(Split-Path -Parent $standardPath);$env:Path"
        $ollamaCommand = Get-Command ollama -ErrorAction SilentlyContinue
    }
}

if (-not $ollamaCommand) {
    throw "Ollama was not found. Install Ollama, open a new PowerShell window, then rerun this script."
}

if ($Models.Count -eq 0) {
    $Models = @(
        Read-DotEnvValue -Path $EnvPath -Name "QWEN_MODEL" -Default "qwen2.5:3b"
        Read-DotEnvValue -Path $EnvPath -Name "GEMMA_MODEL" -Default "gemma3:1b"
        Read-DotEnvValue -Path $EnvPath -Name "GLM_MODEL" -Default "glm4:9b"
    )
}

$Models |
    Where-Object { $_ -and $_.Trim() } |
    Select-Object -Unique |
    ForEach-Object {
        Write-Host "Pulling $_"
        ollama pull $_
    }

Write-Host "Available local models:"
ollama list
