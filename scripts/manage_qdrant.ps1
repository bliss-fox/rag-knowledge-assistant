param(
    [ValidateSet('start', 'stop', 'restart', 'status', 'logs', 'pull')]
    [string]$Action = 'status'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$composeFile = Join-Path $projectRoot 'deploy\qdrant\compose.yaml'
$projectName = 'modular-rag-qdrant'

function Assert-DockerDaemon {
    try {
        $dockerExecutable = (Get-Command docker.exe -ErrorAction Stop).Source
    }
    catch {
        throw 'Docker CLI is unavailable. Install Docker Desktop, then retry.'
    }

    $probe = Start-Process -FilePath $dockerExecutable `
        -ArgumentList @('version', '--format', '{{.Server.Version}}') `
        -WindowStyle Hidden -PassThru
    if (-not $probe.WaitForExit(10000)) {
        $probe.Kill()
        throw 'Docker daemon did not respond within 10 seconds. Check Docker Desktop diagnostics.'
    }
    if ($probe.ExitCode -ne 0) {
        throw 'Docker daemon is unavailable. Start Docker Desktop, then retry.'
    }
}

function Invoke-QdrantCompose {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)

    & docker compose --project-name $projectName --file $composeFile @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose failed with exit code $LASTEXITCODE"
    }
}

function Wait-QdrantHealth {
    $deadline = (Get-Date).AddSeconds(90)
    do {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:6333/healthz' `
                -TimeoutSec 2
            if ($response.StatusCode -eq 200) {
                Write-Output 'Qdrant is healthy at http://127.0.0.1:6333'
                return
            }
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    } while ((Get-Date) -lt $deadline)

    throw 'Qdrant did not become healthy within 90 seconds.'
}

if (-not (Test-Path -LiteralPath $composeFile -PathType Leaf)) {
    throw "Qdrant compose file is missing: $composeFile"
}

Assert-DockerDaemon

switch ($Action) {
    'start' {
        Invoke-QdrantCompose up -d --wait
        Wait-QdrantHealth
    }
    'stop' {
        # Preserve the container and named volume so a later start is lossless.
        Invoke-QdrantCompose stop
    }
    'restart' {
        Invoke-QdrantCompose restart
        Wait-QdrantHealth
    }
    'status' {
        Invoke-QdrantCompose ps
    }
    'logs' {
        Invoke-QdrantCompose logs --tail 200
    }
    'pull' {
        Invoke-QdrantCompose pull
    }
}
