param(
  [int]$MaxJobs = 4,
  [string]$Symbols = "SOLUSDT",
  [string]$Regimes = "TREND",
  [string]$Seeds   = "1337 2027 4041",
  [string]$Windows = "2020-08_2021-12 2022-01_2022-12 2023-01_2023-12"
)

$ErrorActionPreference = "Stop"

# Root = repo (este .ps1 vive en bat/)
$ROOT = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ROOT

# Un RUN_ID para toda la corrida
$runId = (Get-Date -Format "yyyy-MM-dd_HH-mm-ss")

# Split lists (por espacios)
$SYMS  = $Symbols.Split(' ', [System.StringSplitOptions]::RemoveEmptyEntries)
$REGS  = $Regimes.Split(' ', [System.StringSplitOptions]::RemoveEmptyEntries)
$SEEDL = $Seeds.Split(' ',   [System.StringSplitOptions]::RemoveEmptyEntries)
$WINL  = $Windows.Split(' ', [System.StringSplitOptions]::RemoveEmptyEntries)

# Dirs base
New-Item -ItemType Directory -Force -Path (Join-Path $ROOT "logs") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ROOT "results") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ROOT "results\pipeline_trades_parts") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ROOT "results\robust") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ROOT "results\grid_refine") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ROOT "results\library") | Out-Null

$LIB         = Join-Path $ROOT "results\library\top_k_library.json"
$DATA_BASE   = Join-Path $ROOT "datasets"
$CONFIG_BASE = Join-Path $ROOT "configs"

# Cola de casos
$cases = @()
foreach ($m in $SYMS) {
  foreach ($r in $REGS) {
    foreach ($s in $SEEDL) {
      foreach ($w in $WINL) {
        $cases += [pscustomobject]@{ symbol=$m; regime=$r; seed=$s; window=$w }
      }
    }
  }
}

Write-Host "ROOT=$ROOT"
Write-Host "RUN_ID=$runId"
Write-Host ("CASES=" + $cases.Count)
Write-Host ("MaxJobs=" + $MaxJobs)

function Wait-ForSlot([int]$max) {
  while (@(Get-Job -State Running).Count -ge $max) {
    Start-Sleep -Seconds 2
  }
}

foreach ($c in $cases) {
  Wait-ForSlot $MaxJobs

  $m = $c.symbol
  $r = $c.regime
  $s = $c.seed
  $w = $c.window

  $data    = Join-Path $DATA_BASE "$m\1m"
  $baseCfg = Join-Path $CONFIG_BASE "backtest_${r}_only.json"

  $robustOut = Join-Path $ROOT ("results\robust\{0}\{1}\seed_{2}\{3}\robust_candidates.json" -f $m,$r,$s,$w)
  $gridDir   = Join-Path $ROOT ("results\grid_refine\{0}\{1}\seed_{2}\{3}" -f $m,$r,$s,$w)
  $bestCfg   = Join-Path $gridDir "best_config.json"

  New-Item -ItemType Directory -Force -Path (Split-Path $robustOut) | Out-Null
  New-Item -ItemType Directory -Force -Path $gridDir | Out-Null

  # ✅ CSV por-caso (evita colisiones) -> REQUIERE el patch al logger (abajo)
  $partDir = Join-Path $ROOT ("results\pipeline_trades_parts\{0}\{1}\{2}\seed_{3}\{4}" -f $runId,$m,$r,$s,$w)
  New-Item -ItemType Directory -Force -Path $partDir | Out-Null
  $partCsv = Join-Path $partDir "pipeline_trades.csv"

  # ✅ Log por-caso
  $logFile = Join-Path $ROOT ("logs\pipeline_{0}_{1}_{2}_seed_{3}_{4}.log" -f $runId,$m,$r,$s,$w)

  Start-Job -Name ("case_{0}_{1}_{2}_{3}" -f $m,$r,$s,$w) -ArgumentList @(
    $ROOT,$m,$r,$s,$w,$data,$baseCfg,$robustOut,$gridDir,$bestCfg,$LIB,$runId,$partCsv,$logFile
  ) -ScriptBlock {
    param($ROOT,$m,$r,$s,$w,$data,$baseCfg,$robustOut,$gridDir,$bestCfg,$LIB,$runId,$partCsv,$logFile)

    Set-Location $ROOT

    # Env para RUN_MODE + meta_json.context
    $env:RUN_MODE = "PIPELINE"
    $env:PYTHONUNBUFFERED = "1"
    $env:PYTHONPATH = $ROOT

    $env:PIPELINE_RUN_ID = $runId
    $env:PIPELINE_SYMBOL = $m
    $env:PIPELINE_REGIME = $r
    $env:PIPELINE_SEED   = $s
    $env:PIPELINE_WINDOW = $w

    # ✅ Donde escribir trades del pipeline (por-caso)
    # (tu logger debe leer esto, ver patch abajo)
    $env:PIPELINE_TRADES_PATH = $partCsv

    # 1) robust_optimizer
    & python -u -m analysis.robust_optimizer `
      --data $data `
      --out $robustOut `
      --seed $s `
      --window $w `
      --min-candles 1 `
      --base-config $baseCfg `
      --symbol $m `
      --interval 1m `
      --warmup 500 >> $logFile 2>&1

    if ($LASTEXITCODE -ne 0) { throw "robust_optimizer failed ($m $r $s $w)" }

    # 2) grid_runner
    & python -u -m backtest.grid_runner `
      --base-config $baseCfg `
      --grid $robustOut `
      --data $data `
      --window $w `
      --results-dir $gridDir `
      --resume >> $logFile 2>&1

    if ($LASTEXITCODE -ne 0) { throw "grid_runner failed ($m $r $s $w)" }

    # 3) topk_indexer
    if (Test-Path $bestCfg) {
      & python -u -m analysis.topk_indexer `
        --input $bestCfg `
        --library $LIB >> $logFile 2>&1
    } else {
      "WARN best_config.json missing: $bestCfg" | Out-File -Append -FilePath $logFile
    }
  } | Out-Null

  Write-Host ("STARTED {0} {1} seed={2} window={3} -> {4}" -f $m,$r,$s,$w,$partCsv)
}

Write-Host "Waiting jobs..."
Get-Job | Wait-Job | Out-Null

$failed = @(Get-Job | Where-Object { $_.State -eq "Failed" })
if ($failed.Count -gt 0) {
  Write-Host "FAILED JOBS:" -ForegroundColor Red
  foreach ($j in $failed) {
    Write-Host $j.Name -ForegroundColor Red
    try { Receive-Job $j -Keep | Out-Host } catch {}
  }
  exit 1
}

Write-Host "ALL OK" -ForegroundColor Green
Write-Host "Now merge partial CSVs into results\pipeline_trades.csv"
Write-Host ("Run: python -u analysis\merge_pipeline_trades.py --run-id {0}" -f $runId)

