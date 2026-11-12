<#
.SYNOPSIS
  One-command launcher: program the Cmod A7 and start the live KWS demo,
  or run the golden-vector simulation.

.DESCRIPTION
  The Cmod A7 is programmed over JTAG, which is volatile -- it loses the design
  on every power cycle. So after you unplug/replug the board, just run:

      .\run.ps1

  It re-programs the FPGA from the built bitstream and starts the live
  mic -> FPGA -> LED demo. Speak a keyword and the board classifies it on-chip
  and lights the matching LED color. Press Ctrl+C to stop.

  Options:
    -Sim             Run the xsim golden-vector check only (no board needed).
    -JustRun         Skip programming; just start the demo (board already loaded).
    -Build           Rebuild the bitstream first.
    -Port  COM8      Serial port (default COM8).
    -Device 1        Microphone device index (default 1).

  Vivado defaults to C:\AMDDesignTools\2026.1; override with the environment
  variable VIVADO_BAT.
#>
[CmdletBinding()]
param(
    [switch]$Sim,
    [switch]$JustRun,
    [switch]$Build,
    [string]$Port   = "COM8",
    [int]$Device    = 1
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

$vivado = if ($env:VIVADO_BAT) { $env:VIVADO_BAT } else { "C:\AMDDesignTools\2026.1\Vivado\bin\vivado.bat" }
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }  # fall back to PATH

$bit  = Join-Path $root "build\vivado\kws_fpga_led.runs\impl_1\top.bit"
$vlog = Join-Path $root "build\last_vivado.log"

function Invoke-Vivado([string]$tcl, [string]$label) {
    if (-not (Test-Path $vivado)) {
        throw "Vivado not found at '$vivado'. Set `$env:VIVADO_BAT to vivado.bat."
    }
    Write-Host ">>> $label ..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Force (Join-Path $root "build") | Out-Null
    & $vivado -mode batch -notrace -nojournal -nolog -source $tcl | Tee-Object -FilePath $vlog
    if ($LASTEXITCODE -ne 0) { throw "$label failed (exit $LASTEXITCODE)." }
}

# --- Simulation only ---------------------------------------------------------
if ($Sim) {
    Invoke-Vivado (Join-Path $root "fpga\vivado\sim_mlp.tcl") "Simulating kws_mlp against 128 golden vectors"
    if (-not (Select-String -Path $vlog -Pattern "tb_kws_mlp: PASS" -Quiet)) { throw "Simulation did not report PASS." }
    Write-Host ">>> PASS: VHDL matches the Python fixed-point model bit for bit." -ForegroundColor Green
    exit 0
}

# --- Program the FPGA (unless told to skip) ----------------------------------
if ($JustRun) {
    Write-Host ">>> Skipping FPGA programming (-JustRun)." -ForegroundColor Yellow
}
else {
    if ($Build -or -not (Test-Path $bit)) {
        Invoke-Vivado (Join-Path $root "fpga\vivado\build.tcl") "Building bitstream"
    }
    Invoke-Vivado (Join-Path $root "fpga\vivado\program.tcl") "Programming FPGA over JTAG"
}

# --- Start the live KWS demo -------------------------------------------------
Write-Host ">>> Starting live mic -> FPGA -> LED demo on $Port (Ctrl+C to stop)." -ForegroundColor Cyan
& $python (Join-Path $root "pc\kws_live.py") `
    --mode live --fpga-inference `
    --port $Port --device $Device `
    --samplerate 44100 --channels 1 `
    --min-rms 0.006 --frame-delay 0.003
