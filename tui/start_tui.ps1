<#
.SYNOPSIS
    Starts the GVS5H harness TUI (tui\gvs5h_tui.py) on Windows.

.DESCRIPTION
    Finds a suitable Python 3 interpreter, makes sure `rich` is installed,
    switches the console to UTF-8 so the TUI's box-drawing and progress
    glyphs render, then launches the TUI from the repository root.
    All extra arguments are passed straight through to the TUI.

    Examples:
        .\start_tui.ps1                               # interactive (chat mode)
        .\start_tui.ps1 --mode harness --iters 4      # interactive, harness
        .\start_tui.ps1 --run "Say hi"                # one prompt, then exit
        .\start_tui.ps1 --run "<problem>" --spec code --json

.PARAMETER TuiArgs
    Arguments forwarded unchanged to gvs5h_tui.py.
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$TuiArgs
)

$ErrorActionPreference = 'Stop'

# Run from the repo root no matter where the script is invoked from.
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $RepoRoot 'tui\gvs5h_tui.py'))) {
    Write-Error "Cannot find tui\gvs5h_tui.py under '$RepoRoot'."
    exit 1
}
Set-Location $RepoRoot

# Pick a Python 3 interpreter: python -> py -3 -> python3.
$python = $null
foreach ($candidate in @(
        @{ Exe = 'python';  Args = @()     },
        @{ Exe = 'py';      Args = @('-3') },
        @{ Exe = 'python3'; Args = @()     }
    )) {
    if (-not (Get-Command $candidate.Exe -ErrorAction SilentlyContinue)) { continue }
    $candidateArgs = $candidate.Args
    # Probe the major version without any quotes in the -c code (PS 5.1 mangles
    # embedded double quotes when passing them to native commands).
    $major = & $candidate.Exe $candidateArgs -c 'import sys;print(sys.version_info[0])' 2>$null
    if ($LASTEXITCODE -eq 0 -and "$major" -match '^3$') {
        $python = $candidate
        break
    }
}
if (-not $python) {
    Write-Error "No Python 3 interpreter found. Install Python 3.10+ (https://www.python.org) and re-run."
    exit 1
}

$pyExe  = $python.Exe
$pyArgs = $python.Args

# Make sure `rich` is available (install once if it is missing).
# A successful import prints nothing, so judge by the exit code, not the output.
$null = & $pyExe @pyArgs -c 'import rich' 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "rich not found for '$pyExe' - installing it (pip install rich)..."
    & $pyExe @pyArgs -m pip install --quiet rich
    if ($LASTEXITCODE -ne 0) {
        Write-Error "'pip install rich' failed. Install rich manually and re-run."
        exit 1
    }
}

# UTF-8 console so rich's glyphs render; remember the previous code page.
$prevCodePage = $null
$ch = (chcp | Out-String)
if ($ch -match '(\d+)') { $prevCodePage = $Matches[1] }
if ($prevCodePage -ne '65001') {
    # NOTE: do NOT write this as [void](chcp ... 2>$null) - PowerShell 5.1 throws
    # an uncatchable error ("No ranking operator ... System.Void ...") for a
    # [void] cast wrapping a native command's 2>$null redirect.
    $null = chcp 65001 2>$null
}

# Launch the TUI in the foreground and pass its exit code back.
& $pyExe @pyArgs (Join-Path $RepoRoot 'tui\gvs5h_tui.py') @TuiArgs
$code = $LASTEXITCODE

if ($prevCodePage -and $prevCodePage -ne '65001') {
    $null = chcp $prevCodePage 2>$null   # restore; see note above re: [void]+2>$null
}
exit $code
