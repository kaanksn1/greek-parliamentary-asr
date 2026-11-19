param([Parameter(Mandatory=$true)][int]$TrainingProcessId)
# Temporary process-scoped sleep prevention; no power-plan/registry changes.
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class P3TrainingPower {
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
try {
    $p3PowerResult=[P3TrainingPower]::SetThreadExecutionState([uint32]2147483649)
    if ($p3PowerResult -eq 0) { throw 'Temporary system-awake request failed' }
    Write-Output ('System sleep prevention active while training process '+$TrainingProcessId+' runs; display may turn off.')
    while (Get-Process -Id $TrainingProcessId -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 15
    }
} finally {
    [void][P3TrainingPower]::SetThreadExecutionState([uint32]2147483648)
}
