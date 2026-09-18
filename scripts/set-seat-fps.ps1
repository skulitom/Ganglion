# Raise the Remote Desktop composition frame cap that limits a seat (a child session).
# Windows caps RDP sessions at 30 fps unless DWMFRAMEINTERVAL is set; the seat tool's fps setup writes 15
# (~66 Hz measured). This writes 10 (~100 Hz expected; `ganglion bench` measures what the seat delivers).
# Needs administrator rights and a reboot to take effect. Run:  powershell -File scripts\set-seat-fps.ps1
param([int]$IntervalMs = 10)
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Start-Process powershell -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -IntervalMs $IntervalMs"
    exit
}
$key = 'HKLM:\SYSTEM\CurrentControlSet\Control\Terminal Server\WinStations'
$old = (Get-ItemProperty -Path $key -Name DWMFRAMEINTERVAL -ErrorAction SilentlyContinue).DWMFRAMEINTERVAL
if ($null -eq $old) { $old = 'unset (30 fps cap)' }
New-ItemProperty -Path $key -Name DWMFRAMEINTERVAL -PropertyType DWord -Value $IntervalMs -Force | Out-Null
$new = (Get-ItemProperty -Path $key -Name DWMFRAMEINTERVAL).DWMFRAMEINTERVAL
Write-Host "DWMFRAMEINTERVAL: $old -> $new   (takes effect after a reboot; the seat must be restarted afterwards)"
Read-Host "Press Enter to close"
