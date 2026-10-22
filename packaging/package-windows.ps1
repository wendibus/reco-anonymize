# Package for Windows x86_64: binaries + FFmpeg DLLs from $env:FFMPEG_DIR\bin.
#   pwsh packaging/package-windows.ps1 <version> <target-dir> <out-dir>
param([string]$Version, [string]$Target = "target/release", [string]$Out = "dist")
$ErrorActionPreference = "Stop"
$Name = "reco-anonymize-$Version"
$Stage = Join-Path ([System.IO.Path]::GetTempPath()) ([System.Guid]::NewGuid()) | Join-Path -ChildPath $Name
New-Item -ItemType Directory -Force -Path $Stage, $Out, "$Stage/scripts" | Out-Null
Copy-Item "$Target/reco-anon.exe", "$Target/reco-anon-review.exe" $Stage
Copy-Item "$env:FFMPEG_DIR/bin/*.dll" $Stage
Get-ChildItem "$Target" -Filter "onnxruntime*.dll" -ErrorAction SilentlyContinue | Copy-Item -Destination $Stage
Copy-Item README.md, LICENSE, SECURITY.md, THIRD-PARTY-NOTICES.md $Stage
Copy-Item scripts/export_rfdetr.py, scripts/fetch_ppocr.py "$Stage/scripts"
$Zip = Join-Path $Out "$Name-windows-x86_64.zip"
Compress-Archive -Path $Stage -DestinationPath $Zip -Force
Write-Output $Zip
