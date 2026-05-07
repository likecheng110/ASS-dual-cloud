param(
    [ValidateSet("isic", "chest", "all")]
    [string]$Dataset = "all",
    [string]$TargetRoot = "data\official_medical_images",
    [switch]$Extract,
    [switch]$DeleteArchivesAfterExtract,
    [switch]$AllowChestMetadataMirror
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
if (-not [System.IO.Path]::IsPathRooted($TargetRoot)) {
    $TargetRoot = Join-Path $repoRoot $TargetRoot
}
$TargetRoot = [System.IO.Path]::GetFullPath($TargetRoot)
New-Item -ItemType Directory -Force -Path $TargetRoot | Out-Null

function Write-ManifestLine {
    param([string]$Path, [string]$Text)
    Add-Content -LiteralPath $Path -Value $Text -Encoding UTF8
}

function Download-File {
    param(
        [string]$Uri,
        [string]$Destination
    )
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
    if (Test-Path $Destination) {
        $existing = Get-Item -LiteralPath $Destination
        if ($existing.Length -gt 0) {
            Write-Host "Exists, skip: $Destination"
            return
        }
    }
    Write-Host "Downloading: $Uri"
    Write-Host "       To: $Destination"
    try {
        Start-BitsTransfer -Source $Uri -Destination $Destination -DisplayName "ASS dataset download" -Description $Uri -ErrorAction Stop
    } catch {
        Write-Host "BITS failed, falling back to Invoke-WebRequest: $($_.Exception.Message)"
        Invoke-WebRequest -Uri $Uri -OutFile $Destination -UseBasicParsing
    }
}

$manifestPath = Join-Path $TargetRoot "download_manifest.md"
Set-Content -LiteralPath $manifestPath -Value "# Official Medical Dataset Download Manifest`n" -Encoding UTF8
Write-ManifestLine $manifestPath "- Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-ManifestLine $manifestPath "- Target root: $TargetRoot"
Write-ManifestLine $manifestPath ""

if ($Dataset -in @("chest", "all")) {
    $chestRoot = Join-Path $TargetRoot "ChestXray14"
    $archiveRoot = Join-Path $chestRoot "archives"
    $imageRoot = Join-Path $chestRoot "images"
    New-Item -ItemType Directory -Force -Path $archiveRoot, $imageRoot | Out-Null

    $nihImageUrls = @(
        "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz",
        "https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz",
        "https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz",
        "https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz",
        "https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz",
        "https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz",
        "https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz",
        "https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz",
        "https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz",
        "https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz",
        "https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz",
        "https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz"
    )

    Write-ManifestLine $manifestPath "## NIH ChestX-ray14"
    Write-ManifestLine $manifestPath "- Official homepage: https://nihcc.app.box.com/v/ChestXray-NIHCC"
    Write-ManifestLine $manifestPath "- Image archives: NIH Box static files from the official Box release."
    Write-ManifestLine $manifestPath "- Note: the official Box metadata CSV is exposed through the Box web UI. If direct automated metadata download is unavailable, place Data_Entry_2017.csv or Data_Entry_2017_v2020.csv in $chestRoot."
    Write-ManifestLine $manifestPath ""

    for ($i = 0; $i -lt $nihImageUrls.Count; $i++) {
        $name = "images_{0:d3}.tar.gz" -f ($i + 1)
        $dest = Join-Path $archiveRoot $name
        Download-File -Uri $nihImageUrls[$i] -Destination $dest
        if ($Extract.IsPresent) {
            Write-Host "Extracting: $dest"
            tar -xzf $dest -C $imageRoot
            if ($LASTEXITCODE -ne 0) {
                throw "tar extraction failed for $dest"
            }
            if ($DeleteArchivesAfterExtract.IsPresent) {
                Remove-Item -LiteralPath $dest -Force
            }
        }
    }

    if ($AllowChestMetadataMirror.IsPresent) {
        Write-Host "Downloading NIH metadata from a public mirror of the official metadata files."
        Download-File -Uri "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/raw/main/data/Data_Entry_2017_v2020.csv" -Destination (Join-Path $chestRoot "Data_Entry_2017_v2020.csv")
        Download-File -Uri "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/raw/main/data/train_val_list.txt" -Destination (Join-Path $chestRoot "train_val_list.txt")
        Download-File -Uri "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/raw/main/data/test_list.txt" -Destination (Join-Path $chestRoot "test_list.txt")
        Write-ManifestLine $manifestPath "- Metadata mirror enabled: downloaded NIH metadata mirror files from Hugging Face dataset repo. Verify against NIH Box before final publication if strict official-only provenance is required."
    }
}

if ($Dataset -in @("isic", "all")) {
    $isicRoot = Join-Path $TargetRoot "ISIC2018"
    $archiveRoot = Join-Path $isicRoot "archives"
    New-Item -ItemType Directory -Force -Path $archiveRoot | Out-Null

    $isicFiles = @(
        @{ Name = "ISIC2018_Task3_Training_Input.zip"; Uri = "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Training_Input.zip" },
        @{ Name = "ISIC2018_Task3_Training_GroundTruth.zip"; Uri = "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Training_GroundTruth.zip" }
    )

    Write-ManifestLine $manifestPath "## ISIC 2018 Task 3"
    Write-ManifestLine $manifestPath "- Official homepage: https://challenge2018.isic-archive.com/"
    Write-ManifestLine $manifestPath "- Files: official ISIC Challenge 2018 S3 files for Task 3 training input and ground truth."
    Write-ManifestLine $manifestPath ""

    foreach ($file in $isicFiles) {
        $dest = Join-Path $archiveRoot $file.Name
        Download-File -Uri $file.Uri -Destination $dest
        if ($Extract.IsPresent) {
            Write-Host "Extracting: $dest"
            Expand-Archive -LiteralPath $dest -DestinationPath $isicRoot -Force
            if ($DeleteArchivesAfterExtract.IsPresent) {
                Remove-Item -LiteralPath $dest -Force
            }
        }
    }
}

$envPath = Join-Path $TargetRoot "official_medical_image_env.ps1"
$chestPath = Join-Path $TargetRoot "ChestXray14"
$isicPath = Join-Path $TargetRoot "ISIC2018"
$envText = @"
`$env:ASS_CHESTXRAY14_DIR = "$chestPath"
`$env:ASS_CHESTXRAY14_LABELS = "$chestPath\Data_Entry_2017_v2020.csv"
`$env:ASS_ISIC2018_DIR = "$isicPath"
`$env:ASS_ISIC2018_LABELS = "$isicPath\ISIC2018_Task3_Training_GroundTruth\ISIC2018_Task3_Training_GroundTruth.csv"
"@
Set-Content -LiteralPath $envPath -Value $envText -Encoding UTF8
Write-ManifestLine $manifestPath ""
Write-ManifestLine $manifestPath "Environment file: $envPath"
Write-Host "Done. Manifest: $manifestPath"
Write-Host "Environment file: $envPath"
