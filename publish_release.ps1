# Публикация на GitHub: репозиторий (если его ещё нет), код и релиз со снимком данных.
#
#   powershell -ExecutionPolicy Bypass -File publish_release.ps1 -Visibility private
#   powershell -ExecutionPolicy Bypass -File publish_release.ps1 -Visibility public -Name sberindex-forecast-read
#
# Перед этим:  .venv\Scripts\python -m worker release  (или с --raw)
# Нужен GitHub CLI, выполнен вход: gh auth login
#
# Что делает:
#  1. нет git remote origin – создаёт репозиторий на GitHub (-Visibility, -Name)
#     и отправляет туда код; есть – просто отправляет текущую ветку;
#  2. создаёт релиз vГГГГ.ММ.ДД с файлами из dist/ и описанием dist/RELEASE_NOTES.md.
# После этого setup.ps1 на любом ПК сам скачивает данные из последнего релиза.

param(
    [ValidateSet("private", "public")] [string]$Visibility = "private",
    [string]$Name = "sberindex-forecast",
    [string]$Tag = ("v" + (Get-Date -Format "yyyy.MM.dd"))
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
# весь вывод – ещё и в logs\publish_release.log: его можно переслать, если что-то не установилось
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
Start-Transcript -Path "logs\publish_release.log" -Append | Out-Null
trap { Write-Host "Ошибка: $_" -ForegroundColor Red; Stop-Transcript | Out-Null; exit 1 }

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    Write-Host "Нужен GitHub CLI: https://cli.github.com/" -ForegroundColor Red; Stop-Transcript | Out-Null; exit 1
}
gh auth status 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Host "Сначала войдите: gh auth login" -ForegroundColor Red; Stop-Transcript | Out-Null; exit 1 }

$assets = @("dist\sberindex-data.zip", "dist\sberindex-data.zip.sha256")
if (Test-Path "dist\sberindex-raw.zip") { $assets += @("dist\sberindex-raw.zip", "dist\sberindex-raw.zip.sha256") }
foreach ($a in $assets + @("dist\RELEASE_NOTES.md")) {
    if (-not (Test-Path $a)) {
        Write-Host "Нет $a – сначала: .venv\Scripts\python -m worker release" -ForegroundColor Red; Stop-Transcript | Out-Null; exit 1
    }
}
if (git status --porcelain) {
    Write-Host "Есть незакоммиченные изменения – закоммитьте их, чтобы релиз соответствовал коду." -ForegroundColor Red
    Stop-Transcript | Out-Null; exit 1
}

$origin = git remote get-url origin 2>$null
if (-not $origin) {
    Write-Host "Создаю репозиторий $Name ($Visibility) и отправляю код…"
    gh repo create $Name "--$Visibility" --source . --remote origin --push
    if ($LASTEXITCODE -ne 0) { Stop-Transcript | Out-Null; exit 1 }
} else {
    Write-Host "Отправляю код в $origin…"
    git push -u origin HEAD
    if ($LASTEXITCODE -ne 0) { Stop-Transcript | Out-Null; exit 1 }
}

Write-Host "Создаю релиз $Tag…"
gh release create $Tag @assets --title "Данные $Tag" --notes-file dist\RELEASE_NOTES.md
if ($LASTEXITCODE -ne 0) { Stop-Transcript | Out-Null; exit 1 }

$repo = gh repo view --json nameWithOwner -q .nameWithOwner
Write-Host ""
Write-Host "Готово: https://github.com/$repo/releases/tag/$Tag" -ForegroundColor Green
Write-Host "На другом ПК: git clone https://github.com/$repo.git; затем setup.ps1"
Stop-Transcript | Out-Null
