"""Снимок готовых данных: упаковать для GitHub Release и скачать на другом ПК.

  python -m worker snapshot pack            → dist/sberindex-data.zip (+ .sha256)
  python -m worker snapshot pack --raw      → ещё dist/sberindex-raw.zip (исходники, ~200 МБ)
  python -m worker snapshot fetch [--url U] → скачать и распаковать в проект

Распаковка пишет только в data/ и reports/ – архив не может перезаписать код.
Контрольная сумма проверяется, если рядом с архивом лежит файл .sha256.
"""

import hashlib
import logging
import os
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from src import store
from src.config import NO_WINDOW, ROOT, load_config

log = logging.getLogger("worker")

DIST = ROOT / "dist"
ASSET = "sberindex-data.zip"
ALLOWED = ("data/", "reports/")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _zip(paths: list[str], out: Path, skip=()) -> Path:
    DIST.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in paths:
            p = ROOT / rel
            files = [p] if p.is_file() else sorted(q for q in p.rglob("*") if q.is_file()) if p.exists() else []
            for f in files:
                name = f.relative_to(ROOT).as_posix()
                if any(s in name for s in skip):
                    continue
                z.write(f, name)
    (out.with_name(out.name + ".sha256")).write_text(f"{_sha256(out)}  {out.name}\n", encoding="ascii")
    return out


def pack(raw: bool = False, echo=print) -> list[Path]:
    store.publish()     # в снимок идёт свежая база
    inc = load_config("pipeline")["snapshot"]["include"]
    out = [_zip(inc, DIST / ASSET, skip=(".backup", ".lock", ".tmp"))]
    if raw:
        out.append(_zip(["data/raw"], DIST / "sberindex-raw.zip"))
    for o in out:
        log.info("snapshot_pack  %s %.1f МБ sha256=%s", o.name, o.stat().st_size / 1e6, _sha256(o))
        echo(f"{o.relative_to(ROOT)}: {o.stat().st_size / 1e6:.1f} МБ")
    return out


def github_repo() -> str | None:
    """owner/repo из адреса git remote origin, если это GitHub."""
    import re
    import subprocess
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "remote", "get-url", "origin"],
                             capture_output=True, text=True, timeout=5, creationflags=NO_WINDOW).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$", out)
    return f"{m.group(1)}/{m.group(2)}" if m else None


def resolve_url() -> str | None:
    """Адрес снимка: из конфига, а если там пусто – последний релиз репозитория
    на GitHub (после публикации ничего править не нужно)."""
    url = load_config("pipeline")["snapshot"].get("url")
    if url:
        return url
    repo = github_repo()
    return f"https://github.com/{repo}/releases/latest/download/{ASSET}" if repo else None


def fetch(url: str | None = None, echo=print) -> None:
    url = url or resolve_url()
    if not url:
        raise SystemExit("адрес снимка не задан: configs/pipeline.yaml → snapshot.url, "
                         "git remote на GitHub или --url")
    DIST.mkdir(exist_ok=True)
    zpath = DIST / Path(url.split("?")[0]).name
    log.info("snapshot_fetch  %s", url)
    echo(f"скачиваю {url}")
    sha_file = zpath.with_name(zpath.name + ".sha256")
    sha_file.unlink(missing_ok=True)
    part = zpath.with_name(zpath.name + ".part")
    try:
        # сначала во временный файл: обрыв связи не оставит битый архив под
        # настоящим именем; длину сверяем с заголовком сервера. Обрыв или
        # долгая пауза сети – повтор с докачкой с места обрыва (Range), до 5 раз
        part.unlink(missing_ok=True)
        want_len = 0
        for attempt in range(1, 6):
            have = part.stat().st_size if part.exists() else 0
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            try:
                with urllib.request.urlopen(req, timeout=120) as r,                         open(part, "ab" if have and r.status == 206 else "wb") as f:
                    if r.status == 206:
                        total = r.headers.get("Content-Range", "").rsplit("/", 1)[-1]
                        want_len = int(total) if total.isdigit() else 0
                    else:
                        want_len = int(r.headers.get("Content-Length") or 0)
                    shutil.copyfileobj(r, f)
                break
            except urllib.error.HTTPError:
                raise
            except OSError as e:          # таймаут, сброс соединения, нет сети
                if attempt == 5:
                    raise
                got = part.stat().st_size if part.exists() else 0
                log.warning("snapshot_retry  попытка %d: %s (скачано %.1f МБ)", attempt, e, got / 1e6)
                echo(f"связь прервалась ({e}), повтор {attempt + 1}/5 с места обрыва")
                time.sleep(3 * attempt)
        got_len = part.stat().st_size
        if want_len and got_len != want_len:
            part.unlink()
            log.error("snapshot_truncated  %s: получено %d из %d байт", zpath.name, got_len, want_len)
            raise SystemExit(f"загрузка оборвалась: получено {got_len} из {want_len} байт, повторите")
        os.replace(part, zpath)
        try:
            with urllib.request.urlopen(url + ".sha256", timeout=30) as r:
                sha_file.write_bytes(r.read())
        except OSError:
            pass
    except urllib.error.HTTPError as e:
        # приватный репозиторий: прямая ссылка без входа даёт 404 – качаем через gh
        repo = github_repo()
        if e.code not in (401, 403, 404) or not repo or "github.com" not in url:
            raise
        log.warning("snapshot_fetch  HTTP %s, пробую gh release download (%s)", e.code, repo)
        echo(f"прямая ссылка недоступна ({e.code}), пробую gh release download (нужен gh auth login)")
        import subprocess
        subprocess.run(["gh", "release", "download", "--repo", repo, "--clobber", "--dir", str(DIST),
                        "--pattern", zpath.name, "--pattern", zpath.name + ".sha256"], check=True)
    if sha_file.exists():
        want = sha_file.read_text(encoding="ascii").split()[0]
        got = _sha256(zpath)
        if got != want:
            zpath.unlink()
            log.error("snapshot_checksum_mismatch  %s: %s != %s", zpath.name, got, want)
            raise SystemExit(f"контрольная сумма не совпала: {got} != {want}")
        log.info("snapshot_checksum_ok  %s %.1f МБ", zpath.name, zpath.stat().st_size / 1e6)
        echo("контрольная сумма совпала")
    else:
        log.warning("snapshot_no_checksum  %s – проверка пропущена", zpath.name)
        echo("файла .sha256 нет – контрольная сумма не проверена")
    try:
        z = zipfile.ZipFile(zpath)
    except zipfile.BadZipFile:
        log.error("snapshot_bad_zip  %s", zpath.name)
        raise SystemExit(f"{zpath.name} повреждён (не zip) – скачайте заново") from None
    with z:
        bad_file = z.testzip()
        if bad_file:
            log.error("snapshot_bad_member  %s: %s", zpath.name, bad_file)
            raise SystemExit(f"{zpath.name}: повреждён файл {bad_file} – скачайте заново")
        names = z.namelist()
        bad = [n for n in names if not n.startswith(ALLOWED) or ".." in Path(n).parts]
        if bad:
            log.error("snapshot_bad_paths  %s: %s", zpath.name, bad[:5])
            raise SystemExit(f"в архиве недопустимые пути: {bad[:3]}")
        z.extractall(ROOT)
    log.info("snapshot_extracted  %s: файлов %d", zpath.name, len(names))
    echo(f"распаковано файлов: {len(names)}")


def prepare_release(raw: bool = False, echo=print) -> Path:
    """Всё для релиза в dist/: архивы, контрольные суммы и описание релиза.

    Ничего не публикует. Публикация – publish_release.ps1 (запускает человек).
    """
    import json
    from datetime import date
    from contextlib import closing

    files = pack(raw=raw, echo=echo)
    with closing(store._duck()) as con:
        counts = json.loads(con.execute("select tables from _meta").fetchone()[0])
        period = con.execute("select min(date), max(date), count(distinct oktmo) from spending_mo").fetchone()
    from src import app_service
    s = app_service.summary()
    from src.runlog import _git_rev
    rows = "\n".join(f"| `{f.name}` | {f.stat().st_size / 1e6:.1f} МБ | `{_sha256(f)[:16]}…` |" for f in files)
    notes = f"""Снимок данных SberIndex Terminal от {date.today():%d.%m.%Y} (код {_git_rev()}).

Данные: расходы на жителя по {period[2]} МО, {period[0]:%m.%Y}–{period[1]:%m.%Y}.
Лучшая модель: {s['best_model']['name'] if s['best_model'] else '–'}, MAE {s['best_model']['MAE']:.0f} ₽ \
(на {s['gain_vs_prophet'] * 100:.0f}% меньше, чем у Prophet).

## Установка на новом ПК

```powershell
git clone <адрес этого репозитория>
cd <папка>
powershell -ExecutionPolicy Bypass -File setup.ps1
```

`setup.ps1` сам скачает `{ASSET}` из последнего релиза и проверит контрольную сумму.

## Файлы

| файл | размер | sha256 |
|---|---|---|
{rows}

`{ASSET}` – готовые данные, база для приложения и отчёты (таблиц в базе: {len(counts)}).
""" + ("`sberindex-raw.zip` – исходные выгрузки для полной пересборки "
       "(`python -m worker run --force`).\n" if raw else "") + """
Источники: СберИндекс (sberindex.ru), Росстат, ЕМИСС, ЦБ РФ, Wikidata, Open-Meteo.
"""
    out = DIST / "RELEASE_NOTES.md"
    out.write_text(notes, encoding="utf-8")
    log.info("release_prepared  %s", [f.name for f in files])
    echo(f"описание релиза: {out.relative_to(ROOT)}")
    echo("опубликовать (когда решите): powershell -ExecutionPolicy Bypass -File publish_release.ps1 "
         "-Visibility private")
    return out
