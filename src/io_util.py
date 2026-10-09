"""Запись файлов, не меняющая их без нужды.

Воркер решает, что пересчитывать, по времени изменения входных файлов.
Если загрузчик каждый раз перезаписывает файл тем же содержимым, время
изменения обновляется, и воркер зря запускает пересчёт моделей (при
скачивании раз в 6 часов – четыре раза в сутки). Поэтому файл пишется
только когда содержимое действительно другое.
"""

import os
from pathlib import Path


def write_if_changed(path: Path, data: bytes) -> bool:
    """True – файл записан (новый или изменился), False – содержимое то же."""
    path = Path(path)
    if path.exists() and path.stat().st_size == len(data) and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)            # атомарно: читатель не увидит полузаписанный файл
    return True
