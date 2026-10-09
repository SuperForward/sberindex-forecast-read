"""python -m server [--host 127.0.0.1] [--port 8000] [--allow-write]"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m server")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 – доступ из сети (например, в Docker)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--allow-write", action="store_true", help="разрешить пересчёт из браузера (только локально)")
    a = ap.parse_args()
    if a.allow_write:
        os.environ["SBERINDEX_ALLOW_WRITE"] = "1"
    from app.logging_setup import setup_logging
    setup_logging()
    import uvicorn
    print(f"SberIndex Terminal: http://{a.host}:{a.port}   API: http://{a.host}:{a.port}/docs"
          f"   {'запись разрешена' if a.allow_write else 'только чтение'}")
    uvicorn.run("server.app:app", host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
