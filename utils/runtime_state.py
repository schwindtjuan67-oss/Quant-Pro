import json
import os
import tempfile
import time
from typing import Dict, Any


def safe_write_json(path: str, data: Dict[str, Any], retries: int = 5) -> None:
    """
    Escritura atómica y segura para Windows.
    Cada llamada usa un .tmp único → no colisiona entre procesos.
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    for attempt in range(retries):
        fd, tmp_path = tempfile.mkstemp(
            prefix=os.path.basename(path) + ".",
            suffix=".tmp",
            dir=os.path.dirname(path),
        )

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_path, path)
            return  # ✅ éxito

        except PermissionError:
            time.sleep(0.05 * (attempt + 1))

        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    # último recurso: escribir directo (evita spam de errores)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
