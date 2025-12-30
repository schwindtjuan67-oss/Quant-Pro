# Live/hybrid_param_hotswap.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


def _is_param_key(k: Any) -> bool:
    if not isinstance(k, str):
        return False
    k = k.strip()
    if not k:
        return False
    # Convención: parámetros "tuneables" suelen ser UPPER_SNAKE
    # (igual dejamos pasar números/underscore)
    return all(ch.isalnum() or ch == "_" for ch in k)


def _stable_hash(d: Dict[str, Any]) -> str:
    try:
        return json.dumps(d or {}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(d)


@dataclass
class HotSwapResult:
    applied: bool
    reason: str
    new_params_hash: str
    changed: Dict[str, Tuple[Any, Any]]
    skipped: Dict[str, Any]


class HybridHotSwapApplier:
    """
    Aplica params sin romper un trade abierto:

    - Por default, setea en instancia.
    - Si el atributo existe en la clase, también lo setea en la clase
      (útil si Hybrid usa class-params tipo Strategy).
    - Solo aplica keys 'param-like' y valores JSON-safe (numbers/bool/str).
    """

    def __init__(self, *, only_uppercase: bool = True):
        self.only_uppercase = bool(only_uppercase)

    def apply(self, hybrid_obj: Any, params: Dict[str, Any]) -> HotSwapResult:
        if hybrid_obj is None:
            return HotSwapResult(False, "hybrid_none", _stable_hash(params), {}, dict(params or {}))

        p = dict(params or {})
        new_hash = _stable_hash(p)

        changed: Dict[str, Tuple[Any, Any]] = {}
        skipped: Dict[str, Any] = {}

        # Reglas defensivas
        for k, v in p.items():
            if not _is_param_key(k):
                skipped[str(k)] = v
                continue

            kk = k.strip()
            if self.only_uppercase and kk != kk.upper():
                # si te pasaron keys mixtas, no las tocamos
                skipped[kk] = v
                continue

            # valores permitidos
            if not isinstance(v, (int, float, bool, str)) and v is not None:
                skipped[kk] = v
                continue

            # 1) instancia
            old_inst = getattr(hybrid_obj, kk, None)
            inst_has = hasattr(hybrid_obj, kk)

            # 2) clase (para Strategy-style params)
            cls = hybrid_obj.__class__
            cls_has = hasattr(cls, kk)
            old_cls = getattr(cls, kk, None) if cls_has else None

            # Si no existe ni en instancia ni en clase, igual podemos setear en instancia
            # (no rompe y puede ser útil si Hybrid consume getattr dinámico).
            try:
                setattr(hybrid_obj, kk, v)
            except Exception:
                skipped[kk] = v
                continue

            # si existía en clase, también lo actualizamos
            if cls_has:
                try:
                    setattr(cls, kk, v)
                except Exception:
                    pass

            # registrar cambios reales
            # priorizamos reportar el "old" más relevante
            old = old_inst if inst_has else (old_cls if cls_has else None)
            if old != v:
                changed[kk] = (old, v)

        return HotSwapResult(
            applied=True,
            reason="ok",
            new_params_hash=new_hash,
            changed=changed,
            skipped=skipped,
        )
