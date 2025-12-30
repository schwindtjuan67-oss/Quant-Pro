
# Live/event_bus.py
# Mini event bus muy simple para desacoplar el scalper de Telegram / dashboard.

from typing import Callable, Dict, List, Any


class EventBus:
    """Bus de eventos en memoria (single-process, thread-safe básico)."""

    def __init__(self) -> None:
        self._subs: Dict[str, List[Callable[[Any], None]]] = {}

    def subscribe(self, event_type: str, callback: Callable[[Any], None]) -> None:
        """Registra un callback para un tipo de evento dado."""
        if event_type not in self._subs:
            self._subs[event_type] = []
        self._subs[event_type].append(callback)

    def emit(self, event_type: str, payload: Any) -> None:
        """Emite un evento a todos los suscriptores registrados."""
        callbacks = self._subs.get(event_type, [])
        for cb in callbacks:
            try:
                cb(payload)
            except Exception as e:
                # No queremos que un fallo en un listener rompa el resto del flujo
                print(f"[EventBus] Error en listener de '{event_type}': {e}")
