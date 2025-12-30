# dashboard_rt/ws.py
import json
import asyncio
from typing import Set
from fastapi import WebSocket

_clients: Set[WebSocket] = set()
_lock = asyncio.Lock()


async def connect(ws: WebSocket):
    await ws.accept()
    async with _lock:
        _clients.add(ws)


def disconnect(ws: WebSocket):
    try:
        _clients.discard(ws)
    except Exception:
        pass


async def broadcast(data: dict):
    if not _clients:
        return

    msg = json.dumps(data, ensure_ascii=False)

    async with _lock:
        clients = list(_clients)

    dead = []
    for ws in clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)

    if dead:
        async with _lock:
            for ws in dead:
                _clients.discard(ws)
