# dashboard_rt/app.py
import asyncio
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from dashboard_rt.state import read_runtime_state
from dashboard_rt.ws import connect, disconnect, broadcast

RT_INTERVAL_SEC = 1.0  # debe coincidir con runtime_state_interval del shadow

BASE_DIR = os.path.dirname(__file__)

app = FastAPI(title="Shadow RT Dashboard")

# Static files
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "static")),
    name="static",
)


@app.get("/")
def index():
    """Landing del dashboard"""
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await connect(ws)
    try:
        while True:
            # mantenemos la conexión viva
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        disconnect(ws)


async def rt_loop():
    """
    Loop RT:
    - lee logs/runtime_state.json
    - broadcast a todos los WS conectados
    - nunca debe romper (dashboard always-on)
    """
    while True:
        try:
            data = read_runtime_state()
            if data is not None:
                await broadcast(data)
        except Exception as e:
            # IMPORTANTE: nunca romper el loop RT
            print("[RT] read/broadcast error:", e)

        await asyncio.sleep(RT_INTERVAL_SEC)


@app.on_event("startup")
async def startup():
    asyncio.create_task(rt_loop())
