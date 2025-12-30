# infra/gpu_candle_feeder.py
from __future__ import annotations

from typing import Iterable, Dict, Any, List, Optional

try:
    import cupy as cp
    from cupy.cuda import stream as _cp_stream
    _HAS_CUPY = True
except Exception:
    cp = None
    _cp_stream = None
    _HAS_CUPY = False

import numpy as np


class GPUCandleFeeder:
    """
    GPU-aware candle feeder with:
    - pinned host memory
    - CuPy streams
    - batch-based feeding
    - transparent NumPy fallback

    Responsibilities:
    - Take iterable[dict candle]
    - Convert to columnar batches
    - Overlap H2D transfer + compute
    - Call strategy.on_bar(...) in-order
    """

    def __init__(
        self,
        *,
        batch_size: int = 256,
        use_gpu: bool = True,
        streams: int = 2,
    ):
        self.batch_size = int(batch_size)
        self.use_gpu = bool(use_gpu and _HAS_CUPY)
        self.streams_n = max(1, int(streams))

        if self.use_gpu:
            self.xp = cp
            self.streams = [_cp_stream.Stream(non_blocking=True) for _ in range(self.streams_n)]
        else:
            self.xp = np
            self.streams = []

        self._stream_idx = 0

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def run(
        self,
        candles: Iterable[Dict[str, Any]],
        *,
        on_bar,
    ) -> None:
        """
        Feeds candles into strategy.on_bar(candle).

        Parameters
        ----------
        candles : iterable of candle dicts
        on_bar  : callable(candle_dict)
        """

        batch: List[Dict[str, Any]] = []

        for c in candles:
            batch.append(c)
            if len(batch) >= self.batch_size:
                self._process_batch(batch, on_bar)
                batch.clear()

        if batch:
            self._process_batch(batch, on_bar)

        # ensure all GPU work finished
        if self.use_gpu:
            cp.cuda.Stream.null.synchronize()

    # ------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------
    def _process_batch(
        self,
        batch: List[Dict[str, Any]],
        on_bar,
    ) -> None:
        if not self.use_gpu:
            # CPU fallback: identical semantics
            for c in batch:
                on_bar(c)
            return

        # round-robin stream
        stream = self.streams[self._stream_idx]
        self._stream_idx = (self._stream_idx + 1) % self.streams_n

        with stream:
            # ------------------------------
            # 1) pinned host buffers
            # ------------------------------
            n = len(batch)

            ts_h = cp.cuda.alloc_pinned_memory(n * 8)
            o_h  = cp.cuda.alloc_pinned_memory(n * 4)
            h_h  = cp.cuda.alloc_pinned_memory(n * 4)
            l_h  = cp.cuda.alloc_pinned_memory(n * 4)
            c_h  = cp.cuda.alloc_pinned_memory(n * 4)
            v_h  = cp.cuda.alloc_pinned_memory(n * 4)

            ts_np = np.frombuffer(ts_h, dtype=np.int64, count=n)
            o_np  = np.frombuffer(o_h,  dtype=np.float32, count=n)
            h_np  = np.frombuffer(h_h,  dtype=np.float32, count=n)
            l_np  = np.frombuffer(l_h,  dtype=np.float32, count=n)
            c_np  = np.frombuffer(c_h,  dtype=np.float32, count=n)
            v_np  = np.frombuffer(v_h,  dtype=np.float32, count=n)

            for i, cd in enumerate(batch):
                ts_np[i] = int(cd.get("timestamp") or cd.get("timestamp_ms") or 0)
                o_np[i]  = float(cd.get("open", 0.0))
                h_np[i]  = float(cd.get("high", 0.0))
                l_np[i]  = float(cd.get("low", 0.0))
                c_np[i]  = float(cd.get("close", 0.0))
                v_np[i]  = float(cd.get("volume", 0.0))

            # ------------------------------
            # 2) async H2D
            # ------------------------------
            ts_d = cp.asarray(ts_np)
            o_d  = cp.asarray(o_np)
            h_d  = cp.asarray(h_np)
            l_d  = cp.asarray(l_np)
            c_d  = cp.asarray(c_np)
            v_d  = cp.asarray(v_np)

        # ------------------------------
        # 3) ordered feed (semantics-safe)
        # ------------------------------
        # NOTE:
        # We do NOT call on_bar inside the stream.
        # Strategy remains sequential & deterministic.
        stream.synchronize()

        for i in range(len(batch)):
            on_bar({
                "timestamp": int(ts_d[i].item()),
                "open": float(o_d[i].item()),
                "high": float(h_d[i].item()),
                "low": float(l_d[i].item()),
                "close": float(c_d[i].item()),
                "volume": float(v_d[i].item()),
            })
