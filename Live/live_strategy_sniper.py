import math
import time


class LiveSniperStrategy:
    """
    Estrategia LIVE tipo sniper:
      - Muy selectiva (1–3 trades/día)
      - Usa tendencia (EMAs), VWAP y slot para Delta
      - Stop basado en ATR
      - TP basado en RR (risk/reward)
      - Respeta límites de DD, trades diarios y racha (streak)
    """

    def __init__(self, client, config):
        self.client = client
        self.config = config

        # ----- Parámetros de trading / riesgo -----
        self.symbol = config.get("symbol", "SOLUSDC")
        self.rr = float(config.get("rr", 2.0))                   # Risk/Reward
        self.vol_target = float(config.get("vol_target", 0.02))  # % de equity arriesgado

        self.max_daily_dd_pct = float(config.get("max_daily_dd_pct", 3.0))
        self.max_trades_per_day = int(config.get("max_trades_per_day", 3))
        self.max_streak = int(config.get("max_streak", 3))

        # ----- Filtros técnicos -----
        self.use_trend_filter = bool(config.get("trend_filter", True))
        self.use_vwap_filter = bool(config.get("vwap_filter", True))
        self.use_absorption_filter = bool(config.get("absorption_filter", True))

        # Parámetros técnicos
        self.ema_fast_len = int(config.get("ema_fast_len", 10))
        self.ema_slow_len = int(config.get("ema_slow_len", 60))
        self.atr_period = int(config.get("atr_period", 14))
        self.atr_stop_mult = float(config.get("atr_stop_mult", 1.5))

        # Delta mínimo para considerar entrada
        self.delta_min = float(config.get("delta_min", 800000.0))

        # ----- Estado interno -----
        self.ema_fast = None
        self.ema_slow = None

        self.atr = None
        self.atr_initialized = False
        self.atr_counter = 0

        self.prev_close = None

        # VWAP acumulado
        self.cum_pv = 0.0
        self.cum_vol = 0.0
        self.vwap = None

        # Slot para Delta (lo llenará delta_live.py más adelante)
        self.delta_imbalance = 0.0

        # Posición actual (simple: una sola posición a la vez)
        self.position = None  # dict: {side, qty, entry, sl, tp, opened_at}

    # ============================================================
    # Interfaz para que otro módulo alimente delta en tiempo real
    # ============================================================
    def update_delta(self, delta_imbalance: float):
        """
        Permite que un módulo externo (delta_live) actualice el desequilibrio
        de agresión comprador-vendedor.
        """
        self.delta_imbalance = delta_imbalance

    # ============================================================
    # Cálculo de contexto por vela
    # ============================================================
    def build_context(self, candle):
        """
        Recibe una vela:
          candle = {
              "open": ...,
              "high": ...,
              "low": ...,
              "close": ...,
              "volume": ...
          }
        Devuelve un diccionario ctx con todo lo necesario para decidir.
        """
        close = float(candle["close"])
        high = float(candle["high"])
        low = float(candle["low"])
        vol = float(candle["volume"])

        # ----- EMAs -----
        alpha_fast = 2.0 / (self.ema_fast_len + 1.0)
        alpha_slow = 2.0 / (self.ema_slow_len + 1.0)

        if self.ema_fast is None:
            # Inicializar con el primer close
            self.ema_fast = close
            self.ema_slow = close
        else:
            self.ema_fast = self.ema_fast + alpha_fast * (close - self.ema_fast)
            self.ema_slow = self.ema_slow + alpha_slow * (close - self.ema_slow)

        trend_long = self.ema_fast > self.ema_slow
        trend_short = self.ema_fast < self.ema_slow

        # ----- ATR (Wilder) -----
        if self.prev_close is None:
            tr = high - low
        else:
            tr = max(
                high - low,
                abs(high - self.prev_close),
                abs(low - self.prev_close),
            )

        if not self.atr_initialized:
            # Fase de calentamiento
            if self.atr is None:
                self.atr = tr
            else:
                self.atr = ((self.atr * self.atr_counter) + tr) / (self.atr_counter + 1)
            self.atr_counter += 1
            if self.atr_counter >= self.atr_period:
                self.atr_initialized = True
        else:
            # Fórmula de Wilder
            self.atr = ((self.atr * (self.atr_period - 1)) + tr) / self.atr_period

        self.prev_close = close

        # ----- VWAP acumulativo -----
        self.cum_pv += close * vol
        self.cum_vol += vol
        if self.cum_vol > 0:
            self.vwap = self.cum_pv / self.cum_vol
        else:
            self.vwap = close

        vwap_long_ok = close > self.vwap
        vwap_short_ok = close < self.vwap

        # ----- Estado de riesgo global -----
        status = self.client.get_status()
        can_trade = (
            status["safety_enabled"]
            and status["drawdown_pct"] < self.max_daily_dd_pct
            and status["trades_today"] < self.max_trades_per_day
            and abs(status["streak"]) <= self.max_streak
        )

        ctx = {
            "close": close,
            "high": high,
            "low": low,
            "volume": vol,

            "ema_fast": self.ema_fast,
            "ema_slow": self.ema_slow,
            "trend_long": trend_long,
            "trend_short": trend_short,

            "atr": self.atr,
            "atr_ready": self.atr_initialized,

            "vwap": self.vwap,
            "vwap_long_ok": vwap_long_ok,
            "vwap_short_ok": vwap_short_ok,

            "delta_imbalance": self.delta_imbalance,
            "can_trade": can_trade,
            "status": status,
        }

        return ctx

    # ============================================================
    # Lógica principal por cierre de vela
    # ============================================================
    def on_new_candle(self, candle, ctx):
        close = ctx["close"]

        # Log resumido (podemos afinar después)
        print("\n===== VELA CERRADA (SNIPER) =====")
        print(
            "Close=%.4f | EMA_fast=%.4f | EMA_slow=%.4f | ATR=%.4f"
            % (close, ctx["ema_fast"], ctx["ema_slow"], ctx["atr"] if ctx["atr"] else 0)
        )
        print(
            "VWAP=%.4f | Trend: long=%s short=%s | VWAP long=%s short=%s"
            % (
                ctx["vwap"],
                ctx["trend_long"],
                ctx["trend_short"],
                ctx["vwap_long_ok"],
                ctx["vwap_short_ok"],
            )
        )
        print(
            "Delta_imbalance=%.0f | can_trade=%s | trades_dia=%d | DD_pct=%.2f%%"
            % (
                ctx["delta_imbalance"],
                ctx["can_trade"],
                ctx["status"]["trades_today"],
                ctx["status"]["drawdown_pct"],
            )
        )
        print("=================================")

        # Si hay posición abierta, manejamos SL/TP
        if self.position is not None:
            self._manage_open_position(candle, ctx)
            return

        # Si no hay posición, ver si podemos abrir una
        if not ctx["can_trade"]:
            print("❌ No se puede tradear: safety o límites de riesgo.")
            return

        if not ctx["atr_ready"] or ctx["atr"] is None:
            print("⏳ ATR aún no listo, se espera más histórico.")
            return

        # =============================
        # REGLAS DE ENTRADA SNIPER
        # =============================

        # 1) Filtro de tendencia
        long_ok = ctx["trend_long"]
        short_ok = ctx["trend_short"]

        if self.use_trend_filter:
            if not (long_ok or short_ok):
                print("❌ Tendencia indefinida, se descarta entrada.")
                return

        # 2) Filtro VWAP
        if self.use_vwap_filter:
            if long_ok and not ctx["vwap_long_ok"]:
                print("❌ LONG rechazado: precio por debajo de VWAP.")
                long_ok = False
            if short_ok and not ctx["vwap_short_ok"]:
                print("❌ SHORT rechazado: precio por encima de VWAP.")
                short_ok = False

        # 3) Filtro Delta / Absorción (por ahora, solo umbral simple)
        if self.use_absorption_filter:
            if abs(ctx["delta_imbalance"]) < self.delta_min:
                print(
                    "❌ Delta insuficiente: |%.0f| < %.0f"
                    % (ctx["delta_imbalance"], self.delta_min)
                )
                long_ok = False
                short_ok = False
            else:
                print("✅ Delta filtro OK (%.0f >= %.0f)" %
                      (abs(ctx["delta_imbalance"]), self.delta_min))

        # Decidir lado final
        side = None
        if long_ok and not short_ok:
            side = "BUY"
        elif short_ok and not long_ok:
            side = "SELL"
        else:
            # En caso de conflicto o ninguno
            print("❌ No se definió lado claro (long_ok=%s, short_ok=%s)." %
                  (long_ok, short_ok))
            return

        # =============================
        # Cálculo de tamaño, SL y TP
        # =============================
        atr = ctx["atr"]
        status = ctx["status"]
        balance = status["balance"]

        # Distancia al stop en precio
        stop_dist = atr * self.atr_stop_mult
        if stop_dist <= 0:
            print("❌ stop_dist inválido, ATR=%.4f" % atr)
            return

        # Tamaño en base al riesgo (vol_target%)
        # riesgo $ por trade:
        risk_amount = balance * self.vol_target
        qty = max(risk_amount / stop_dist, 0.0)

        if qty <= 0:
            print("❌ qty calculado inválido (%.6f)." % qty)
            return

        # SL y TP
        if side == "BUY":
            sl_price = close - stop_dist
            tp_price = close + stop_dist * self.rr
        else:
            sl_price = close + stop_dist
            tp_price = close - stop_dist * self.rr

        print("✅ Señal SNIPER %s | qty=%.4f | SL=%.4f | TP=%.4f" %
              (side, qty, sl_price, tp_price))

        # En este punto llamamos al cliente para abrir la posición REAL
        try:
            self.client.open_position(
                side=side,
                qty=qty,
                entry_price=close,
                sl_price=sl_price,
                tp_price=tp_price,
            )
        except AttributeError:
            print(
                "⚠️ client.open_position aún no implementado en BinanceClient. "
                "Se abrirá más adelante."
            )
        except Exception as e:
            print("❌ Error al abrir posición LIVE:", e)
            return

        # Guardamos estado local
        self.position = {
            "side": side,
            "qty": qty,
            "entry": close,
            "sl": sl_price,
            "tp": tp_price,
            "opened_at": time.time(),
        }

        print("📌 Posición registrada en estrategia:", self.position)

    # ============================================================
    # Manejo de posición abierta (SL/TP por vela)
    # ============================================================
    def _manage_open_position(self, candle, ctx):
        if self.position is None:
            return

        high = ctx["high"]
        low = ctx["low"]
        side = self.position["side"]

        hit_sl = False
        hit_tp = False

        if side == "BUY":
            if low <= self.position["sl"]:
                hit_sl = True
            elif high >= self.position["tp"]:
                hit_tp = True
        else:  # SELL
            if high >= self.position["sl"]:
                hit_sl = True
            elif low <= self.position["tp"]:
                hit_tp = True

        if not (hit_sl or hit_tp):
            print("📎 Posición abierta sigue viva. side=%s" % side)
            return

        reason = "TP" if hit_tp else "SL"
        exit_price = self.position["tp"] if hit_tp else self.position["sl"]

        print("🔔 Cierre de posición por %s a %.4f" % (reason, exit_price))

        try:
            self.client.close_position(reason=reason, price=exit_price)
        except AttributeError:
            print(
                "⚠️ client.close_position aún no implementado en BinanceClient. "
                "Se añadirá más adelante."
            )
        except Exception as e:
            print("❌ Error al cerrar posición LIVE:", e)

        self.position = None
