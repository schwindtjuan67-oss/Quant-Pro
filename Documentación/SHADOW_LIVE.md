# Shadow / Live Engine

## Componentes principales
- Estrategia: `Live/hybrid_scalper_pro.py`
- Riesgo: `Live/risk_manager.py`
- Logger: `Live/logger_pro.py`
- Data feed / WS: `Live/ws_futures_1m.py`, `Live/stream_klines.py`
- Adapter / integración: `Live/hybrid_scalper_pro_adapter.py`
- Estado/hotswap: `analysis/config_state_store.py`
- Aplicación de params en caliente: `Live/hybrid_param_hotswap.py`

## Cómo consume parámetros del pipeline (contrato)
Actualmente el research produce JSONs en `results/promotions/`, pero el engine no los lee “mágicamente”.

### Integración recomendada (patrón)
1) Seleccionar un candidato final del JSON (faseC o faseB)
2) Construir un dict `params` (clave: `UPPER_SNAKE` tuneables y JSON-safe)
3) Inyectarlo en el engine:
   - via config del engine (ej. `engine.config["params"] = params`)
   - o via state/hotswap (archivo en `logs/active_config_state.json` o `logs/hotswap_state.json`)

El adapter `Live/hybrid_scalper_pro_adapter.py` tiene una rutina `_apply_pipeline_params_from_engine()`
que aplica `engine.config["params"]` o `engine.config["strategy_params"]` a la instancia de `HybridScalperPRO`.

## Shadow vs Live
- Shadow: simula/ejecuta sin exposición real (o con constraints) para validar comportamiento y logging
- Live: ejecución real; debe usar kill-switch, risk limits, y auditoría estricta
