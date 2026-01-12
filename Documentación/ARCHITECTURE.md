# Arquitectura

Quant-Pro separa estrictamente:

## 1) Research / Pipeline
Objetivo: producir **artefactos** (JSON/CSV) que representen parámetros robustos y auditables.

Entrypoint típico:
- `bat/run_autoloop_A_B_C.bat`

Etapas:
- **Fase A (robust search)**: `analysis/robust_optimizer.py`
- **Post-A**: `analysis/analysis_post_robust.py`
- **Promoción A→B**: `analysis/promoter_faseA_to_B.py` (+ reglas `configs/promotion_rules_A.json`)
- **Fase B (risk calibration)**: `analysis/stage_b_risk_calibration.py`
- **Fase C (pipeline eval final)**: `analysis/stage_c_pipeline_eval.py`

Outputs principales:
- `results/robust/*.json`
- `results/promotions/faseA_promoted.json`
- `results/promotions/faseB_promoted.json` + `faseB_report.csv`
- `results/promotions/faseC_promoted.json` + `faseC_report.csv`
- `results/pipeline_stageC_trades/` (trades por candidato)

## 2) Shadow / Live Engine
Objetivo: ejecutar/simular estrategia con **control de riesgo** y logging, consumiendo parámetros validados.

Módulos clave:
- Estrategia: `Live/hybrid_scalper_pro.py`
- Riesgo: `Live/risk_manager.py`
- Logging: `Live/logger_pro.py`
- Feed/WS: `Live/ws_futures_1m.py`, `Live/stream_klines.py`
- Adapter: `Live/hybrid_scalper_pro_adapter.py`
- State store (hotswap/state): `analysis/config_state_store.py`
- HotSwap applier: `Live/hybrid_param_hotswap.py`

## Contrato entre mundos (Artifacts Contract)
El pipeline produce JSONs con parámetros sobrevivientes.
El engine debe consumirlos vía:
- “config injection” (params dentro de la config del engine)
- o “hotswap state” (archivos state para cambios en caliente)
(ver `docs/ARTIFACTS.md` y `docs/SHADOW_LIVE.md`)
