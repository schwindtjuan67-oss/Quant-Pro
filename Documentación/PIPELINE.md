# Pipeline Research (A → B → C)

## Inputs base
- Dataset: típicamente `datasets/SOLUSDT/1m`
- Config base: `configs/pipeline_research_backtest.json`
- Ventanas y seeds: controlados por el `.bat`

---

## Fase A — Robust Optimizer
Script:
- `analysis/robust_optimizer.py`

Invocación (conceptual):
- `python -m analysis.robust_optimizer --data <DATASET> --base-config <BASE_CFG> --window <YYYY-MM_YYYY-MM> --seed <SEED> --samples <N> --out <OUT_JSON>`

Output:
- `results/robust/robust_<WINDOW>_seed<SEED>.json`

---

## Post-A — Agregación
Script:
- `analysis/analysis_post_robust.py`

Lee:
- `results/robust/robust_*_seed*.json`

Escribe:
- `results/promotions/faseA_promoted.json` (lista/summary de candidatos)

---

## Promoción A → B (gates de estabilidad)
Script:
- `analysis/promoter_faseA_to_B.py`

Reglas:
- `configs/promotion_rules_A.json` (usado por el bat)

Lee (según default/compat):
- `results/post_analysis_summary.json` o `results/promotions/faseA_promoted.json`

Escribe:
- `results/promotions/faseA_promoted.json` (output final A→B)

---

## Fase B — Risk Calibration (holdout + estabilidad)
Script:
- `analysis/stage_b_risk_calibration.py`

Inputs:
- `--fasea results/promotions/faseA_promoted.json`
- `--base-config configs/pipeline_research_backtest.json`
- dataset

Outputs:
- `results/promotions/faseB_promoted.json`
- `results/promotions/faseB_report.csv`

---

## Fase C — Pipeline Eval Final
Script:
- `analysis/stage_c_pipeline_eval.py`

Inputs:
- `--faseb results/promotions/faseB_promoted.json`
- `--data datasets/...`
- `--base-config configs/...`

Outputs:
- `results/promotions/faseC_promoted.json`
- `results/promotions/faseC_report.csv`
- `results/pipeline_stageC_trades/` (trades por candidato)

---

## Resultado final del research
El artefacto que representa “lo que sobrevivió al test” es:
- `results/promotions/faseC_promoted.json` (ideal)
o el último promoted disponible según tu flujo.
