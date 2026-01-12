# Artefactos (Research → Engine)

## 1) Robust outputs (Fase A)
Path:
- `results/robust/robust_<WINDOW>_seed<SEED>.json`

Contenido típico:
- métricas agregadas por folds/ventana
- score
- params candidatos / best params

## 2) Promotions outputs (A/B/C)
Paths:
- `results/promotions/faseA_promoted.json`
- `results/promotions/faseB_promoted.json`
- `results/promotions/faseC_promoted.json`

Recomendación:
- considerar `faseC_promoted.json` como “final survivors”

## 3) Reportes CSV
- `results/promotions/faseB_report.csv`
- `results/promotions/faseC_report.csv`

## 4) Trades por candidato (auditoría)
- `results/pipeline_stageC_trades/`

## Contrato mínimo para ejecución
Para Shadow/Live, lo mínimo a consumir es un dict:

```json
{
  "params": {
    "EMA_FAST": 12,
    "EMA_SLOW": 48,
    "ATR_PERIOD": 14,
    "ATR_STOP_MULT": 2.2,
    "VWAP_WINDOW": 80,
    "VWAP_BAND_MULT": 1.6
  },
  "meta": {
    "source": "results/promotions/faseC_promoted.json",
    "seed_agreement": true,
    "window_agreement": true
  }
}
