# Quant-Pro

Quant-Pro es un **ecosistema cuantitativo de trading algorítmico** con separación estricta entre:

- **Research / Pipeline**: optimización, robustez, calibración de riesgo, evaluación multi-etapa (A→B→C)
- **Shadow/Live**: motor estratégico + ejecución/simulación que consume parámetros validados

El objetivo no es “hacer un bot”, sino **producir artefactos auditables (JSON/CSV)** que sobreviven filtros de robustez y luego se consumen en ejecución controlada.

---

## Estructura del repo

├─ analysis/ # Research pipeline (A/B/C) + tooling de research
├─ configs/ # Configs base + reglas de promoción A/B/C
├─ bat/ # Orquestación Windows (autoloops, post, etc.)
├─ results/ # Artefactos generados por el pipeline
├─ Live/ # Shadow/Live engine (estrategia, risk, logger, ws)
├─ dashboard_rt/ # Dashboards / realtime tools
├─ web_dashboard/ # UI web (coming soon)
├─ datasets/ # Datos históricos (no versionados en el repo debido al peso)
├─ logs/ # Logs de ejecución
└─ tools/ utils/ Validators/ # Utilidades auxiliares