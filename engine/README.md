# wbj — Motor de cómputo de Warren Buffett Jr

Motor determinista en Python que implementa la metodología **Ruta 2030 Wall
Street Agent System v2.0.0**: providers de datos, motores de indicadores /
niveles / valuación, los 6 especialistas, el overlay de juicio, la
agregación con gates y el render del reporte final.

> Regla innegociable: **sin evidencia, no hay número.** Cualquier métrica que
> el motor no puede calcular queda `NOT_SCORABLE` — nunca se rellena con un
> valor inventado.

## Instalación

Requisitos: Python 3.11+.

```bash
cd engine
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q          # 350 tests
```

## Claves de API (opcional)

Los fundamentales base salen de **SEC EDGAR** (gratis, sin key). Para
enriquecer con FMP/FinnHub/FRED, copia tus claves en `API/.env` (en la raíz
del repo, ya está en `.gitignore` — nunca se sube):

```
FMP_API_KEY=tu_clave
FINNHUB_API_KEY=tu_clave
FRED_API_KEY=tu_clave
```

Nunca se imprime material de claves en ningún archivo de `Reportes/` ni
`engine/cache/`.

## Uso por etapas

El pipeline completo (v2.0.0) vive en `wbj.pipeline` y se expone por CLI:

```bash
wbj full NVDA --offline           # pipeline completo con el packet golden (sin red)
wbj full NVDA                      # en vivo (requiere red + EDGAR/FMP)
wbj full NVDA --overlay juicios.json   # inyecta juicios cualitativos
wbj aggregate NVDA --offline       # imprime el FinalReport en JSON
wbj report NVDA --offline          # escribe report.md + report.json + charts
```

Etapas programáticas (`wbj/pipeline.py`):

| Etapa | Qué hace |
|---|---|
| `stage_packet` | Arma/carga el packet de datos (validación, facts table, staleness, hash). |
| `stage_compute` | Corre los 6 especialistas **independientes** y congela sus outputs. |
| `stage_aggregate` | Aplica overlay → overrides → gates → contradicciones → síntesis. |
| `stage_report` | Genera charts (reglas de visualización) y `report.md` / `report.json`. |
| `run_all` | Orquesta todo y guarda en `Reportes/<TICKER>/<YYYY-MM-DD>/`. |

## Flujo de overlay (sub-agentes de Claude)

Las métricas cualitativas (clasificación de moat, tiers de TAM, catalizadores,
thesis-killers) se emiten como `JudgmentRequest` y se escriben en
`cache/<T>/artifacts/judgment_requests.json`. Un sub-agente (o Victor) las
responde en un JSON de `Judgment` (con `evidence_class` y `source`
obligatorios) y se re-inyectan con `--overlay`, que **re-puntúa** la dimensión
afectada y recomputa coverage y el hash del output.

## Modo offline

`--offline` carga el packet golden `tests/fixtures/packet/NVDA_packet.json`
(o `cache/<T>/packet.json` si existe) sin tocar la red — así el end-to-end es
reproducible y testeable sin API keys. La estabilidad del reporte se ancla en
`tests/fixtures/golden/NVDA_report.json`.

## Smoke test en vivo (correr localmente)

En una máquina con red y `API/.env` configurado:

```bash
wbj full NVDA                       # debe renderizar el reporte
grep -r "$FMP_API_KEY" ../Reportes ./cache   # DEBE salir vacío (sin fugas de claves)
```

Si un campo de la API en vivo no coincide con los fixtures, se corrige con su
propio test usando un fixture capturado (nunca datos en vivo en los tests).

## Arquitectura del código

```
wbj/
├── core/          Value/null-states, fórmulas, scoring, confianza
├── providers/     FMP, EDGAR, FinnHub, FRED + cache
├── packet/        builder, reconcile (jerarquía de fuentes), staleness
├── engines/       indicators, levels_engine, valuation_engine
├── specialists/   common (envelope) + los 6 agentes
├── overlay/       merge de juicios
├── aggregate/     gates, overrides, contradiction, synthesis, assemble
├── report/        charts, render
└── pipeline.py    orquestación por etapas
```
