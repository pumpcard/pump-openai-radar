# openai-radar

**OpenAI infrastructure FinOps SDK** — part of the [Hyperscaler Radar](https://github.com/gomorsmi) suite.

[![PyPI](https://img.shields.io/pypi/v/openai-radar)](https://pypi.org/project/openai-radar/)
[![Python](https://img.shields.io/pypi/pyversions/openai-radar)](https://pypi.org/project/openai-radar/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Scan your OpenAI org for assistants, vector stores, fine-tunes, batch jobs, and token usage.
Detect external service relationships in assistant instructions, flag cost anomalies via a findings engine,
and export inventory to CSV or a draw.io architecture diagram — all through a clean Python SDK
that mirrors the `openai-agents` Runner API.

---

## Install

```bash
pip install openai-radar                   # SDK + CLI + CSV and draw.io export
pip install openai-radar[agents]           # + openai-agents integration
pip install openai-radar[all]              # everything
```

Requires Python 3.10+.

CSV and draw.io export are part of the base install — no extras needed, same as
the rest of the Radar suite. The `[csv]` and `[drawio]` extras still resolve (as
no-ops) so older pins keep working.

---

## Quick-start

```python
from openai_radar import RadarClient, Runner

# Reads OPENAI_API_KEY from environment
client = RadarClient()

result = Runner.run_sync(client)

print(result.summary())
result.export_csv("./out/")
result.export_drawio("./out/openai_arch.drawio")
```

### With admin key (org-wide visibility)

```python
client = RadarClient(
    api_key="sk-proj-...",
    admin_key="sk-admin-...",  # unlocks cross-project usage data
)
result = Runner.run_sync(client, config)
```

### Scoped to a project

```python
from openai_radar import RadarClient, Runner, RunConfig

client = RadarClient()
config = RunConfig(project_id="proj_xxx", usage_lookback_days=7)
result = Runner.run_sync(client, config)
```

---

## CLI

```bash
# Full scan — findings table to stdout
openai-radar run

# CSVs + draw.io diagram
openai-radar run --csv-dir ./out --drawio-file arch.drawio

# JSON instead of a table
openai-radar run --output json --out-file scan.json

# Admin key for org-wide usage data
openai-radar run --admin-key sk-admin-... --csv-dir ./out

# Findings only
openai-radar findings

# Scope to a project, 7-day lookback
openai-radar run --project proj_xxx --lookback 7

# Push the org cost report to Pump (admin key required)
openai-radar run --admin-key sk-admin-... --upload-token <TOKEN>

# Log in to Pump, check the stored token, or forget it
openai-radar login
openai-radar status
openai-radar logout

# Print the version
openai-radar version
```

Flags follow the Radar suite convention: `--output/-o` selects `table` or `json`,
`--out-file` writes the JSON payload, `--csv-dir` writes per-resource CSVs.
`--upload-token` also writes `report.csv` and uploads it. `--report-file` chooses
that path; with `--csv-dir` and no `--report-file` it is `{csv-dir}/report.csv`.

---

## Pump onboarding

The token is exchanged for a presigned S3 URL, and only the CSV leaves the machine.

1. In the Pump app, mint an upload token (`POST /api/v1/estimate/radar/mint`). Pump
   shows a ready-to-paste command.
2. Run it with an OpenAI admin key:

   ```bash
   openai-radar run --admin-key sk-admin-... --upload-token <TOKEN>
   ```

   This scans the org, pulls daily costs from `/organization/costs`, writes
   `report.csv`, and uploads that file. `--csv-dir` and `--drawio-file` still
   work on the same command. Pass `--report-file` to choose where the CSV is written.

3. Pump detects `report.csv` and runs its analysis.

`report.csv` columns:

| Column    | Meaning                                   |
| --------- | ----------------------------------------- |
| Date      | UTC day (`YYYY-MM-DD`)                    |
| ProjectID | OpenAI project, or `-`                    |
| LineItem  | Cost line item (model and token category) |
| Amount    | Non-zero cost, 6 decimal places           |
| Currency  | e.g. `USD`                                |

Zero-cost buckets are omitted. The token carries no company id — Pump binds the
company and the S3 key server-side. The exchange defaults to `https://api.pump.co`:

```bash
openai-radar run --admin-key sk-admin-... --upload-token <TOKEN> --api-base http://localhost:8001
# or
PUMP_API_BASE=http://localhost:8001 openai-radar run --admin-key sk-admin-... --upload-token <TOKEN>
```

`openai_radar/upload.py` posts `{api_base}/api/v1/estimate/radar/urls` with
`{"token", "role": "report"}`, then `PUT`s the CSV as `Content-Type: text/csv`
(the presigned URL signs that content type).

---

## openai-agents integration

```bash
pip install openai-radar[agents]
```

```python
from openai_radar.agents import build_radar_agent
from agents import Runner

agent = build_radar_agent()
result = Runner.run_sync(agent, "Scan my org and flag any cost anomalies")
print(result.final_output)
```

Or compose individual Radar tools into your own agent:

```python
from agents import Agent
from openai_radar.agents.tools import scan_assistants, run_findings, export_drawio

agent = Agent(
    name="My FinOps Agent",
    instructions="...",
    tools=[scan_assistants, run_findings, export_drawio],
)
```

---

## Findings engine

| Rule ID   | Severity | Condition                                      |
| --------- | -------- | ---------------------------------------------- |
| ASST_001  | LOW      | Assistant has zero tools                       |
| ASST_002  | INFO     | Code Interpreter enabled (file storage costs)  |
| VS_001    | MEDIUM   | Vector store > 5 GB                            |
| VS_002    | HIGH     | Vector store expires within 7 days             |
| FT_001    | MEDIUM   | Fine-tune job in `failed` state                |
| BATCH_001 | HIGH     | Batch job failure rate > 10%                   |
| USAGE_001 | MEDIUM   | Model consumes > 10M tokens in lookback window |

---

## Service relationship detection

`AssistantScanner` inspects each assistant's `instructions` field for external service signals
and emits `ServiceRelationship` objects (visualized as edges in the draw.io diagram):

| Kind     | Signals                                         |
| -------- | ----------------------------------------------- |
| AWS      | `aws`, `s3`, `ec2`, `lambda`, `dynamodb`, `sqs` |
| GCP      | `gcp`, `bigquery`, `gcs`, `google cloud`        |
| AZURE    | `azure`, `blob.core.windows`, `cosmosdb`        |
| DATABASE | `postgres`, `mysql`, `mongo`, `redis`, `neon`   |
| SLACK    | `slack`                                         |
| EMAIL    | `sendgrid`, `mailgun`, `smtp`                   |
| WEBHOOK  | `webhook`, `http://`                            |

---

## SDK structure

```
src/openai_radar/
├── client.py            # RadarClient (auth, project vs admin key)
├── runner.py            # Runner, RunConfig, RunResult
├── findings.py          # FindingEngine, Finding, Severity
├── models/base.py       # Pydantic v2 models for all resource types
├── scanners/            # One scanner per resource type
├── exporters/           # CSV + draw.io exporters
├── report.py            # organization cost report.csv
├── upload.py            # Pump presigned-URL upload (role: report)
├── agents/              # openai-agents tools + build_radar_agent()
└── cli.py               # openai-radar CLI
```

---

## Development

Python 3.10 or newer. From a checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[all]"
```

The package lives in `src/openai_radar`, including `cli.py` and `pump_login.py`.

From this checkout:

```bash
./openai-radar --help
./openai-radar login
./openai-radar status
./openai-radar logout
./openai-radar version
```

`login` opens a browser for the Pump OAuth flow and writes the token to
`$XDG_CONFIG_HOME/openai-radar/credentials.json`, or
`~/.config/openai-radar/credentials.json` when `XDG_CONFIG_HOME` is unset.
`OPENAI_RADAR_CONFIG_DIR` overrides that directory. `PUMP_API_BASE` and
`PUMP_APP_BASE` override the Pump origins, as do `--api-base` and `--app-base`.
Scans still read `OPENAI_API_KEY` and, for org-wide usage, `OPENAI_ADMIN_KEY`.

Login tests talk to a local fake Pump:

```bash
python -m unittest tests.test_pump_login
```

---

## Part of the Hyperscaler Radar suite

`aws-radar` · `gcp-radar` · `azure-radar` · `oci-radar` · `openai-radar` ·`claude-radar` · `gemini-radar` · `datadog-radar`
