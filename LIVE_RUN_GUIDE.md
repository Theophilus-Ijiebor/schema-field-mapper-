# Live Run Guide (Anthropic + Azure AI Foundry)

You're running this locally with your own keys, so nothing here ever leaves
your machine. This walks through lighting up both providers, sanity-checking
they're actually reachable before you trust any output, running both demo
tenants live, and comparing against the offline-fallback baseline that
shipped in `output/`.

## 0. Setup

```bash
cd schema_field_mapper
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and fill in:

```
ANTHROPIC_API_KEY=sk-ant-...

AZURE_AI_FOUNDRY_ENDPOINT=https://<your-resource>.openai.azure.com
AZURE_AI_FOUNDRY_API_KEY=...
AZURE_AI_FOUNDRY_DEPLOYMENT=<your chat model's deployment name>
AZURE_AI_FOUNDRY_API_VERSION=2024-10-21
```

`AZURE_AI_FOUNDRY_DEPLOYMENT` is the *deployment name* you gave the model in
the Foundry portal's Deployments tab, not the base model name (e.g. it might
be `gpt-4o-mini-mapper`, not `gpt-4o-mini`). If a call fails with something
mentioning `api-version`, the portal's deployment page will show the current
value to use -- Azure revises these more often than most vendors.

`.env` is already covered by the project's ignore patterns implicitly by
never being referenced from any committed file except `.env.example` -- just
don't zip/commit your real `.env` anywhere. Every entrypoint (`cli/main.py`,
`api/main.py`, `run_eval.py`, `build_golden.py`, `providers/selftest.py`)
calls `load_dotenv()` on import, so `.env` in your cwd is picked up
automatically; no manual `export` needed.

## 1. Confirm both providers are actually reachable

Credentials present isn't the same as credentials working -- this makes one
real minimal call to each configured backend:

```bash
python3 -m app.providers.selftest
```

Expect:

```json
[
  {"provider": "anthropic", "configured": true, "live": true, "detail": "responded correctly"},
  {"provider": "azure_ai_foundry", "configured": true, "live": true, "detail": "responded correctly"},
  {"provider": "openai_embeddings", "configured": false, "live": false, "detail": "no credentials found"}
]
```

`openai_embeddings` shows `configured: false` unless you also set
`OPENAI_API_KEY` or `AZURE_AI_FOUNDRY_EMBEDDING_DEPLOYMENT` -- that's fine,
retrieval just uses TF-IDF in that case (see README's "Retrieval backend"
section). If `azure_ai_foundry` shows `live: false` with credentials
present, it's almost always the API version or the deployment name -- check
the portal.

## 2. Re-seed tenants (picks up the provider pins)

```bash
python3 -m app.tenancy.registry
```

This prints each tenant's API keys once -- save them somewhere. `acme-hr` is
pinned to `anthropic`; `globex-hr` is pinned to `azure_ai_foundry` (see
`app/tenancy/registry.py::seed_demo_data`). Running the same schema pair
through both, via identical application code, is the actual proof the
`ModelProvider` abstraction works -- not just that one SDK call succeeds.

## 3. Run both tenants live

```bash
python3 -m app.cli.main --tenant acme-hr --auto-approve --output-dir output/live
python3 -m app.cli.main --tenant globex-hr --auto-approve --output-dir output/live
```

Watch the stderr JSON-lines log for `"event": "load_config"` -- it names the
resolved provider per run (`"provider": "anthropic"` / `"azure_ai_foundry"`).
Compare `reasoning` text in `output/live/*/mapping_*.json` against
`output/acme-hr/mapping_*.json` (the offline-fallback baseline already
committed) -- the destination fields chosen should mostly match (the
underlying task is the same, and both a competent LLM and the offline
heuristic converge on the same obviously-correct fields), but the wording
and confidence calibration will differ, and it's worth actually reading a
few to confirm the live reasoning is *specific* to the field pair rather
than generic.

`--auto-approve` skips the interactive HITL prompt for the demo; drop it to
see the live confidence scores actually route fields to human review the
way they're designed to.

## 4. Run the eval harness live (DeepEval GEval, not the heuristic fallback)

```bash
LATEST=$(ls -t output/live/acme-hr/mapping_*.json | head -1)
python3 -m app.evals.run_eval --input "$LATEST" --report-out output/eval_report_live.md
```

(`ls -t | head -1` rather than the glob directly -- if you've run the CLI
more than once, `output/live/acme-hr/` will have more than one
`mapping_*.json` and the glob would expand to multiple filenames, which
`--input` rejects. Grab the run_id the CLI printed and point at that file
directly if you want a specific run instead of the latest.)

The report header will say `Quality-metric engine: **deepeval (GEval)**`
instead of `heuristic (...)` -- that's the signal it actually used a live
judge model. Read a few `relevance_explanation` / `faithfulness_explanation`
rows; they should reference the actual field names/types, not generic
boilerplate, since DeepEval's `GEval` derives explicit evaluation steps from
the criteria before scoring (see README's "DeepEval GEval" section for how
this differs from a single-shot judge prompt).

## 5. What to actually compare, and what not to read too much into

Worth comparing: whether `reasoning` sentences are more specific/insightful
live vs. offline; whether any field's `confidence` crosses the 0.85/0.92
review threshold differently live (a live model might be more or less
confident than the retrieval-score-derived offline heuristic); whether
DeepEval's relevance/faithfulness scores agree with the heuristic's on the
same field mappings.

Not worth reading into: small differences in `destination_field` choice on
already-borderline fields (e.g. if a live run picks a different plausible
candidate for one HITL-gated field than the offline fallback did) -- both
`dept_id`-style ambiguous cases and the deliberately-unmapped `dob` field
should still resolve the same way, since those are governed by structural
rules (primary-key routing, the content-overlap gate) that don't change
based on which provider answered.

## 6. Afterward

Rotate or revoke whichever key(s) you used for this test if they're not
otherwise in routine use -- standard hygiene for any key that's been active
outside your normal application's runtime. Don't commit your real `.env`;
only `.env.example` (blank template) is meant to ship.
