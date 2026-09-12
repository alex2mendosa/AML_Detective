# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

**AML Detective** is an automated Anti-Money Laundering screening and adverse-media
research system for **Moldindconbank (MICB)**. For each contragent (counterparty
company) in the bank's daily monitoring set it produces a structured compliance
verdict; the deep-path final report is written in **Romanian** and conforms to
Section IV of `technical_docs/Model Nota de analiza AML MP_11.12.docx`.

Two pipelines run in series, communicating only through the filesystem:

1. **Fast screening** (`__test_opensanction_api_v8_prod.py`) — seconds per company.
   Takes a list of contragents (read from Oracle DWH by the caller), queries
   OpenSanctions `/match/default` → `/entities/{id}` (for beneficial-owner
   enumeration), runs an LLM verdict **only** when the entity matched as a
   sanctions target. Writes one JSON per contragent to `output_sanctions/{IDNO}.json`.

2. **Deep research** (`4_11_2_research_assistant.py`) — minutes per company.
   Reads the fast-screening JSONs as its batch input, runs a LangGraph agent over
   Google (SerpAPI) + Tavily with HyDe semantic filtering and multi-language
   adverse-media search, consolidates evidence claims, and produces the final
   Romanian report. Sanctions data is fed into the final LLM prompt as authoritative
   pre-known evidence so the deep agent's `FinalReport` already reflects sanctions
   precedence. Writes one JSON per contragent to `output_ad_media/{sname}.json`.

Both pipelines are **Airflow-callable**: each exposes a plain function that takes
secrets as arguments (`run_screening` / `run_osint_agent`), so a `PythonOperator`
can import the module and call it without relying on `.env`.

| Source | Purpose |
|---|---|
| `__test_opensanction_api_v8_prod.py` / `.ipynb` | **Pipeline 1 — fast screening (current)** |
| `4_11_2_research_assistant.py` / `.ipynb` | **Pipeline 2 — deep research (current)** |
| `agent_components/` | Shared library: `Config`, states, prompts, LLM config, query builder, keyword dictionary, HyDe data, logger factory |

Older versions (`__test_opensanction_api_v7_prod.*`, `4_11_1_research_assistant.*`,
`4_10_*`, the `src/` tree, `claude_check/`, deleted-but-untracked files in
`git status`) are **out of scope** unless explicitly asked.

The user converts `.ipynb` → `.py` via `jupyter nbconvert`. The notebook is the
source of truth for pipeline 2; pipeline 1 is maintained as both (see
**Known Issues** — the two have drifted on import style).

---

## External dependencies — providers, APIs and data

### Third-party APIs

| # | Provider / API | Endpoints used | What it is used for | Auth | Pipeline |
|---|---|---|---|---|---|
| 1 | **OpenSanctions** `api.opensanctions.org` | `POST /match/default`, `GET /entities/{id}`, `GET /healthz` | Sanctions/PEP/watchlist screening of the company by registration number; the second call expands the ownership/directorship graph for beneficial-owner screening | `Authorization: ApiKey <key>` — `OPENSANC_API_KEY` | P1 (live). P2 does not call it — it consumes P1's JSON files |
| 2 | **OpenAI** `api.openai.com` | Chat Completions via `langchain-openai`; Embeddings; `models.list` (preflight) | P1: the single AML verdict (`gpt-4o`). P2: search-payload generation, name variations, per-URL summarisation, evidence consolidation, reflection, final Romanian report, plus embeddings for HyDe similarity | `Bearer` — `OPENAI_API_KEY` | Both |
| 3 | **Tavily** `api.tavily.com` | `/search`, `/extract` | Adverse-media search (AI-optimised, advanced depth) and full page-content extraction in batches of 20 URLs | `api_key` arg — `TAVILY_API_KEY` | P2 |
| 4 | **SerpAPI** `serpapi.com` | `/search` (`engine=google`) | Google adverse-media search with `hl` / `lr` language scoping. Used **instead of** the native Google API — `GOOGLE_API_KEY` is not used anywhere | `api_key` query param — `SERP_GOOGLE_API_KEY` | P2 |

### Models in use

| Model | Where |
|---|---|
| `gpt-4o` | P1 `AMLAssessment` verdict (`Config.AML_MODEL`) |
| `gpt-4.1` | P2 URL summarisation, evidence aggregation, lead researcher, final report |
| `gpt-4.1-mini` | P2 search-payload generation, name variations |
| `text-embedding-3-large` | P2 HyDe cosine-similarity filter (`embedding_cross_lang`) |

### Data sources

| Data | Where it lives | Used by |
|---|---|---|
| **Contragent list** — `ARCDATE, ID, IDENTIFYCODE, SNAME` from `DM_NM.AR_NOTA_MONITORIZARE_PJ_DAILY_CONTRAGENT_INFO` on Oracle DWH `db1prodDWH.cs.mcb.md:1521 / PWH`, login schema `MDWH` | Oracle, read via SQLAlchemy + `oracledb` (thin mode, no Instant Client) | P1 `__main__` / the DAG. **Not** read inside `run_screening` — the caller passes the rows in |
| **Sanctions program map** (id → title) | `agent_components/sanc_prog_dict.json` | P1 `extract_for_llm` via `Config.sanctions_programs_map()` |
| **Dataset map** (id → title) | `agent_components/opensanctions-sources-2026-04-26.csv` | P1 `extract_for_llm` via `Config.datasets_description_map()`. **Refresh periodically** — unmapped ids fall back to `"Unknown Dataset"` |
| **Risk topic descriptions** (FollowTheMoney topics) | `Config.RISK_TOPICS` in `agent_components/config.py` | P1 — turns slugs like `sanction.linked` into prose for the LLM |
| **HyDe reference articles** (~100 per topic, carrying a `<\|company_name\|>` placeholder) | `agent_components/agents_hyde_articles.json` | P2 HyDe subgraph via `Config.hyde_articles()` |
| **Keyword dictionary** (~1.6k terms, financial / corruption / organized_crime × en/ro/ru) | `agent_components/dictionary.py` | P2 `filter_key_terms_node` |
| **Domain blocklists** | `Config.EXCLUDE_DOMAINS` (sent to Tavily), `Config.LOW_VALUE_DOMAINS` (post-search drop), `DOMAIN_EXCLUDE` in `states_v2.py` (Google `-site:` modifiers) | P2 |
| **Web pages** — public news, registries, court records in EN/RO/RU | Fetched live via SerpAPI + Tavily | P2 |
| **Screening results** | `output_sanctions/{IDNO}.json` | Written by P1, read by P2 |
| **Report template** | `technical_docs/Model Nota de analiza AML MP_11.12.docx` (Section IV) | Defines `FinalReport` field names and order — **re-read the .docx before changing the schema; do not infer from code** |

---

## Required Environment Variables

Both pipelines call `load_dotenv(Config.PROJECT_DIR / ".env", override=True)` at
import. This is a **local/CLI convenience only** — under Airflow the secrets are
passed as function arguments and `.env` is absent (a harmless no-op).

```
# OpenAI (both pipelines)
OPENAI_API_KEY

# Adverse-media search (deep pipeline only)
TAVILY_API_KEY
SERP_GOOGLE_API_KEY

# Sanctions screening (fast pipeline only)
OPENSANC_API_KEY

# Oracle DWH (fast pipeline only — source of contragents, read in __main__/DAG)
ORACLE_SQL_USERNAME
ORACLE_SQL_PASSWORD
ORACLE_SQL_CONNECTION_STRING       # host:port
ORACLE_SQL_SERVICE_NAME
```

`GOOGLE_API_KEY` is **not** used — we use SerpAPI for Google search.

Inside P2 the two search keys are funnelled through `APIVault`
(`agent_components/utils.py`) because the tool functions read them from a module
global (`key_vault`) rather than from arguments.

---

## `agent_components/config.py` — the `Config` class

Introduced with v8 / 4_11_2. Single source of truth for paths, endpoints and
static reference data, shared by both pipelines. Prefer adding constants here over
re-declaring them in a notebook.

```python
Config.PROJECT_DIR      # repo root = agent_components/../
Config.COMPONENTS_DIR   # agent_components/
Config.OUTPUT_DIR       # output_sanctions/   (P1 output)
Config.SANCTIONS_DIR    # output_sanctions/   (P2 input — same dir, named for the reader)
Config.AD_MEDIA_DIR     # output_ad_media/    (P2 output)
Config.LOG_PATH         # sanctions_screening.log
Config.RESEARCH_LOG     # aml_research.log

Config.SOURCE_TABLE     # DM_NM.AR_NOTA_MONITORIZARE_PJ_DAILY_CONTRAGENT_INFO
Config.OS_MATCH_URL     # https://api.opensanctions.org/match/default
Config.OS_BASE_URL      # https://api.opensanctions.org
Config.MATCH_PARAMS     # algorithm=logic-v2, limit=3, threshold=0.7, changed_since=2015-01-01
Config.AML_MODEL        # gpt-4o

Config.RISK_TOPICS        # FollowTheMoney topic → description
Config.LOW_VALUE_DOMAINS  # set of full URLs dropped after search
Config.EXCLUDE_DOMAINS    # bare domains passed to Tavily exclude_domains

Config.sanctions_programs_map()   # cached JSON load
Config.datasets_description_map() # cached CSV load
Config.hyde_articles()            # cached JSON load (raw, placeholder not substituted)
```

All three loaders are lazy and cached on class attributes, so importing `Config`
is cheap and each file is read at most once per process.

---

## Pipeline 1 — Fast Screening (`__test_opensanction_api_v8_prod.py`)

Pure Python, no LangGraph.

### Entry points

```python
# Airflow / programmatic — the real entry point
run_screening(
    contragents      = [{"IDENTIFYCODE": "...", "SNAME": "...", "ID": 1}, ...],
    openai_api_key   = "...",
    opensanc_api_key = "...",
    jurisdiction     = "md",
    output_dir       = None,      # defaults to Config.OUTPUT_DIR
    clean_output     = True,      # wipe *.json before the run
)  # → {"screened": n, "output_dir": "..."}

# Lower-level, per company
raw     = screen_company("1021600048015", "md", opensanc_api_key)
chain   = build_aml_chain(openai_api_key)     # build ONCE per run, reuse
verdict = assess_aml_risk(raw, chain)
```

`run_screening` validates both secrets **once** before the loop, builds the AML
chain once, and does **not** touch Oracle — the caller supplies the rows. The
`__main__` block is what reads Oracle (and falls back to a hardcoded 6-IDNO sample
if the DB is unreachable).

### Flow inside `screen_company`

1. **Input validation** — rejects empty `registration_number`/`jurisdiction`
   without hitting the API; returns a `validation_error` shape.
2. **`match_company_by_registration`** — `POST Config.OS_MATCH_URL` with
   `Config.MATCH_PARAMS`, body `schema=Company` + `registrationNumber` +
   `jurisdiction`. Returns one of three shapes:
   - successful match (`{"results": [...], ...}`),
   - genuine no-match (`{"results": [], "total": {"value": 0}}`),
   - API failure sentinel (`{"_api_error": "..."}`).

   The sentinel is load-bearing — it prevents transient API failures from
   collapsing into `is_target=False` → silent false-negative AML verdicts.
3. **`extract_for_llm`** — flattens the top match into a dict: `entity_id`,
   `entity_name`, `match_score`, `is_target`, `jurisdiction`, `risks` (topics
   mapped through `Config.RISK_TOPICS`), `other_topics` (unmapped, preserved as an
   audit trail), `sanctions_programs`, `datasets`, `notes`, `last_seen`.
4. **`get_entity`** — second call to `/entities/{id}` so we can walk
   relationships (the `/match` payload doesn't expand them). Returns `None` on
   failure — a *partial* failure; the primary entity verdict is still valid.
5. **`extract_people_from_entity`** — walks `directorshipOrganization` +
   `ownershipAsset`, dedupes by id, marks `status="former"` when `endDate` exists,
   flags `is_target`. Guards against `None` entity_data.
6. **`entity_fetch_error`** flag — set when `/entities` failed despite a valid
   `entity_id`, so `assess_aml_risk` refuses to clean-verdict on incomplete BO data.

Occupation inference was removed in v7 (enrichment only, no compliance impact).

### HTTP retry (`_request_with_retry`)

Retries on network errors (`ConnectionError`, `Timeout`), `429` (honours the
server's `Retry-After`), and `5xx`. Fails fast on all other `4xx`.
`MAX_RETRIES=5`, exponential backoff from `RETRY_DELAY=5s`.

### Flow inside `assess_aml_risk(screening_result, aml_chain)` — branch order is load-bearing

1. `validation_error` → `ERROR / safe_to_engage=False`
2. `api_error` → `ERROR / safe_to_engage=False` (must precede the `is_target`
   check — `is_target` is `None` on API failure and `None` is falsy)
3. `is_target is False` → `GREEN / safe_to_engage=True`, **no LLM call**
4. `entity_fetch_error` → `ERROR / safe_to_engage=False` (BO graph incomplete)
5. Defensive guard: `is_target=True` but `entity_id` missing → `ERROR`
6. `is_target=True` → `aml_chain.invoke(...)` → `AMLAssessment(risk_level, reason,
   sanction_conclusion, safe_to_engage)`. RED = direct prohibition; YELLOW = EDD
   required; GREEN = standard DD. An LLM failure surfaces as `ERROR`, not a crash.

### Outputs

- `output_sanctions/{IDNO}.json` per company — directory wiped at run start when
  `clean_output=True`.
- Every output carries `sname` (DB-authoritative, always present even when the API
  failed) and `registration_number` (belt-and-braces), injected by `run_screening`.
- Log via `agent_components/logger.get_logger()` → `sanctions_screening.log` (append).

### Known limitation

No cross-entity graph traversal — the API doesn't connect e.g. Lukoil RU to
Lukoil MD. Don't assume parent/subsidiary lookups work.

---

## Pipeline 2 — Deep Research (`4_11_2_research_assistant.py`)

Built on **LangGraph** (`StateGraph(UnifiedResearchState)`).

### Entry points

```python
# Airflow / programmatic — the sync wrapper a PythonOperator calls
run_osint_agent(openai_api_key, tavily_api_key, serp_api_key, NUM_RESULTS_PER_QUERY=10)

# the async body
await main(openai_api_key, tavily_api_key, serp_api_key, NUM_RESULTS_PER_QUERY=10)
```

CLI (`python 4_11_2_research_assistant.py`) calls `run_osint_agent` with keys from
the environment and `NUM_RESULTS_PER_QUERY=5`.

### What happens at import vs. inside `main()`

At **import**: constants, `Models` / tool definitions, subgraph + graph compile,
and a handler-less `logger`. Nothing else — importing the module is safe.

Inside **`main()`**, in order:

1. `setup_logging()` — attaches file + console handlers (`aml_research.log`, `mode='w'`).
2. Builds `APIVault` and registers `serp_google_key` / `tavily_key`.
3. Builds `Models(openai_api_key)` and assigns the **module globals the graph nodes
   read** (`key_vault`, `llm_with_tools`, `embedding_cross_lang`,
   `llm_url_content_summary`, `llm_agg_summaries`, `llm_evaluation`,
   `llm_lead_researcher`). These are `global` statements — the nodes resolve them
   at call time. Any refactor that removes them breaks every node.
4. `preflight_check_api_keys(openai, tavily)` — raises if either is down.
5. `load_sanctions_by_idno(Config.SANCTIONS_DIR)` → builds `test_companies`;
   raises `RuntimeError` if `output_sanctions/` yields nothing usable.
6. Wipes and recreates `Config.AD_MEDIA_DIR`.
7. Sequential batch loop; each company is wrapped in try/except so one failure does
   not abort the batch.

### Model wiring — `class Models`

P2 builds its own LLMs inline (`Models.__init__`), one `ChatOpenAI` per role plus
`OpenAIEmbeddings`. `agent_components/llm_config.py` (`build_llms`) holds an older,
near-equivalent factory that **nothing currently imports** — see Known Issues.

### Pipeline tuning constants (module level, read by nodes as globals)

```python
MAX_JOURNALISTS          = 10     # vestigial — personas were removed
RECURSION_LIMIT          = 50     # LangGraph safety cap
HYDE_RELEVANCE_THRESHOLD = 0.45   # cosine cutoff in filter_semantic_similarity_node
HYDE_TOP_K               = 30     # max pages kept after the threshold
CUSTOM_MAX_QUERIES       = 5      # cap on queries the lead researcher fires per loop
CUSTOM_RESULTS_PER_QUERY = 5      # Tavily max_results for each custom query
scenario_pool = list(get_args(ScenarioPool))
```

`NUM_RESULTS_PER_QUERY` is now a **parameter of `main()`** (default 10), passed
into state as `num_results_alias` — no longer a module constant.

### Batch input (built inside `main`)

There is **no** hardcoded company list — the batch comes from `output_sanctions/*.json`:

```python
sanctions_by_idno = load_sanctions_by_idno(Config.SANCTIONS_DIR)
test_companies = [(d["sname"], d["registration_number"])
                  for d in sanctions_by_idno.values()
                  if d.get("sname") and d.get("registration_number")]
```

For each `(company, idno)`:

```python
parts, name_re = generate_name_regex(company, models.llm_names_variation)
qc = QueryComponentsCalc(company, DOMAIN_EXCLUDE, parts)
qc.build_all()
batch_input = qc.to_state().model_copy(update={"names_search_regexp": name_re.pattern})

result = await graph.ainvoke({
    "messages": [],
    "max_journalists": MAX_JOURNALISTS,
    "journalists": [],
    "hyde_list": [],
    "num_results_alias": NUM_RESULTS_PER_QUERY,
    "query_components": batch_input,
    "registration_number": idno,
    "tool_pool": tool_list_deployed,
    "sanctions_data": sanctions_by_idno.get(idno),   # ← merged into the final prompt
    "scenario_pool": scenario_pool,
    "scenario_used": []
}, {"recursion_limit": RECURSION_LIMIT})
```

### Node graph (execution order)

1. `scenario_selection_node` — picks the next scenario from `scenario_pool` minus
   `scenario_used`. `ScenarioPool = {tool_google_search, tool_tavily_search,
   custom_scenario, exit_scenario}`. Returns `exit_scenario` when exhausted.
2. `route_trigger_search` (conditional) → `search_payload_generate_node`,
   `generate_custom_payload_execute_node`, or `generate_risk_assessment_node`.
3. `search_payload_generate_node` — `llm_with_tools` (gpt-4.1-mini) emits tool
   calls for Google/Tavily. `APITimeoutError` retried (MAX_RETRIES=3, RETRY_DELAY=10s).
4. `tool_execute_search_node` — executes tool calls, dedupes URLs, drops
   `Config.LOW_VALUE_DOMAINS`, appends `LinkCollection`s to `search_results_raw`,
   records `QueryPerformance` rows.
5. `generate_custom_payload_execute_node` — the `custom_scenario` branch.
   `llm_lead_researcher` (gpt-4.1, bound to the `ConductSearch` tool) inspects
   current evidence + per-query performance and generates targeted Tavily queries,
   capped by `CUSTOM_MAX_QUERIES` / `CUSTOM_RESULTS_PER_QUERY`.
6. `extract_content_node` — `tavily_content_extractor` in batches of 20, depth
   `advanced`, format `markdown`. Applies the entity-name regex post-extract to
   drop pages that never mention the target.
7. `filter_key_terms_node` — regex patterns built from `dictionary.topic_keywords`.
8. `buffer_node_hyde_generation` → `route_expert_generation` — `call_subgraph`
   when `hyde_list` is empty, else straight to semantic filtering.
9. `call_subgraph` — 2-node subgraph. `create_expert_node` is now a **no-op**
   (personas removed); `generate_hyde_document_node` loads
   `Config.hyde_articles()` and substitutes the entity name into the
   `<|company_name|>` placeholder. **No LLM call.**
10. `filter_semantic_similarity_node` — `text-embedding-3-large` cosine similarity
    between HyDe articles and extracted content. Drops below
    `HYDE_RELEVANCE_THRESHOLD`, keeps top `HYDE_TOP_K`.
11. `generate_url_summary_node` (async, batches of 10) — `llm_url_content_summary`
    (gpt-4.1) emits a `ContentSummary` per URL: `claim_type`, `severity_level`
    (Level_1–5), `date_published`, `summary`. Token-escalation retry
    (3500 → 4500 → 5000). Never raises — returns a placeholder on failure.
12. `extract_evidence_claims_node` — `llm_agg_summaries` (gpt-4.1) consolidates
    per-URL summaries into a deduped `EvidenceClaim` list, preserving
    `supporting_urls` provenance. Token-escalation retry.
13. `url_verification_node` → `route_evidence_validation` — checks every
    `supporting_url` exists in the actual URL pool; loops back to
    `extract_evidence_claims_node` with corrective feedback until clean.
14. `reflect_evidence_quality_node` — `llm_agg_summaries` framed as a cautious
    long-term customer; produces `AssessEvidenceQuality{evidence_quality, reasoning}`
    ∈ `{repeat_search, convinced}`.
15. `route_should_run_tool` — loops back to `scenario_selection_node` or proceeds.
16. `generate_risk_assessment_node` — `llm_evaluation` (gpt-4.1) returns the
    `FinalReport`. **Sanctions data is inlined into this prompt** as
    `OpenSanctions screening result:\n{sanctions_block}` before the public-source
    evidence block. `search_result_assets` is built deterministically here as
    `media_urls + sanctions_dataset_titles`. → END.

### State (`UnifiedResearchState` in `agent_components/states_v2.py`)

- `registration_number: str`, `sanctions_data: dict` — populated per row from `output_sanctions/`
- `messages: Annotated[List[AnyMessage], add_messages]`
- `search_results_raw`, `search_results_entity_filtered`: `Annotated[..., operator.add]` — append-only audit trail
- `search_results_kterms_filtered`, `search_results_sm_filter`: `Annotated[..., merge_links_by_url]` — **non-None-only** update by URL, never overwrites with `None`
- `search_query_performance: Annotated[List[QueryPerformance], merge_query_data_by_id]` — same semantics, keyed by `query_id`
- `scenario_used: Annotated[List[str], operator.add]`
- `scenario_selected: str` — no reducer; single-writer
- `final_conclusion: FinalReport`
- `search_result_assets: list[str]` — becomes `rezultate_cautare` in the written file

When manipulating state outside a node (tests, debugging), apply the reducer
manually — `Annotated` only kicks in on node returns.

### Final output — `FinalReport` (Romanian, in `states_v2.py`)

Section IV row labels and order — **always re-read
`technical_docs/Model Nota de analiza AML MP_11.12.docx` before changing the
schema; do not infer from code**:

`rezumat_analiza`, `scor_risc` (0–100), `analiza_suspiciuni`, `situatie_actuala`,
`traiectorie`, `recomandare_relatie_afaceri`, `concluzie_finala`.

`scor_risc` scale: 0–25 Low, 26–50 Medium, 51–75 High, 76–100 Critical.

**`rezultate_cautare` is NOT a `FinalReport` field.** It is injected into the
dumped dict in `main()` from `result["search_result_assets"]`, so the LLM can never
hallucinate a URL. The written JSON therefore has 8 keys where the model has 7.

### Sanctions → media precedence

Enforced inside `final_summary_prompt` (`agent_components/prompts_v2.py`),
**not in code**:

- `risk_level=RED` → `scor_risc ≥ 76`, `recomandare TERMINATE/SUSPEND`, regardless of media
- `risk_level=YELLOW` → `scor_risc ≥ 51`, EDD recommended
- `risk_level=UNDEFINED` / `ERROR` → `SUSPEND` (engagement must not proceed on an unscreened entity)
- `risk_level=GREEN` → defer to media findings
- The LLM is instructed to cite the OpenSanctions `reason` and
  `sanction_conclusion` / `aml_conclusion` verbatim when RED or YELLOW.
- The empty-evidence override (sanctions GREEN/absent AND media empty) fires inside the prompt.

### Output file shape (changed in 4_11_2)

The written file is a **DB-ready row wrapper**, not a bare `FinalReport`:

```json
{
  "contragentid": "1021600048015",
  "arcdate":      "2026-06-01",
  "runtime":      "2026-06-02 08:24:11",
  "aml_report":   { "...FinalReport...": "...", "rezultate_cautare": ["..."] },
  "opensanc_data":{ "...the whole Pipeline-1 JSON...": "..." }
}
```

`arcdate` is **yesterday** (`date.today() - timedelta(days=1)`), matching the DWH
archive convention; `runtime` is the local wall-clock time of the run.

Written atomically:

```python
tmp = out_path.with_suffix(".json.tmp")
with open(tmp, "w", ...) as f: json.dump(row, f, ...)
tmp.replace(out_path)        # atomic rename
```

to `output_ad_media/{safe_name}.json` where
`safe_name = company.replace(" ", "_").replace("/", "-")`.
Log → `aml_research.log` (`mode='w'`, truncated each run).

---

## Agent Components (`agent_components/`)

Always import from the **v2** files (`states_v2.py`, `prompts_v2.py`,
`query_components_v2.py`). Older `query_components.py` and any non-v2
prompts/states are deprecated.

| File | Contents |
|---|---|
| `config.py` | `Config` — paths for both pipelines, Oracle table, OpenSanctions URLs + match params, `AML_MODEL`, `RISK_TOPICS`, `LOW_VALUE_DOMAINS` / `EXCLUDE_DOMAINS`, and the three cached loaders. See its own section above. |
| `states_v2.py` | All Pydantic models + `UnifiedResearchState` TypedDict + reducers (`merge_links_by_url`, `merge_query_data_by_id`). Also exports `DOMAIN_EXCLUDE` (`facebook.com`, `wikipedia.org`, `wikimedia.org`) and `ScenarioPool`. |
| `prompts_v2.py` | `lead_researcher_prompt`, `expert_instructions`, `url_summary_instructions`, `system_messages_search_tools`, `extract_evidence_claims_prompt`, `final_summary_prompt` (contains the sanctions-precedence ladder and the Romanian field instructions). |
| `query_components_v2.py` | `QueryComponentsCalc(entity_name, DOMAIN_EXCLUDE, entity_names_variations)`. Call `build_all()` then `to_state()`. Builds Google modifier strings (`-site:`, OR'd quoted variants), Tavily query templates, and topic/language keyword sets across **6 topics × 3 languages** (`financial`, `corruption`, `organized_crime`, `sanctions`, `legal`, `reputational` × `en`/`ro`/`ru`). Requires `entity_names_variations` from `generate_name_regex` before `build_modifiers` works. |
| `dictionary.py` | `topic_keywords`: 3 top-level topics (`financial`, `corruption`, `organized_crime`) × 3 languages, ~1.6k terms. Used by `filter_key_terms_node`. **Distinct** from the 6-topic structure in `QueryComponentsCalc`. |
| `utils.py` | `APIVault` (store/retrieve API keys by name), `first_n_words`, `list_to_string`, `count_leaves`. Canonical (`__init__.py` duplicates this content). |
| `logger.py` | `get_logger()` factory used by Pipeline 1. Pipeline 2 builds its own logger in `setup_logging()`. |
| `llm_config.py` | `build_llms(key_vault)` — older LLM factory. **Currently imported by nothing**; P2 uses its inline `Models` class instead. |
| `sanc_prog_dict.json` | OpenSanctions program-id → title map. |
| `opensanctions-sources-2026-04-26.csv` | Dataset-id → title map. **Refresh periodically.** |
| `agents_personas.json` | 20 pre-baked journalist personas. **No longer loaded** by 4_11_2 (personas were removed); referenced only in stale header comments. |
| `agents_hyde_articles.json` | Pre-generated HyDe reference articles (~100 per topic), loaded via `Config.hyde_articles()`. |

### Core Pydantic models you'll touch

- `LinkCollection` — per-URL record. Required: `displayLink`, `link`. Optional
  everything else (`summary`, `claim_type`, `severity_level`, `hyde_score`,
  `raw_content`, …). The reducer expects optional fields to be `Optional[str]`,
  not `str` with default `None`.
- `ContentSummary` — single-URL summary output (LLM-structured).
- `EvidenceClaim` — `claim_text`, `claim_type` ∈ `{allegation, investigation,
  charge, conviction, settlement, sanction_listing, other}`, `supporting_urls`,
  `date_publish`.
- `QueryComponentsInState` — `entity_name`, `DOMAIN_EXCLUDE`, `search_queries`,
  `search_topics`, `entity_names_variations`, three `google_search_modifier_*`
  fields, optional `names_search_regexp`.
- `AssessEvidenceQuality` — `evidence_quality` ∈ `{repeat_search, convinced}` + `reasoning`.
- `FinalReport` — 7 Romanian fields, see above.
- `AMLAssessment` (Pipeline 1) — `risk_level` ∈ `{RED, YELLOW, GREEN}` + `reason`,
  `sanction_conclusion`, `safe_to_engage`.

---

## Key Design Patterns

- **Secrets as arguments** — both pipelines' entry points take keys as parameters
  so Airflow can supply them from Connections/Variables. `.env` loading is a local
  convenience, never a requirement.
- **`Config` over notebook constants** — paths, endpoints and reference data live
  in one class shared by both pipelines.
- **HyDe filtering** — pre-generated synthetic articles are embedded and compared
  to real search results via cosine similarity. A cheap noise filter before the
  expensive summarisation step. Fully offline since personas were dropped.
- **Multi-language by default** — every search runs in EN/RO/RU.
- **Append-or-merge, never overwrite** — `merge_links_by_url` /
  `merge_query_data_by_id` only write non-None values onto existing records.
- **Claim provenance** — every `EvidenceClaim` retains its `supporting_urls`, and
  `url_verification_node` enforces that those URLs exist in the search pool.
- **Fast-path short-circuit** — non-targets skip the LLM entirely in Pipeline 1.
- **API-failure ≠ no match** — `match_company_by_registration` returns an
  `_api_error` sentinel; `assess_aml_risk` maps it to `ERROR / safe_to_engage=False`.
  Never collapse API failure into GREEN.
- **Scenario loop** — `scenario_selection_node` walks `scenario_pool`
  deterministically; `exit_scenario` jumps straight to the final report.
- **Sanctions precedence in the prompt** — the risk-level → score-floor mapping
  lives inside `final_summary_prompt`, not in Python. Change the prompt to change
  the policy.
- **Deterministic `rezultate_cautare`** — built as `media_urls +
  sanctions_dataset_titles` and injected after `model_dump()`. The LLM never
  generates URLs.
- **Batch isolation** — one company's exception is logged and skipped; the batch continues.

## Notebook & deployment conventions

- Notebook filename: `{major}_{minor}_{patch}_research_assistant[_prod].ipynb`.
  **`4_11_2_research_assistant` is current.** Older numbered notebooks and other
  `__test_*` notebooks are out of scope unless asked.
- Before running `nbconvert`, ensure no cell contains `get_ipython()`,
  `!shell-command`, `%magic`, or `from IPython.display import …` — those crash a
  plain-Python interpreter. The current notebooks have them commented out.
- For a server run, the script's directory must contain `.env` (CLI only),
  `agent_components/`, `output_sanctions/`, `output_ad_media/`.

---

## Cross-pipeline contract

The two pipelines communicate **only through the filesystem**: pipeline 1 writes
`output_sanctions/{IDNO}.json`, pipeline 2 reads them via
`load_sanctions_by_idno(Config.SANCTIONS_DIR)`.

The join key is `registration_number` (IDNO), with the filename stem as fallback.
Every screening output also carries `sname` (DB-authoritative). Pipeline 2 builds
its batch list from those two fields, so `sanctions_by_idno.get(idno)` is
guaranteed present by construction.

When changing the screening output shape, the deep pipeline expects at minimum:

- `registration_number: str`
- `sname: str`
- `risk_level: "RED" | "YELLOW" | "GREEN" | "ERROR" | "UNDEFINED"`
- `reason: str`
- `sanction_conclusion: str` (or `aml_conclusion` — the prompt accepts either)
- `datasets: dict[str, str]` (id → title) — becomes part of `rezultate_cautare`
- `is_target: bool | None` (None = API failure)

The whole screening JSON is also echoed verbatim into the P2 output under
`opensanc_data`, so any field added upstream shows up downstream for free.

---

## Known Issues — Fix Later

The pipelines run correctly today; these are deferred cleanups unless marked otherwise.

### `__test_opensanction_api_v8_prod.py`

1. **The `.py` cannot be run as a plain script (real bug).** It uses relative
   imports — `from .agent_components.logger import get_logger`,
   `from .agent_components.config import Config` — so
   `python __test_opensanction_api_v8_prod.py` fails with
   `ImportError: attempted relative import with no known parent package`, even
   though the file still carries an `if __name__ == "__main__":` block. The
   notebook uses **absolute** imports (`from agent_components.config import Config`),
   so `.py` and `.ipynb` have drifted. Either run it as a package module
   (`python -m <pkg>.__test_opensanction_api_v8_prod`) or switch the `.py` back to
   absolute imports. Airflow calling `run_screening` via a package import is unaffected.
2. **DB-unavailable fallback uses a hardcoded 6-IDNO sample.** Useful for smoke
   tests, dangerous if it silently fires in prod — the log then shows a fake-looking
   6-row run. Consider gating it behind `AML_ALLOW_DB_FALLBACK=true`.
3. **`preflight_check` hardcodes `api.opensanctions.org/healthz`** and a bare
   `OpenAI()` rather than using `Config.OS_BASE_URL`. Only called from `__main__`.
4. **`datasets_description_map` reads a dated CSV** (`opensanctions-sources-2026-04-26.csv`).
   Unmapped ids fall back to `"Unknown Dataset"` (preserved, not dropped). Refresh periodically.

### `4_11_2_research_assistant.py`

1. **Node globals are set inside `main()`** (`llm_with_tools`,
   `embedding_cross_lang`, `key_vault`, …). Importing the module and calling
   `graph.ainvoke` without going through `main()` raises `NameError`. Intentional
   for the Airflow shape, but it means the graph is not independently testable.
2. **Dead imports**: `sqlalchemy` (`create_engine`, `Integer`, `DateTime`, oracle
   `NCLOB`) is imported but never used — left over from an aborted write-to-Oracle
   step. Also `sys`, `threading`, `subprocess`, `MemorySaver`, `operator`, and a
   duplicated `cosine_similarity` / `numpy` import block.
3. **Stale docstrings**: `tavily_search` / `tool_tavily_search` still refer to a
   `_HARDCODED_EXCLUDE_DOMAINS` constant that no longer exists — the code uses
   `Config.EXCLUDE_DOMAINS`. The `exclude_domains` parameter is still accepted and
   still ignored. The file header still lists `agents_personas.json` and names the
   upstream pipeline as `__test_opensanction_api_v7_prod.py` (now v8).
4. **`MAX_JOURNALISTS` is vestigial** — personas were removed and
   `create_expert_node` is a no-op, yet the value is still threaded into state as
   `max_journalists`.
5. **`call_subgraph` passes `search_topic=""`** as a dummy. Harmless while
   `create_expert_node` ignores it; must become the real scenario topic if that changes.
6. **`links_this_query` debug log shows 0** in `tool_execute_search_node` —
   `links_before` is captured after the inner loop instead of before. Debug-only impact.
7. **`generate_url_summary` unreachable safety fallback** references `e` outside
   any `except` block. Replace `str(e)[:100]` with a literal — `e` is deleted after
   the `except` scope in Python 3.
8. **`evidence_feedback.evidence_quality` accessed without a None guard** in
   `route_should_run_tool`. Acknowledged in the docstring; upstream is robust today.
9. **Log file `mode='w'`** clobbers per run by design. Switch to append +
   `RotatingFileHandler` if retention matters.
10. **Sequential batch loop** — fine for ≤20 entities; revisit `asyncio.gather` +
    `Semaphore(3)` if batch size grows.

### `agent_components/states_v2.py`

1. **`HydePerspectives` field has no description (real bug, still open)** — the
   description string sits in the **default-value slot** of `Field(...)`, not the
   `description=` kwarg: `Field("Comprehensive list of analysts…")`. Replace with
   `Field(default_factory=list, description="…")`. Low impact — the model is unused
   now that personas are gone.
2. **Reducers mutate `existing` in place** — `merge_links_by_url` and
   `merge_query_data_by_id` call `existing.append(...)` / `setattr(...)`. Safe today
   because the pipeline is strictly sequential (no parallel fan-out, no checkpoint
   replay). Switch to copy-then-merge the moment that changes.
3. **Channels without reducers** — `scenario_selected`, `scenario_pool`,
   `tool_pool`, `search_result_assets`, `journalists`, `hyde_list`,
   `evidence_claims`, `url_feedback`, `evidence_feedback`, `final_conclusion`,
   `query_counter`, `num_results_alias`, `max_journalists`, `registration_number`,
   `sanctions_data`. Each has exactly one writer today; concurrent writes would
   raise `InvalidUpdateError`.
4. **`search_results_raw` / `search_results_entity_filtered` use `operator.add`
   with no dedup** — the same URL from multiple queries duplicates. Downstream
   `merge_links_by_url` channels handle dedup. Acceptable as a raw audit trail.
5. **`None` cannot overwrite a value** in either custom reducer — intentional, but
   it means a state update can never clear a field.

### `agent_components/`

- **`llm_config.py` is dead code** — `build_llms` duplicates what `Models` does in
  4_11_2 and is imported by nothing. Delete it or switch P2 to use it; keeping both
  guarantees they drift.
- **`__init__.py` duplicates `utils.py`** verbatim.
- Docstring typo at `merge_query_data_by_id` (`"liqidnk"` → `"qid"`), and it is
  missing a return type annotation.
