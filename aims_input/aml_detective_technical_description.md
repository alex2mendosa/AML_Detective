# AML Screen Agent) — Technical Description

**Version 0.1 — draft for architecture review**  
**Date:** 10 September 2026  
The code described in this note is not committed to version control.

Build compliance agents that continuously retrieve adverse media, sanctions exposure, and public risk signals, helping investigators resolve cases faster with a complete audit trail.

- Retrieve adverse media and entity intelligence automatically
- Monitor vendors beyond onboarding
- Produce source-backed findings for audit and regulatory review

Two annexes are supplied as separate image files:

- **Annex 1 — Component map**, referenced from section 3.
- **Annex 2 — Agent graph**, referenced from section 5. It is generated from the graph definition in the code, not drawn by hand; its Mermaid source accompanies it.

------------------------------------------------------------------------

## 1. Purpose

AML, KYC Agent screens Moldindconbank's corporate counterparties against sanctions and watchlist data and researches public adverse media about them. Its work list is the bank's daily monitoring set of legal-entity counterparties; it supports the bank's monitoring note for those entities.

For each counterparty it produces two artefacts:

- a **sanctions screening verdict** — risk level RED, YELLOW or GREEN, a reason and a compliance conclusion;
- an **analysis note in Romanian**, following Section IV of the bank's template *Model Nota de analiza AML MP*: executive summary, risk score (0–100), analysis of suspicions, current situation, risk trajectory, recommendation, sources consulted and final conclusion.

The output supports the decision whether the bank may enter into or continue a business relationship. The recommendation is one of: terminate, suspend, enhanced due diligence, continue with monitoring, continue.

**How it works, end to end.** Two programs run in sequence; the second consumes the output of the first.

1.  The list of counterparties is read from the bank's data warehouse.
2.  The **screening pipeline** asks OpenSanctions whether each company is on a sanctions or watchlist, and who its directors and owners are. If the company is a sanctions target, a language model classifies it as RED, YELLOW or GREEN; otherwise the verdict is GREEN, with no model involved.
3.  The screening result is stored and becomes the second program's input.
4.  The **research agent** searches open sources in English, Romanian and Russian, retrieves pages, discards irrelevant ones, summarises the rest and consolidates the findings into evidence claims tied to their sources.
5.  A final model call combines the sanctions verdict with the media evidence and writes the Romanian note, including the risk score and recommendation.
6.  The note and the verdict are written to the database.

There are two results: the **screening verdict**, mostly decided by rules in code, and the **analysis note**, written by a model, which incorporates the verdict.

**Terms used in this note.** A *counterparty* is a company the bank does, or is considering doing, business with. *Adverse media* is negative coverage of that company in news and public records. A *sanctions target* is an entity directly designated on a sanctions list, as opposed to one merely linked to a designated party.

## 2. Technology classification

**Family.** **Generative AI**, deployed as an **agentic system**. General-purpose foundation models are called over an API; no model is trained, fine-tuned or hosted by the bank. This is **not supervised or unsupervised machine learning**: there is no training set, labelled data, fitted parameters or clustering. Classification is **zero-shot with constrained output** — a general model is prompted and its answer forced into a fixed schema, rather than a trained classifier being applied.

**Agent type.** The research half is a **goal-based agent with internal state**: it perceives search results and page content, keeps state, chooses actions from a defined tool set and works towards a goal. It is not a simple reflex agent — it reasons over what it has already found; not utility-based — it optimises no scored objective; and not a learning agent — nothing carries from one run to the next. The screening half is a **rule-based procedure** — fixed if/then rules written by a developer decide five of the six possible outcomes, and a model is consulted only in the sixth.

**Further characteristics.** Retrieval is **live web search, not retrieval-augmented generation over a bank corpus** — there is no vector database or index of bank documents; embeddings serve only as a similarity filter, and nothing is retained between runs. Part of the decision policy is **expressed in natural language**: the rule mapping a sanctions verdict to a risk-score band lives in the prompt, not in code. The system also uses **synthetic data as input**: its relevance filter compares retrieved pages with reference articles generated once by a language model (sections 5 and 7).

**Level of automation.** **Level 5 — full automation**; **heteronomous**, not autonomous.

*Table 1 — Relationship between autonomy, heteronomy and automation*

| Class | Level of automation | Description |
|----|----|----|
| Autonomous | 6 — Autonomy | The system is capable of modifying its intended domain of use or its goals without external intervention, control or oversight. |
| Heteronomous | **5 — Full automation** | **The system is capable of performing its entire mission without external intervention.** |
| Heteronomous | 4 — High automation | The system performs parts of its mission without external intervention. |
| Heteronomous | 3 — Conditional automation | Sustained and specific performance by a system, with an external agent being ready to take over when necessary. |
| Heteronomous | 2 — Partial automation | Some sub-functions of the system are fully automated while the system remains under the control of an external agent. |
| Heteronomous | 1 — Assistance | The system assists an operator. |
| Heteronomous | 0 — No automation | The operator fully controls the system. |

Once started, the system runs the whole sequence — work list, screening, search, filtering, summarisation, consolidation, note — with no step waiting for a person. It is not at level 6 because it cannot change its goal or domain of use: the rounds, report schema and decision rules are fixed outside the model. Its mission ends at a stored record; the business decision lies outside the system.

## 3. Composition

| Component | What it does | Where it comes from | Who operates it |
|----|----|----|----|
| Screening pipeline | Queries sanctions data per counterparty, lists its directors and owners, classifies the result | Built in-house | The Bank |
| Research agent | Runs the adverse-media investigation and writes the note | Built in-house | The Bank |
| Shared library | Prompts, report schema, agent state, reference maps, synthetic articles, keyword lists | Built in-house | The Bank |
| Screening result store | Files holding one verdict per counterparty; the interface between the pipelines | Built in-house | The Bank |
| Final report store | Files holding one finished note per counterparty | Built in-house | The Bank |
| Log files | One log per pipeline (section 10) | Built in-house | The Bank |
| Bank database | Receives the finished records (as described by the system owner) | Another Bank system | The Bank |
| Oracle data warehouse | Supplies the counterparty list | Another Bank system | The Bank |
| OpenSanctions API | Sanctions, watchlist and ownership data | Third-party service | OpenSanctions |
| OpenAI models `gpt-4o`, `gpt-4.1`, `gpt-4.1-mini` | All generative and classification steps | Third-party service | OpenAI |
| OpenAI embedding model `text-embedding-3-large` | Scores retrieved pages for relevance | Third-party service | OpenAI |
| Tavily API | Web search and page-text extraction | Third-party service | Tavily |
| SerpAPI | Google web search | Third-party service | SerpAPI |

*See Annex 1 — Component map.*

## 4. Tooling and libraries

**Language**

- **Python 3.10 or later** — the whole system: two executable modules and a shared library.

**Agent and model libraries**

- **LangGraph** — builds and runs the research agent as a state graph, enforcing the round sequence and step limit.
- **LangChain** (`langchain-openai`, `langchain-core`, `langchain-text-splitters`) — model calls, structured output, tool definitions, and text chunking before embedding.
- **OpenAI Python SDK** — the start-up connectivity check.
- **Pydantic** — defines every structured output a model must return, making answers machine-checkable rather than free text.

**Retrieval and search clients**

- `tavily-python` — Tavily web search and page-text extraction.
- `google-search-results` — SerpAPI client for Google search.
- `requests` — HTTP calls to OpenSanctions, with in-house retry and rate-limit handling.

**Data and numerical libraries**

- **SQLAlchemy** with `oracledb` in thin mode — reads the warehouse without installing an Oracle client.
- **pandas** — holds the counterparty list during the batch.
- **scikit-learn** and **NumPy** — cosine similarity for the relevance filter.

**Supporting**

- `python-dotenv` — loads credentials for local runs; scheduled runs pass them as arguments.

No library version is pinned, and the repository has no dependency manifest.

## 5. Approach, algorithm and model

**The models.** All run on OpenAI infrastructure and are called as a service: `gpt-4o` for screening classification; `gpt-4.1-mini` for search queries and company-name variants; `gpt-4.1` for page summaries, evidence consolidation, the sufficiency check and the final note; `text-embedding-3-large` for relevance scoring.

**The screening pipeline** is deterministic except for one model call. It sends the registration number and jurisdiction to the OpenSanctions match endpoint, then fetches the matched entity record to list directors and owners and flag any who are sanctioned. Supplier codes for topics, sanctions programmes and datasets are translated into readable titles through reference maps; unmapped codes are kept.

The verdict follows a fixed order of checks in code. The order matters — each check assumes the earlier ones did not fire — and only the last uses a model.

| Order | Condition | Verdict |
|----|----|----|
| 1 | Input incomplete; no query made | Error — engagement not permitted |
| 2 | Sanctions API call failed | Error — engagement not permitted |
| 3 | No match, or a match not flagged as a sanctions target | GREEN — no model call |
| 4 | Match, but the ownership graph could not be retrieved | Error — engagement not permitted |
| 5 | Result internally inconsistent | Error — engagement not permitted |
| 6 | Matched as a sanctions target | `gpt-4o` returns RED, YELLOW or GREEN with a reason, a conclusion and whether engagement may proceed |

A failed API call is never treated as a clean result.

**The research agent** is a state graph with a fixed sequence of rounds: Google, Tavily, then free-form. After each round a model judges whether the evidence suffices; if so, or when the rounds run out, the agent writes the note. In the Google and Tavily rounds the model issues queries pre-built by code, setting language and result count; in the free-form round it writes its own, using the counterparty's name and generated variations of the possible crimes reported in relevant news — free-form AML topics based on the results of the previous rounds. Steps are bounded by the fixed rounds, a graph recursion limit and caps on free-form queries and results. The tools are described in section 6.

Retrieved pages pass four filters before the costly steps: a domain block list, a company-name filter, a multilingual keyword filter and a semantic filter. The semantic filter uses Hypothetical Document Embeddings: the counterparty's name is inserted into pre-written synthetic articles describing the adverse coverage sought; these and the retrieved pages are embedded, and the pages with the highest cosine similarity are kept. The articles — 100 across five topics, generated once by `gpt-4.1-mini` — are static synthetic input; no model generates them at run time.

**Failures.** If OpenAI or Tavily is unreachable at start, the run stops. Within a counterparty's run, a failed Google search contributes no links, a failed extraction batch leaves pages empty so they are filtered out, and a failed page summary is replaced by an error placeholder. A failed Tavily search, embedding, consolidation, sufficiency or note call, or hitting the recursion limit, ends that counterparty's run: the failure is logged, no record is written, and the batch continues.

*See Annex 2 — Agent graph, generated from the compiled graph.*

**Agent flow overview.** The fourteen steps, in graph order:

| Step | What it does |
|----|----|
| `scenario_selection_node` | Picks the next unused round from the fixed pool, or signals exit. No model call. |
| `search_payload_generate_node` | A model composes the round's search queries: wording, language and result count. |
| `tool_execute_search_node` | Runs the searches, removes duplicates and blocked domains, and records each query's yield. No model call. |
| `generate_custom_payload_execute_node` | Free-form round: a lead-researcher model reviews the evidence and query yield, then composes and runs targeted searches: the counterparty's name with generated variations of the possible crimes found in relevant news. |
| `extract_content_node` | Retrieves page text and drops pages that never mention the counterparty. No model call. |
| `filter_key_terms_node` | Keeps pages with financial-crime, corruption or organised-crime vocabulary in English, Romanian or Russian. No model call. |
| `buffer_node_hyde_generation` | Decides whether the synthetic articles still need loading. No model call. |
| `call_subgraph` | Loads the synthetic articles and inserts the counterparty name. No model call. |
| `filter_semantic_similarity_node` | Embeds pages and articles; keeps the highest-scoring pages above the cut-off. Embedding calls only. |
| `generate_url_summary_node` | A model summarises each page: claim type, severity level, publication date and summary. |
| `extract_evidence_claims_node` | Consolidates the summaries into deduplicated evidence claims, each with its supporting URLs. |
| `url_verification_node` | Checks that every cited URL was returned by search in this run; sends the claims back for correction if not. No model call. |
| `reflect_evidence_quality_node` | A model judges whether the evidence suffices or another round is needed. |
| `generate_risk_assessment_node` | Writes the Romanian note from the evidence and sanctions verdict; code assembles the source list. |

Four branches are conditional: the choice of round or final step, skipping article loading, the correction loop to claim consolidation, and another round or the note.

**The final note** is one `gpt-4.1` call. The screening verdict is placed in the prompt as authoritative evidence, ahead of the media evidence. The precedence rule — RED forces a critical score and a terminate or suspend recommendation, YELLOW a high score, an undetermined or errored screening a suspension — is in the prompt, not in code, and code does not check the output against it. Code assembles the source list from the cited URLs and the names of matched sanctions datasets.

**Where the model boundary sits.** Code controls the round order, all filtering and deduplication, the screening verdict except for sanctions-target matches, the URL check and the source list. The model produces the search queries, the verdict for target matches, page summaries, evidence claims, the sufficiency judgement and the full note, including risk score and recommendation.

**Evaluation.** The system has no evaluation component: no automated test suite, no accuracy measure, and no separate evaluation of the screening pipeline or of individual steps. A comparison against five reference cases was run once during development testing and is not part of the system (as described by the system owner).

## 6. Agent scope and tools

**Scope.** The research agent handles one counterparty at a time: it finds public adverse-media evidence and writes the note. The models decide what to search for, how to summarise and classify findings, when the evidence suffices and what the note says. The round order, tools, filters and report format are fixed in code.

**Tools.** The tools are web-search API calls: each sends search text to an external service and returns a list of links.

- **Google search** — via SerpAPI, a service that runs Google searches on the system's behalf.
- **Tavily search** — via Tavily, a web-search service designed for AI applications; also used in the free-form round.

The model writes the search text and chooses the language; code sets the remaining parameters, limits the results and removes duplicates and unwanted sites. Each service uses its own API key, supplied at start-up.

**What happens with the results.** Code retrieves the text of the linked pages through Tavily, filters it and passes the remaining pages to the models. The agent's later decisions — what to summarise, whether to search again, what to write — therefore rest on public web content.

## 7. Data

**Data roles.**

- **Training data** (used to train a model): none — the bank trains no model.
- **Validation data** (used to make or check design choices): not determinable from the repository.
- **Test data** (used to assess the final system before deployment): none maintained; five reference cases were used once during development testing (as described by the system owner).
- **Production data** (acquired in operation, for which the system produces output): the daily monitoring set and what the suppliers return for it.

**Data acquisition and nature.** Production data comes from the bank's **Oracle data warehouse** — the short name and registration number of each monitored legal entity, read with a select-only query — and, on each run, from the suppliers: sanctions and ownership data from OpenSanctions, and search results and page text via SerpAPI and Tavily. This data is **real**. The system also uses a **synthetic dataset**: the reference articles behind the relevance filter, generated once by a language model. The keyword dictionary is semi-synthetic — developer-written seed terms expanded by a language model (data augmentation). Presence in the monitoring set is itself information about the bank's business relationships.

**On personal data.** As described by the system owner, the system processes no personal data: its subjects are legal entities, only their corporate identity is taken from the warehouse, and its reference articles are synthetic.

**Outputs.** Intermediate files hold the screening verdicts and pass them to the research agent. Finished records — counterparty identifier, archive date, run timestamp, note and verdict — are written to the bank's database (as described by the system owner; the code shows the record assembled in that shape and written to files). Diagnostics go to log files (section 10).

## 8. Interfaces and what leaves the Bank

| Service | Endpoint | Sent | Returned | Contains client or counterparty data |
|----|----|----|----|----|
| OpenSanctions | `api.opensanctions.org` — match, entity retrieval, health check | Registration number and jurisdiction | Matching entities, risk topics, sanctions programmes, datasets, directors and owners | Yes |
| OpenAI | `api.openai.com` — chat completions, embeddings, model list | Sanctions match data, screening verdict, generated queries, full page text, counterparty name, consolidated evidence | Risk classification, page summaries, evidence claims, embeddings, final note | Yes |
| Tavily | `api.tavily.com` — search and extract | Queries with the counterparty name and risk terms, free-form queries with the counterparty name and AML topics based on previous results, URLs to extract | Result links and full page text | Yes |
| SerpAPI | `serpapi.com` — search | Google queries with the counterparty name and risk terms | Result links | Yes |

All four suppliers are external, and calls go directly to their public endpoints. Every monitored counterparty's identity leaves the bank; OpenAI also receives the screening result and the full text of retrieved pages.

## 9. How the output is used

The output is a stored record plus supporting files. It does not reach customers: nothing sends the note to a counterparty, publishes it or communicates it outside the bank.

It is a recommendation, not an action. The note gives a recommendation and risk score; the verdict states whether engagement may proceed. The system acts on neither: it opens no case, blocks no account, updates no record beyond its own output and notifies no one.

No person stands between the model's output and the stored record. The check that decides whether to search again is a model call, and there is no approval step, queue, sign-off or means to accept, reject or edit a note. Any review happens outside the system.

## 10. Logging and audit trail

**Mechanism.** Both pipelines use Python's standard logging, each writing its own log file and console output. The screening log appends across runs; the research log is overwritten each run and records full detail. Neither rotates, and logs are not forwarded to a central platform.

**What is recorded.**

- *Runs and items:* each counterparty's name and registration number at start; the screening verdict summary; the research record's location, or the failure with a full error trace; counts of rows and files read.
- *Supplier calls:* OpenSanctions failures and retries; truncated search queries with result counts. Returned content is not recorded.
- *Model calls:* prompts are not recorded, except the final note's input (screening record and claims). Claims and the sufficiency judgement with its reasoning are recorded in full; page summaries and the note text are not. Model names and token usage are not recorded.
- *Intermediate decisions:* counts per filter, relevance score ranges, dropped and unverified links.
- *Errors and retries:* recorded with the error message.

**Linking.** There is no run identifier. Timestamps are local, to the second, without time zone. Only the first line of each counterparty's section names it; later lines follow in order. The stored record carries the registration number and run time (local); the embedded screening record has a UTC timestamp.

**The stored record as audit evidence.** Each record holds the registration number, an archive date (the day before the run), the run time, the note with its sources, and the full screening record. It does not hold model, prompt or code versions, queries, page text, summaries, claims or relevance scores. No human review is recorded.

**Sensitive content in logs.** Both logs hold counterparty names and registration numbers; the research log also holds the screening record, claims and page URLs. The OpenSanctions, OpenAI and Tavily keys are not logged. The SerpAPI key is sent as a URL parameter, and network-error messages from failed Google searches include that URL, so the key can reach the research log and console.

**Protection of the logs.** Logs stay on the host. Nothing in the code prevents the writing process from altering or deleting them; the research pipeline truncates its own log each run. Permissions, retention and forwarding are set outside the repository.

**Failure of logging.** If a log file cannot be opened, the pipeline fails at start; a write failure mid-run is reported to standard error and the run continues.

| To reconstruct | Recorded — yes / partly / no | Where |
|---|---|---|
| The inputs read from Bank systems | Partly | Stored record — registration number and name |
| What each supplier was sent and returned | Partly | Stored record — OpenSanctions results; research log — truncated queries and counts, until the next run |
| Which model and model version produced each generated part | No | — |
| Which prompt version was used | No | — |
| Which code version ran | No | — |
| The intermediate steps that led to the output | Partly | Research log, until the next run; stored record — source list only |
| Whether a person reviewed it, who, and what they decided | No | — |

A given note cannot be reconstructed later from what the system records.

## 11. Change

Behaviour can change without the bank deploying anything:

- the model names `gpt-4o`, `gpt-4.1`, `gpt-4.1-mini` and `text-embedding-3-large` are floating aliases, not dated snapshots — when the supplier updates a model, the next run uses it;
- OpenSanctions data is updated continuously, so a counterparty can screen differently on different days;
- web search results change constantly, so a repeated investigation will not retrieve the same pages.

**Static** — changing only when someone replaces them: the sanctions-programme and dataset reference maps, the synthetic reference articles, the keyword lists, the prompt texts, the filter cut-offs and step limits, and the round sequence.

There is no training, fine-tuning, online learning or feedback loop; nothing the system produces changes its future behaviour.
