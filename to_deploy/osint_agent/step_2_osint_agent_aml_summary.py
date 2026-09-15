#!/usr/bin/env python
# coding: utf-8

# In[27]:


#!/usr/bin/env python
 # coding: utf-8

 # ══════════════════════════════════════════════════════════════
 # HOW TO RUN FROM AIRFLOW: the Airflow job (DAG task) calls these functions, in this order
 # ══════════════════════════════════════════════════════════════
 #
 #   from osint_agent.step_2_osint_agent_aml_summary import (
 #       Config, make_engine, load_contragents, run_osint_agent, upload_reports_to_oracle)
 #
 #   def osint_agent_task():                          # e.g. the python_callable of a PythonOperator
 #       # 1. read the companies from Oracle (table Config.SOURCE_TABLE)
 #       engine = make_engine(ORACLE_SQL_USERNAME, ORACLE_SQL_PASSWORD,
 #                            ORACLE_SQL_CONNECTION_STRING, ORACLE_SQL_SERVICE_NAME)
 #       contragents = load_contragents(engine, Config.SOURCE_TABLE).to_dict("records")
 #       engine.dispose()
 #
 #       # 2. research every company -> writes output_ad_media/{contragentid}.json
 #       run_osint_agent(
 #           contragents=contragents,                   # list of dicts with ID, IDENTIFYCODE, SNAME
 #           openai_url=OPENAI_URL,                     # APISIX, format http://<gateway>:8002/api/providers/openai/v1
 #           openai_api_key=OPENAI_API_KEY,
 #           tavily_url=TAVILY_URL,                     # APISIX base URL, WITHOUT /search or /extract
 #           tavily_api_key=TAVILY_API_KEY,
 #           serp_api_key=SERP_GOOGLE_API_KEY,          # SerpAPI goes direct, no gateway URL
 #           NUM_RESULTS_PER_QUERY=5,
 #       )
 #
 #       # 3. upload the reports -> Config.TARGET_TABLE (DM_NM.NOTA_MONITORIZARE_PJ_DAILY_OSINT_AGENT)
 #       upload_reports_to_oracle(
 #           reports_dir=Config.AD_MEDIA_DIR,
 #           ORACLE_SQL_USERNAME=ORACLE_SQL_USERNAME,
 #           ORACLE_SQL_PASSWORD=ORACLE_SQL_PASSWORD,
 #           ORACLE_SQL_CONNECTION_STRING=ORACLE_SQL_CONNECTION_STRING,   # "host:port"
 #           ORACLE_SQL_SERVICE_NAME=ORACLE_SQL_SERVICE_NAME,
 #       )
 #
 #   All keys, gateway URLs and DB credentials come from Airflow (Variables / Connections).
 #   .env is only for local runs. openai_url / tavily_url = None -> calls go direct to OpenAI / Tavily.
 #   output_ad_media/ and aml_research.log are written next to this file: the folder must be writable.
 #
 # CHANGES vs the previous PROD version (what the Airflow job must adapt)
 #   - step_1_collect_data_opensanc.py is no longer used (OpenSanctions removed): remove that task.
 #     make_engine and load_contragents are now imported from THIS file.
 #   - run_osint_agent: new argument contragents; serpapi_url removed.
 #   - upload_reports_to_oracle: same arguments as before. Writes IDNO (the target table needs an
 #     IDNO column), no OPENSANC_DATA. A same-day rerun replaces rows (DB user needs DELETE rights).
 #   - preflight_check_api_keys removed.
 #
 # QUESTIONS FOR DEVOPS (please answer before the first run)
 #   1. Tavily through APISIX: what is the base URL? The code adds /search and /extract to it.
 #      Are both routes set up under that one base URL?
 #   2. OpenAI through APISIX: is /embeddings routed too, not only /chat/completions?
 #      Models used: gpt-4.1, gpt-4.1-mini, text-embedding-3-large.
 #   3. Is the key sent in the "apikey" header on the Tavily route as well as on OpenAI? Is that key
 #      the APISIX consumer key (the gateway adds the real vendor key) or the vendor key itself?
 #   4. SerpAPI does NOT go through the gateway. Can the Airflow worker reach https://serpapi.com directly?
 #   5. Task duration: about 15 min per company. What execution_timeout does the DAG task have,
 #      and is it enough for the daily number of companies?
 #
 # HOW THIS .py IS PRODUCED (the notebook 4_11_3_research_assistant.ipynb is the source)
 #   jupyter nbconvert --to script 4_11_3_research_assistant.ipynb --output step_2_osint_agent_aml_summary
 #   then in the .py change the import lines "from agent_components." to "from .agent_components."
 #   (package imports, needed when Airflow imports osint_agent as a package)
 # ══════════════════════════════════════════════════════════════

 # ──────────────────────────────────────────────────────────────
 # Dependencies — install with:
 #
 #   pip install -r requirements.txt            (pinned versions, file next to this script)
 #
 # Notes:
 #   - google-search-results : SerpAPI client (imported as `serpapi`); used in place of
 #                             the native Google API key — see SERP_GOOGLE_API_KEY below.
 #   - langchain-core        : pulled in transitively by langchain-openai, listed for clarity.
 #   - tavily-python         : Tavily search + /extract content endpoint.
 #   - scikit-learn          : only for cosine_similarity in HyDe semantic filter.
 #
 # Local modules (NOT installable via pip — copy the folder from this project):
 #   - agent_components/prompts_v2.py             system + user prompt templates
 #   - agent_components/states_v2.py              Pydantic models, UnifiedResearchState,
 #                                                custom reducers (merge_links_by_url, …)
 #   - agent_components/query_components_v2.py    QueryComponentsCalc — builds per-language
 #                                                Google/Tavily query payloads
 #   - agent_components/dictionary.py             ~1.6k topic keywords (financial / corruption /
 #                                                organized_crime × en/ro/ru) for filter_key_terms_node
 #   - agent_components/utils.py                  APIVault + small text helpers
 #   - agent_components/agents_personas.json      NOT used any more (personas removed)
 #   - agent_components/agents_hyde_articles.json ~100 pre-baked HyDe reference articles per topic
 #
 # Python: 3.10+ (Literal / Annotated / get_args usage from typing)
 # ──────────────────────────────────────────────────────────────


 # ──────────────────────────────────────────────────────────────
 # External components & dependencies
 # ──────────────────────────────────────────────────────────────
 #
 # EXTERNAL APIs
 #   1. OpenAI API                  through APISIX (openai_url); openai_url=None -> https://api.openai.com
 #        - gpt-4.1                 URL summarisation, evidence-claim consolidation,
 #                                  reflection, final risk assessment
 #        - gpt-4.1-mini            tool-selection, search-payload generation,
 #                                  name-variation, expert/HyDe generation
 #        - text-embedding-3-large  HyDe cosine-similarity scoring (3072-dim)
 #        Auth: openai_api_key (also sent as "apikey" header for APISIX)
 #
 #   2. Tavily API                  through APISIX (tavily_url); tavily_url=None -> https://api.tavily.com
 #        - /search                 multi-language adverse-media search (advanced depth)
 #        - /extract                page content extraction, batches of 20 URLs
 #        Auth: tavily_api_key (also sent as "apikey" header for APISIX)
 #
 #   3. SerpAPI (Google Search)     https://serpapi.com  (direct, NOT through the gateway)
 #        - /search                 Google adverse-media search with hl/lr language scoping
 #        Auth: serp_api_key, sent as the api_key query parameter
 #
 # INPUT (passed in, not read from files)
 #   - contragents: rows [{"ID": ..., "IDENTIFYCODE": ..., "SNAME": ...}] given to run_osint_agent()
 #     (ID = DWH contragent ID, written to contragentid; IDENTIFYCODE = IDNO; SNAME = searched name)
 #     by the caller (read with make_engine + load_contragents from this file, see HOW TO RUN FROM AIRFLOW)
 #
 # LOCAL FILES (must exist next to this script)
 #   - .env                                    local runs only; under Airflow secrets come as arguments
 #   - agent_components/                       package described in the header above
 #
 # OUTPUTS WRITTEN BY THIS SCRIPT
 #   - output_ad_media/{contragentid}.json      one FinalReport per contragent (Romanian
 #                                             fields: rezumat_analiza, scor_risc,
 #                                             analiza_suspiciuni, situatie_actuala,
 #                                             traiectorie, recomandare_relatie_afaceri,
 #                                             rezultate_cautare, concluzie_finala).
 #                                             Directory is WIPED at the start of every run.
 #   - aml_research.log                        write-mode log, TRUNCATED each run
 #                                             (see logging.FileHandler at top of script)
 #   - Oracle DM_NM.NOTA_MONITORIZARE_PJ_DAILY_OSINT_AGENT   one row per report file, written by
 #                                             upload_reports_to_oracle() (not by run_osint_agent)
 #


# In[ ]:


# Standard Library
import os
import re
import sys
import copy
import json
import time
import asyncio
import operator
import threading
import traceback
import shutil
from datetime import datetime, date, timedelta
from typing import TypedDict, Annotated, List, Dict, Optional, Set, Literal, Callable, get_args
from urllib.parse import urlparse
from collections import Counter

# Environment
from dotenv import load_dotenv

# OpenAI
from openai import OpenAI, APITimeoutError, LengthFinishReasonError

# LangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, AnyMessage, ToolMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool, StructuredTool, InjectedToolArg
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.prompts import ChatPromptTemplate

# LangGraph
from langgraph.graph import add_messages, START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# Pydantic
from pydantic import BaseModel, Field, ValidationError

# Search
# 1from googleapiclient.discovery import build
from tavily import TavilyClient
from tavily.errors import InvalidAPIKeyError, ForbiddenError, BadRequestError, MissingAPIKeyError
from serpapi import GoogleSearch
import requests

# ML
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import numpy.typing as npt

# Notebook
import subprocess
#from IPython.display import Image, display

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from pathlib import Path
from types import SimpleNamespace

from .agent_components.config import Config

from sqlalchemy import create_engine, text, URL, Integer, DateTime
from sqlalchemy.dialects.oracle import NCLOB
import pandas as pd


load_dotenv(Config.PROJECT_DIR / ".env", override=True) 


# In[ ]:


#!jupyter nbconvert --to script 4_11_2_research_assistant.ipynb 


# In[30]:


import logging
logger = logging.getLogger("aml_research")    # safe at import — no handlers, no file

def setup_logging():
      """Attach file + console handlers. Call at run start, NOT at import."""
      logger.setLevel(logging.DEBUG)
      if logger.handlers:               # guard against double-setup
          logger.handlers.clear()

      file_handler = logging.FileHandler(Config.RESEARCH_LOG, mode='w', encoding="utf-8")
      file_handler.setLevel(logging.DEBUG)

      console_handler = logging.StreamHandler()
      console_handler.setLevel(logging.INFO)

      formatter = logging.Formatter(
          "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
          datefmt="%Y-%m-%d %H:%M:%S",
      )
      file_handler.setFormatter(formatter)
      console_handler.setFormatter(formatter)

      logger.addHandler(file_handler)
      logger.addHandler(console_handler)
      logger.info(f"Logger initialized — writing to {Config.RESEARCH_LOG}")


# In[31]:


# Prompts (all from prompts_v2 - system + user templates used across the graph)
from .agent_components.prompts_v2 import (
    extract_evidence_claims_prompt,      # user prompt for consolidating per-URL summaries into EvidenceClaim list
    expert_instructions,                 # system prompt for journalist-persona generation (subgraph)
    url_summary_instructions,            # system prompt for per-URL ContentSummary generation
    system_messages_search_tools,        # dict: tool_name -> system message for search-payload generation
    final_summary_prompt,                # system prompt for final FinalReport (Romanian)
    lead_researcher_prompt               # system prompt for scenario-loop / lead-researcher routing
)

# States & Pydantic models (everything from states_v2 the notebook needs at runtime)
from .agent_components.states_v2 import (
    Journalist, HydePerspectives,                          # persona + persona-list models
    AllowedClaimType, EvidenceClaim, ClaimsFromSummaries,  # claim taxonomy + consolidated-claim container
    ScenarioPool, Scenario_Selected,                       # Literal of scenario names + structured-output wrapper
    LinkCollection, AllowedSeverityLevel, ContentSummary,  # per-URL record + severity Literal + LLM summary schema
    QueryComponentsInState,                                # serialized query-builder output stored in state
    DOMAIN_EXCLUDE,                                        # global domain blocklist (facebook/wiki*)
    AllowedEvidenceAssessment, AssessEvidenceQuality,      # repeat_search/convinced Literal + structured response
    merge_links_by_url, merge_query_data_by_id,            # custom LangGraph reducers (append + non-None-only update)
    UnifiedResearchState, QueryPerformance,                # main TypedDict state + per-query metrics record
    FinalReport                                            # final structured AML report (Romanian fields, 0-100 scor_risc)
)

# Query components - builds Google/Tavily query payloads and topic keyword sets per language
from .agent_components.query_components_v2 import QueryComponentsCalc

# ~1.6k-term keyword dictionary (financial/corruption/organized_crime x en/ro/ru) used by filter_key_terms_node
from .agent_components.dictionary import topic_keywords

# Utilities: APIVault (key store), text helpers, leaf counter for nested dicts
from .agent_components.utils import ( APIVault, first_n_words ,  list_to_string , count_leaves)


# In[32]:


logger.info("All imports successful")


# #### connect to DB

# In[ ]:


def make_engine(ORACLE_SQL_USERNAME, ORACLE_SQL_PASSWORD,
                  ORACLE_SQL_CONNECTION_STRING, ORACLE_SQL_SERVICE_NAME,
                  pool_size=1, max_overflow=0):
      # Validate the passed-in credentials — fail loud before building a bad URL.
      creds = {
          "ORACLE_SQL_USERNAME":          ORACLE_SQL_USERNAME,
          "ORACLE_SQL_PASSWORD":          ORACLE_SQL_PASSWORD,
          "ORACLE_SQL_CONNECTION_STRING": ORACLE_SQL_CONNECTION_STRING,
          "ORACLE_SQL_SERVICE_NAME":      ORACLE_SQL_SERVICE_NAME,
      }
      missing = [name for name, val in creds.items() if not val]
      if missing:
          raise RuntimeError(f"make_engine missing required credentials: {missing}")

      host, _, port = ORACLE_SQL_CONNECTION_STRING.partition(":")   # "host:port"
      url = URL.create(                                               # escapes special characters (e.g. @ in the password)
          "oracle+oracledb",
          username=ORACLE_SQL_USERNAME,
          password=ORACLE_SQL_PASSWORD,
          host=host,
          port=int(port) if port else None,
          query={"service_name": ORACLE_SQL_SERVICE_NAME},
      )
      engine = create_engine(
          url,
          pool_size=pool_size,
          max_overflow=max_overflow,
      )
      return engine



# In[ ]:


# ──────────────────────────────────────────────────────────────
# Load contragents from DM_NM
# ──────────────────────────────────────────────────────────────

def load_contragents(engine, SOURCE_TABLE) -> pd.DataFrame:
    """Read ARCDATE, ID, IDENTIFYCODE, SNAME from DM_NM and return cleaned DataFrame."""
    sql = f"SELECT ARCDATE, ID, IDENTIFYCODE, SNAME FROM {SOURCE_TABLE}"
    df = pd.read_sql(sql, engine)
    df.columns = df.columns.str.upper()
    df["IDENTIFYCODE"] = df["IDENTIFYCODE"].astype(str).str.strip()
    df["SNAME"]        = df["SNAME"].astype(str).str.strip()
    return df

#logger.info(f"Loaded {len(df_contragents)} rows from {SOURCE_TABLE}")
#df_contragents.head()


# In[ ]:


# ──────────────────────────────────────────────────────────────
# Upload reports to DM_NM
# ──────────────────────────────────────────────────────────────

# Why: writes the report files into the Oracle target table, one row per file.
# Used by: __main__ / the Airflow task, after run_osint_agent(). Never for fallback test runs.
# Arguments are the same as in the PROD version, so an existing call keeps working.
def upload_reports_to_oracle(reports_dir, ORACLE_SQL_USERNAME, ORACLE_SQL_PASSWORD,
                             ORACLE_SQL_CONNECTION_STRING, ORACLE_SQL_SERVICE_NAME,
                             table_target=Config.TARGET_TABLE) -> dict:
    """
    Upload every report file (*.json) in reports_dir to table_target.

    Duplicates: for each file, the row with the same ARCDATE + CONTRAGENTID is deleted and the
    new row inserted in ONE transaction. Running the upload twice on the same day replaces that
    day's rows instead of duplicating them. Rows from other days are never touched.
    A bad file (unreadable, missing field, DB error) is logged and skipped; the rest continue.

    Returns {"uploaded": int, "failed": int, "failures": [(filename, reason), ...]}
    """
    files = sorted(Path(reports_dir).glob("*.json"))
    if not files:
        logger.warning(f"[UPLOAD] No report files in {reports_dir}, nothing to upload")
        return {"uploaded": 0, "failed": 0, "failures": []}
    logger.info(f"[UPLOAD] Found {len(files)} report files in {reports_dir}")

    engine = make_engine(ORACLE_SQL_USERNAME, ORACLE_SQL_PASSWORD,
                         ORACLE_SQL_CONNECTION_STRING, ORACLE_SQL_SERVICE_NAME)   # raises if a credential is missing

    delete_sql = text(f"DELETE FROM {table_target} WHERE ARCDATE = :arcdate AND CONTRAGENTID = :contragentid")
    insert_sql = text(f"""
        INSERT INTO {table_target} (ARCDATE, CONTRAGENTID, IDNO, RUNTIME, AML_REPORT)
        VALUES (:arcdate, :contragentid, :idno, :runtime, :aml_report)
    """)

    uploaded, failed, failures = 0, 0, []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            params = {
                "arcdate":      datetime.strptime(data["arcdate"], "%Y-%m-%d"),
                "contragentid": int(data["contragentid"]),
                "idno":         str(data["idno"]),
                "runtime":      datetime.strptime(data["runtime"], "%Y-%m-%d %H:%M:%S"),
                "aml_report":   json.dumps(data["aml_report"], ensure_ascii=False),
            }
            with engine.begin() as conn:          # one transaction: delete + insert both happen, or neither
                replaced = conn.execute(delete_sql, {"arcdate": params["arcdate"], "contragentid": params["contragentid"]}).rowcount
                conn.execute(insert_sql, params)
            uploaded += 1
            note = f", replaced {replaced} existing row(s) for {data['arcdate']}" if replaced else ""
            logger.info(f"[UPLOAD] Uploaded {fp.name} (CONTRAGENTID={params['contragentid']}, IDNO={params['idno']}){note}")
        except Exception as e:
            failed += 1
            reason = f"{type(e).__name__}: {str(e)[:200]}"
            failures.append((fp.name, reason))
            logger.error(f"[UPLOAD] FAILED {fp.name}, skipped: {reason}")

    engine.dispose()
    logger.info(f"[UPLOAD] Done: {uploaded} uploaded, {failed} failed")
    return {"uploaded": uploaded, "failed": failed, "failures": failures}


# #### Define LLM Engines

# In[ ]:


def out_of_credits(e) -> str:
    """Log label when OpenAI says the account has no credits left (code 'insufficient_quota').
    Also checks wrapped causes (e.g. RuntimeError(...) from e), up to 5 levels."""
    for _ in range(5):
        if e is None:
            break
        if getattr(e, "code", None) == "insufficient_quota":
            return " | OPENAI OUT OF CREDITS (insufficient_quota)"
        e = e.__cause__ or e.__context__
    return ""


class Models:
    """Plain LLMs + embedding from one OpenAI key — no tool binding here."""

    def __init__(self, openai_url, openai_api_key):
        gateway = {"base_url": openai_url, "default_headers": {"apikey": openai_api_key}}   # APISIX; openai_url=None = direct to OpenAI
        def chat(model, **kw):
            return ChatOpenAI(model=model, api_key=openai_api_key, max_retries=3, **gateway, **kw)
        self.llm_search_tool_payload = chat("gpt-4.1-mini", temperature=0.2, max_tokens=2000, top_p=0.95, timeout=120)
        self.llm_names_variation     = chat("gpt-4.1-mini", temperature=0.0, max_tokens=400, timeout=30)   # was 50: long names were cut off
        self.llm_url_content_summary = chat("gpt-4.1",      temperature=0.2, max_tokens=2000, top_p=0.95, timeout=120)
        self.llm_agg_summaries       = chat("gpt-4.1",      temperature=0.1, max_tokens=10000, top_p=0.95, timeout=120)
        self.llm_evaluation          = chat("gpt-4.1",      temperature=0.2, max_tokens=3500, top_p=0.95, timeout=120)
        self.llm_lead_researcher     = chat("gpt-4.1",      temperature=0.7, max_tokens=3000, top_p=0.95, timeout=60)
        self.embedding_cross_lang    = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-large", timeout=60, max_retries=3, **gateway)  # default is NO timeout (can hang forever)

        


# In[36]:


logger.info("LLMs are defined")


# #### Variations of company names

# In[37]:


# Why: produces a list of plausible name variants (original, transliterations, no-spaces) and a compiled regex so search payloads and post-extract filters can recognise the same entity across languages.
# Used by: query-component setup and extract_content_node for entity-presence filtering.
def generate_name_regex(entity_name: str, llm_instance: BaseChatModel) -> tuple[list[str], re.Pattern]:
    """
    Generate name variations using LLM and compile regex pattern for matching.
    Uses LLM to create multiple name variants (original, transliterated Russian, 
    versions without spaces) and builds a case-insensitive regex to match any variant.
    
    Args:
        entity_name: Company or entity name to generate variations for
        llm_instance: Language model instance for generating name variations
        
    Returns:
        tuple containing:
            - entity_names_variations (list[str]): List of name variant strings
            - names_search_regexp (re.Pattern): Compiled regex pattern matching any variant
            
    Raises:
        RuntimeError: If LLM invocation fails or returns invalid response
        ValueError: If no valid name variations can be generated
        
    Example:
        ['Acme SRL', 'Акме СРЛ', 'AcmeSRL', 'АкмеСРЛ', 'Acme', 'Акме']
    """
    prompt = f"""
    You are an expert at generating a single-line string of company name variants separated by '|'.

    OUTPUT FORMAT (choose exactly one based on spaces in the Original name):
    - If the Original name CONTAINS whitespace:
    <ORIGINAL_NAME>|<RUSSIAN_NAME>|<ORIGINAL_NO_SPACES>|<RUSSIAN_NO_SPACES>
    - If the Original name DOES NOT CONTAIN whitespace:
    <ORIGINAL_NAME>|<RUSSIAN_NAME>

    Rules:
    - ORIGINAL_NAME = the input company name verbatim (do not modify punctuation, casing, or suffixes).
    - RUSSIAN_NAME = a **transliteration/phonetic rewrite** of ORIGINAL_NAME into Cyrillic. **Do NOT translate any words** (brand words, common nouns, legal suffixes, country names, etc.). Only rewrite letters to approximate English pronunciation. If the input is already Cyrillic/Russian, repeat it verbatim. If unsure, copy the original as is.
    - ORIGINAL_NO_SPACES = ORIGINAL_NAME with ALL whitespace removed (preserve punctuation/casing). Include ONLY if the Original name contains whitespace.
    - RUSSIAN_NO_SPACES = RUSSIAN_NAME with ALL whitespace removed (preserve punctuation/casing). Include ONLY if the Original name contains whitespace.
    - Preserve hyphens, punctuation, and casing exactly where they appear in ORIGINAL_NAME.
    - Output MUST be exactly one line, no leading/trailing spaces, no extra spaces around '|', no quotes, no notes, no extra variants, no newlines.

    Original name: {entity_name}
    """.strip()

    try:
        resp = llm_instance.invoke(prompt)
        pipe_line = (getattr(resp, "content", "") or "").strip()

        if not pipe_line:
            raise ValueError("LLM returned empty response")

        if resp.response_metadata.get("finish_reason") == "length":
            raise ValueError("answer was cut off at max_tokens — name variants incomplete")

        # Parse "Original | Russian | ...": drop empty parts (an empty part would match every page),
        # and always keep the DB name itself
        parts = [p.strip() for p in pipe_line.split("|") if p.strip()]
        if entity_name not in parts:
            parts.insert(0, entity_name)
        logger.info(f"[generate_name_regex] {entity_name} → {len(parts)} variants: {parts}")

        if not parts:
            raise ValueError("No valid name variations generated")

        # Compile strict, case-insensitive pattern
        pattern_str = r"(?:%s)" % "|".join(map(re.escape, parts))
        name_re = re.compile(pattern_str, re.IGNORECASE)

        return parts, name_re
    
    except Exception as e:
        raise RuntimeError(f"Name variation generation failed for '{entity_name}': {e}") from e
    

# Why: derives the scheme://netloc stub from a URL so we can dedupe, blocklist, and group results by source.
# Used by: tool execution nodes when building LinkCollection.displayLink.
def extract_domain(url:str) -> str: 
    parsed = urlparse(url)
    displayLink = f"{parsed.scheme}://{parsed.netloc}"
    return displayLink



# ####  Google tool definition

# In[38]:


# Why: pulls the right slice of the nested search_queries dict (per search-tool key) so prompt templates get only the queries relevant to the tool they are configuring.
# Used by: search-payload prompt formatting in search_payload_generate_node.
def pull_search_query(
    state: QueryComponentsInState, 
    search_tool: str, 
    search_topic: Optional[str] = None,  # Make it optional
    search_lang: Optional[str] = None     # Make it optional too
):   # Can return string or dict
    """
    Safely retrieve a search query from nested state.
    
    - If search_topic is None: returns state.search_queries.get(search_tool)
    - If search_lang is None: returns state.search_queries.get(search_tool).get(search_topic)
    - Otherwise: returns the specific query string
    
    Returns None if any key in the path doesn't exist.
    """
    queries = state.search_queries.get(search_tool)
    if queries is None:
        return None
    
    # If no topic specified, return the entire tool's queries
    if search_topic is None:
        return queries
    
    topics = queries.get(search_topic)
    if topics is None:
        return None
    
    # If no language specified, return all languages for this topic
    if search_lang is None:
        return topics
    
    return topics.get(search_lang)

#pull_search_query(state, "google")                          # → all google queries
#pull_search_query(state, "google", "financial")             # → all languages for financial
#pull_search_query(state, "google", "financial", "en")       # → single query string


# #### google_search function

# In[39]:


#Confirmed — SerpAPI ignores num=2 and returns 9. 
#This is a known SerpAPI behavior where num is treated as a minimum hint, not a hard limit.

# Why: thin wrapper around SerpAPI that normalises results into the project's LinkCollection shape and returns an empty list on failure instead of raising (retries transient failures and "no results").
# Used by: tool_google_search and connectivity-check cells.
def google_search(
        # llm adjustable, derived from passed json file 
        query: str,
        hl: Literal["en", "ro", "ru"] = "ro",
        lr: Literal["lang_en", "lang_ro", "lang_ru"] = "lang_ro",
        num: int = 10, # Serp ignores it and always return max
        
        # default
        engine : str = "google",
        start: int = 1,
        safe: str = "off",
        filter: int = 0,
       # tbm: str = None, # stands for to be matched
        no_cache: str = "false",   # cached results (1 h) are free and instant
        output: str = "json"
    ) -> List[dict]:
        
        """Search Google via SerpApi for adverse media screening.
        
         Wraps the SerpApi /search endpoint with up to SERPAPI_MAX_ATTEMPTS attempts
         (5 s pause) on timeouts, request/JSON failures and other "error" responses.
         "Google hasn't returned any results" is not retried: logged as INFO, returns an empty list.
         Returns an empty list if all attempts are exhausted.
         Invalid API key / exhausted quota (SERPAPI_FATAL_ERRORS) is not retried: logged as
         ERROR and returns an empty list. Never raises — the research flow always continues.
        
        Args:
            hl: Interface language code. Controls the language of the Google search interface.
   
            lr: Content language restriction using Google's lang_ format. Limits search 
                results to pages in the specified language.
                
            num: Number of search results to return per query. Valid range: 1-10.
                
            query: Search query string. Can use Google search operators like inurl:, 
                site:, intitle:, intext:, etc. This parameter MUST be provided by the LLM
                using the exact query strings from the pre-configured JSON.
                
            engine: Search engine to use. Fixed to "google".
            
            start: Result offset for pagination (0-based indexing where 0 is 
                the first page, 10 is the second page, etc.). Default: 1
                
            safe: Adult content filtering level. Options: "active", "off". 
                
            filter: Enable/disable filtering for similar and omitted results.
                
            no_cache:  Force fresh results, bypassing SerpApi cache. 
            
            output: Response format. Fixed to "json" for structured data.
        
        Returns:
          List[dict] — one dict per result, each with keys:
              link, displayLink, search_engine ("tool_google_search"), query.
          Empty list on exhausted retries or empty response.
        
        Raises:
            Nothing — every failure is logged and returns an empty list.
        """

        # Build params dict
        api_key = key_vault.get_key("serp_google_key")
        params = {
            "engine": engine,
            "q": query,
            "start": start,
            "device": "desktop",
            "hl": hl,
            "lr": lr,
            "safe": safe,
            "filter": filter,
            "num": num,
            "no_cache": no_cache,
            "api_key": api_key,
            "output": output
        }

        google_client = GoogleSearch(params)
        google_client.timeout = SERPAPI_TIMEOUT   # GoogleSearch() does not accept timeout; library default is 60000 s (~16.7 h)

        # Retry on timeouts, request/JSON failures and other SerpAPI errors.
        # "Google hasn't returned any results" is NOT retried — test run 2026-09-12: 0 of 14 such retries found links.
        # Invalid key / exhausted quota is NOT retried: ERROR log + empty list, the flow continues.
        for attempt in range(1, SERPAPI_MAX_ATTEMPTS + 1):
            try:
                results = google_client.get_json()   # can raise on timeout, connection error, empty/invalid body
            except Exception as e:
                reason = f"request failed: {type(e).__name__}: {e}"
            else:
                if results.get("organic_results"):
                    links_collected = [
                        {
                            'link': item.get('link', ''),
                            'displayLink': item.get('displayed_link', ''), #displayed_link correct
                            'search_engine': 'tool_google_search',
                            'query': query,
                        }
                        for item in results["organic_results"]
                    ]
                    if attempt > 1:
                        logger.info(f"[google_search] Succeeded on attempt {attempt}/{SERPAPI_MAX_ATTEMPTS} | query: {query[:80]}")
                    return links_collected[:num]  # Need to enforce because SerpAPI ignores num

                reason = results.get("error") or "no organic_results in response"
                if "run out of searches" in reason.lower():   # SerpAPI docs: 429 "Your account has run out of searches."
                    logger.error(f"[google_search] SERPAPI OUT OF SEARCH CREDITS, not retrying, continuing without Google results: {reason} | query: {query[:80]}")
                    return []
                if any(fatal in reason.lower() for fatal in SERPAPI_FATAL_ERRORS):
                    logger.error(f"[google_search] SerpAPI key invalid, not retrying, continuing without Google results: {reason} | query: {query[:80]}")
                    return []
                if "hasn't returned any results" in reason:
                    logger.info(f"[google_search] Google returned no results, not retrying | query: {query[:80]}")
                    return []

            if attempt == SERPAPI_MAX_ATTEMPTS:
                logger.warning(f"[google_search] NO LINKS RETURNED after {attempt} attempts | last reason: {reason} | query: {query[:80]}")
                return []

            pause = 5 * attempt   # 5 s, then 10 s
            logger.warning(f"[google_search] Attempt {attempt}/{SERPAPI_MAX_ATTEMPTS} failed: {reason} | retrying in {pause}s | query: {query[:80]}")
            time.sleep(pause)
        return []
                            


# #### Test Goolge tool

# In[40]:


# Possible error related to the incorrect payload, usually traced when number of arguments is less than 5
# InjectedToolCall sees what we send to the model to generate, InjectedToolArg is only for function parameters

# Tool wrapper with InjectedToolArg for system-controlled parameters
GOOGLE_SEARCH_DESCRIPTION = (
    "A Google based search engine optimized for comprehensive, accurate, and trusted results. "
    "Preferred over Tavily when broader web coverage is needed, when searching across specific domains, "
    "or when precise control over results is required via search operators."
)


# Why: exposes google_search as a LangChain StructuredTool so the search-payload LLM can call it with parallel_tool_calls=True over per-language query variants.
# Used by: llm_search_tool_payload.bind_tools(...) and tool_execute_search_node.
@tool(name_or_callable = "tool_google_search" ,  description=GOOGLE_SEARCH_DESCRIPTION)
def tool_google_search(
    # LLM controls these parameters
    query: str,
    hl: Literal["en", "ro", "ru"] = "ro",
    lr: Literal["lang_en", "lang_ro", "lang_ru"] = "lang_ro",
    num: int = 10,
    
    # System controls these - hidden from LLM schema
    engine: Annotated[str, InjectedToolArg] = "google",
    start: Annotated[int, InjectedToolArg] = 1, # we verified that thsi parameters are injected , erro message `start` parameter must be an integer greater than or equal to 0. 
    safe: Annotated[str, InjectedToolArg] = "off",
    filter: Annotated[int, InjectedToolArg] = 0,
    no_cache: Annotated[str, InjectedToolArg] = "false",   # was "true"; cached results (1 h) are free and instant
    output: Annotated[str, InjectedToolArg] = "json"
    
) -> List[dict]:
    """Search Google via SerpApi for adverse media screening.
    
    LLM-controllable parameters:
        query: Search query string with Google operators (intext:, site:, etc.)
        hl: Interface language code (en/ro/ru)
        lr: Content language restriction (lang_en/lang_ro/lang_ru)
        num: Number of results to return (1-10)
    
    System-injected parameters (not visible to LLM):
        engine, start, safe, filter, no_cache, output
    
    Returns:
        List of search results with links, display URLs, and metadata.
    """
    return google_search( # later add node for dedupliaction
        query=query,
        hl=hl,
        lr=lr,
        num=num,
        engine=engine,
        start=start,
        safe=safe,
        filter=filter,
        no_cache=no_cache,
        output=output
    )

   


# #### Tavily Search definition

# In[41]:


# common error BadRequestError: Max 20 URLs are allowed.
# https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list#try-it 

## define function first, name of parameters same as in documentation
# llm does not see this function docstring, ist only for my documentation
# however llm sees @tool, docscting content, parameters names and types

# Why: one place that creates the Tavily client: gateway URL (APISIX) + apikey header.
# Used by: tavily_search and tavily_content_extractor.
# api_base_url and client.session exist in the SDK code (tavily-python >= 0.7.20, pinned 0.7.23), not in the Tavily docs.
def make_tavily_client():
    key = key_vault.get_key("tavily_key")
    client = TavilyClient(api_key=key, api_base_url=key_vault.get_key("tavily_url"))
    client.session.headers["apikey"] = key   # APISIX key-auth; add to the headers, never replace them
    return client


# Exception: Error getting results: HTTPSConnectionPool(host='api.tavily.com', port=443): Max retries
# Why: wraps the Tavily search API with project-specific defaults (depth, chunks, excluded domains) and normalises the result schema.
# Used by: tool_tavily_search and the Tavily connectivity check.

def tavily_search(
    query: str,
    exclude_domains: List[str],
    topic: Literal["general", "news", "finance"] = "general",
    search_depth: Literal["basic", "advanced"] = "advanced",
    chunks_per_source: int = 1,
    max_results: int = 10,
    include_answer: bool = False,
 
    hl_dummy: Literal["en", "ro", "ru"] = "ro"

) -> List[dict]:
    """
    Execute a search query using Tavily Search API optimized for AI agents.
    
    Tavily Search provides AI-optimized search results with cleaned content snippets,
    designed specifically for LLM applications and research tasks.
    
    Args:
        query (str): The search query string to execute with Tavily.
        
        exclude_domains: ACCEPTED FOR SIGNATURE COMPATIBILITY ONLY — the function
              overrides this with the module-level _HARDCODED_EXCLUDE_DOMAINS list
              before calling Tavily. Pass [] if you don't care.
        
        topic (Literal["general", "news", "finance"], optional): The category of the search.
            - "general": Broader, general-purpose searches across various sources
            - "news": Real-time updates about politics, sports, and current events
            - "finance": Financial and market-related searches
            Defaults to "general".
        
        search_depth (Literal["basic", "advanced"], optional): The depth of the search.
            - "basic": Generic content snippets (1 API credit)
            - "advanced": Most relevant sources and content snippets (2 API credits)
            Defaults to "advanced".
        
        chunks_per_source (int, optional): Maximum number of content chunks per source.
            Each chunk contains up to 500 characters. Only available when search_depth
            is "advanced". Range: 1-3. Defaults to 1.
        
        max_results (int, optional): Maximum number of search results to return.
            Range: 0-20. Defaults to 10.
        
        include_answer (bool, optional): Include an LLM-generated answer to the query.
            Defaults to False.

        hl_dummy: Interface language code. Used as dummy variable,as Tavily does not support lang argument.         
    
    Returns:
          List[dict] — one per result with keys: link, displayLink, search_engine
          ("tool_tavily_search"), query. Empty list when the response has no "results" key.

      Raises:
          Nothing — transient failures are retried; every failure is logged and returns an empty list.
    """
    
    logger.debug(f"[tavily_search] Query: {query[:80]} | topic: {topic} | max_results: {max_results}")



    params = {
        "query": query,
        "topic": topic,
        "exclude_domains":  Config.EXCLUDE_DOMAINS,
        "search_depth": search_depth,
        "chunks_per_source": chunks_per_source,
        "max_results": max_results,
        "include_answer": include_answer,
    }

    tavily_client = make_tavily_client()

    # Retry transient failures (timeout, 429 rate limit, connection error, 5xx).
    # Invalid key / forbidden / plan limit / bad request is NOT retried.
    # Never raises: on failure log and return [], so the research flow continues.
    for attempt in range(1, TAVILY_MAX_ATTEMPTS + 1):
        try:
            search_results = tavily_client.search(**params)
        except ForbiddenError as e:   # HTTP 403 / 432 plan limit / 433 pay-as-you-go limit
            logger.error(f"[tavily_search] TAVILY OUT OF CREDITS or access denied, not retrying, continuing without Tavily results: {e} | query: {query[:80]}")
            return []
        except TAVILY_FATAL_ERRORS as e:
            logger.error(f"[tavily_search] {type(e).__name__}, not retrying, continuing without Tavily results: {e} | query: {query[:80]}")
            return []
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
        else:
            links_collected = [
                {
                    'link': item.get('url', ''),
                    'displayLink': extract_domain(item.get("url")),
                    'search_engine': 'tool_tavily_search',
                    'query': query,
                }
                for item in search_results.get('results', [])
            ]
            if attempt > 1:
                logger.info(f"[tavily_search] Succeeded on attempt {attempt}/{TAVILY_MAX_ATTEMPTS} | query: {query[:80]}")
            if links_collected:
                logger.debug(f"[tavily_search] Returned {len(links_collected)} results for: {query[:80]}")
            else:
                logger.info(f"[tavily_search] No results (no error) | query: {query[:80]}")
            return links_collected

        if attempt == TAVILY_MAX_ATTEMPTS:
            logger.error(f"[tavily_search] Failed after {attempt} attempts, continuing without Tavily results | last reason: {reason} | query: {query[:80]}")
            return []

        pause = 5 * attempt   # 5 s, then 10 s
        logger.warning(f"[tavily_search] Attempt {attempt}/{TAVILY_MAX_ATTEMPTS} failed: {reason} | retrying in {pause}s | query: {query[:80]}")
        time.sleep(pause)
    return links_collected


# In[42]:


## Now make it as structured tool 

# Possible error related to the incorrect payload, usually traced when number of arguments is less than 5
# InjectedToolCall sees what we send to the model to generate, InjectedToolArg is only for function parameters

# Tool wrapper with InjectedToolArg for system-controlled parameters
TAVILY_SEARCH_DESCRIPTION = (
    "A search engine optimized for comprehensive, accurate, and trusted results. "
    "Preferred over Google when AI-summarized answers are needed across multiple queries in a single call, "
    "or when searching recent news and events with contextual relevance ranking."
)

# TypeError: 'StructuredTool' object is not callable
@tool(name_or_callable = "tool_tavily_search" , description = TAVILY_SEARCH_DESCRIPTION)
# Why: exposes tavily_search as a LangChain tool so the same payload-generation LLM can fan out Tavily searches alongside Google ones.
# Used by: llm_search_tool_payload.bind_tools(...) and tool_execute_search_node.
def tool_tavily_search(
    query: str,
    max_results: int = 10, 
    hl_dummy: Literal["en", "ro", "ru"] = "ro",  # not passed to Tavily, used for LLM consistency only

    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    search_depth: Annotated[Literal["basic", "advanced"], InjectedToolArg] = "advanced",
    chunks_per_source: Annotated[int, InjectedToolArg] = 1,
    include_answer: Annotated[bool, InjectedToolArg] = False

    
) :#-> List[dict]:
    
    """Search using Tavily API for adverse media screening and financial crime investigations.
    
    LLM-controllable parameters:
        query: Search query string for finding adverse media, sanctions, or financial crime information
        max_results: Maximum number of results to return (0-20, default: 10)
        hl_dummy:Interface language code (en/ro/ru)
    
    System-injected parameters (not visible to LLM):
        topic: Search category (general/news/finance)
        search_depth: basic (1 credit) or advanced (2 credits)
        chunks_per_source: Number of content chunks per source (1-3)
        include_answer: Include LLM-generated answer
    
    Note:
        exclude_domains is hardcoded using _HARDCODED_EXCLUDE_DOMAINS constant
    
    Domain exclusion is fixed inside tavily_search() — _HARDCODED_EXCLUDE_DOMAINS
      is applied regardless of caller input.

      Returns:
          List[dict] — see tavily_search().
    """
    return tavily_search(
        query=query,
        exclude_domains= Config.EXCLUDE_DOMAINS,
        topic=topic,
        search_depth=search_depth,
        chunks_per_source=chunks_per_source,
        max_results=max_results,  #  LLM controls this
        include_answer=include_answer
    )


# #### Join ALL search tools

# In[43]:


# join instructions and entity name
## create list of tools, which be called externally with  tool_call['args']
# we separate tools from scenario, custom scenario means custom promot over the baseline 

tools = [   tool_tavily_search , tool_google_search ] # tool_google_search ,
tools_by_name = {tool.name:tool for tool in tools}


# #### Architecture

# In[44]:


# Why: walks scenario_pool deterministically (skipping anything already in scenario_used) so each outer-loop iteration tries a fresh search strategy before exiting.
# Used by: graph entry node, routes via route_trigger_search.
def scenario_selection_node(state: UnifiedResearchState):
    """Node which selects next search scenario deterministically.
    Iterates scenario_pool in order, returns first not yet in scenario_used.
    """
    logger.info(f"[scenario_selection_node] Raw links: {len(state['search_results_raw'])}")

    scenario_used = state["scenario_used"]
    scenario_pool = state["scenario_pool"]

    # Pick first scenario not yet completed
    selected = None
    for scenario in scenario_pool:
        if scenario not in scenario_used:
            selected = scenario
            break
    
    # All scenarios exhausted
    if selected is None:
        selected = "exit_scenario"

    logger.info(f"[scenario_selection_node] Selected: {selected} | Used: {scenario_used}")

    return {"scenario_selected": selected}


# Why: conditional-edge function that maps the just-selected scenario to the right downstream node - baseline search, custom search, or final risk assessment.
# Used by: builder.add_conditional_edges on scenario_selection_node.
def route_trigger_search(state: UnifiedResearchState):
    """Routing function after scenario selection.
    
    Routes:
        - exit_scenario: all scenarios completed → generate final report from whatever evidence exists
        - None/missing: selection failed → END
        - custom_scenario: custom prompt over baseline → generate_custom_payload_node
        - anything else: valid tool selected → search_payload_generate_node
    """
    if state["scenario_selected"] == "exit_scenario":
        logger.warning("[route_trigger_search] Scenarios exhausted → generate_risk_assessment_node")
        return "generate_risk_assessment_node"

    elif not state["scenario_selected"]:
        logger.error("[route_trigger_search] scenario_selected is None or empty — routing to END")
        return "END"

    elif state["scenario_selected"] == "custom_scenario":
        logger.info("[route_trigger_search] custom_scenario → generate_custom_payload_execute_node")
        return "generate_custom_payload_execute_node"

    else:
        logger.info(f"[route_trigger_search] Tool scenario '{state['scenario_selected']}' → search_payload_generate_node")
        return "search_payload_generate_node"


# In[45]:


## Node to create payload for the tool
## this node only suitable for now for baseline non custom tools
# Why: builds the Google/Tavily tool calls in code from the predefined queries for every (topic, language) pair so the next node can dispatch them.
# Used by: graph node after baseline-scenario selection.
def search_payload_generate_node(state: UnifiedResearchState):
    """Builds the Google / Tavily tool calls in code from the predefined queries — no LLM call.

      The queries are already complete in query_components (topic → language → query) and the
      tool arguments follow fixed rules, so every predefined query is used exactly as written.
      Custom (new) queries are generated only by generate_custom_payload_execute_node.
    """
    qc = state["query_components"]
    scenario_selected = state["scenario_selected"]
    tool_pool = state.get("tool_pool") or []

    if scenario_selected not in tool_pool:
        logger.error(f"[search_payload_generate_node] Invalid scenario: '{scenario_selected}' | Valid pool: {tool_pool}")
        raise ValueError(f"Invalid tool selected: {scenario_selected}. Must be one of: {tool_pool}")

    num = state["num_results_alias"]
    engine = "google" if scenario_selected == "tool_google_search" else "tavily"

    tool_calls = []
    for topic, by_lang in pull_search_query(qc, engine).items():
        for lang, query in by_lang.items():
            if scenario_selected == "tool_google_search":
                args = {"query": query, "hl": lang, "lr": f"lang_{lang}", "num": num}
            else:
                args = {"query": query, "max_results": num, "hl_dummy": lang}
            tool_calls.append({"name": scenario_selected, "args": args,
                               "id": f"{scenario_selected}_{topic}_{lang}", "type": "tool_call"})

    logger.info(f"[search_payload_generate_node] Scenario: {scenario_selected} | Entity: {qc.entity_name} | Built {len(tool_calls)} tool calls in code")
    return {"messages": [AIMessage(content="", tool_calls=tool_calls)]}


# In[46]:


## this node should be used for baseline tools and for custom tools 
# can cause error ValidationError: 1 validation error for GoogleSearchSchema

# Why: executes the tool calls produced by the previous node, deduplicates URLs across the run, and appends new LinkCollections plus per-query metrics into state.
# Used by: graph node after search_payload_generate_node.
def tool_execute_search_node(state: UnifiedResearchState):
    """Performs the tool call"""
    logger.info(f"[tool_execute_search_node] Scenario: {state.get('scenario_selected')} | Raw links so far: {len(state['search_results_raw'])}")
    
    result = []
    link_collections = []
    query_states = []

    # initiate counter
    query_id_alias = state.get("query_counter", 0) 

    ## collect links which are already in the database to remove diplicates
    # Track URLs from BOTH previous state AND current batch
    urls_seen = set()
    if state["search_results_raw"]: 
        for link in state["search_results_raw"]:
            urls_seen.add(link.link)     

    # each observation is generaated by separate quer
    for tool_call in state["messages"][-1].tool_calls: # AI message with payload , last ai message with too l calls
        tool_selected = tools_by_name[tool_call["name"]] # select tool, custom scenarious are based on same tools as baseline scenarious , only promot changes 
        observation = tool_selected.invoke(tool_call["args"]) # paste arguments to the tool 
        # common appearance of the JSONDecodeError 
        
        if not observation:
            logger.warning(f"[tool_execute_search_node] '{tool_call['name']}' returned 0 results | query: {tool_call['args'].get('query', '')[:80]}")
        else:
            logger.debug(f"[tool_execute_search_node] '{tool_call['name']}' raw results: {len(observation)} | query: {tool_call['args'].get('query', '')[:80]}")

        # Append Tool Messages 
        result.append(ToolMessage( 
                       content = observation, 
                       tool_call_id=tool_call["id"] , 
                       name = tool_call['name']))      

        ## Collect only meta for query
        query_id_alias += 1
        raw_query = tool_call['args'].get("query")
        query_lang = tool_call['args'].get("hl") or tool_call['args'].get("hl_dummy")

        query_state = QueryPerformance( 
           query_text = [raw_query] if isinstance(raw_query, str) else raw_query,
           query_lang = query_lang ,
           query_id = query_id_alias,
           search_engine = tool_call["name"],
           scenario = state["scenario_selected"],
           links_initial = len(observation)
        )
        query_states.append(query_state)


        # Extract LinkCollection data from observation
        links_before = len(link_collections)
        for item in observation:
            if not isinstance(item, dict):
                logger.error(f"[tool_execute_search_node] Expected dict, got {type(item)} | query_id: {query_id_alias}")
                raise TypeError(
                    f"Expected dict in observation, got {type(item)}. "
                    f"Item content: {item}"
                )
            
            # Check: Do required keys exist?
            required_keys = ['displayLink', 'link', 'search_engine']
            missing_keys = [key for key in required_keys if key not in item]
            
            if missing_keys:
                logger.error(f"[tool_execute_search_node] Missing keys: {missing_keys} | query_id: {query_id_alias} | item: {item}")
                raise KeyError(
                    f"Item missing required keys: {missing_keys}. "
                    f"Available keys: {list(item.keys())}. "
                    f"Item content: {item}. "
                    f"Tool: {tool_call['name']}, Query: {tool_call['args'].get('query')}"
                )
            
            # Is it a duplicate
            if item['link'] in urls_seen:
                    logger.debug(f"[tool_execute_search_node] Duplicate skipped: {item['link']}")
                    continue
            
            ## check if domain is not relevane 
            display = extract_domain(item['link'])
            if display in Config.LOW_VALUE_DOMAINS:
               logger.debug(f"[tool_execute_search_node] Low-value domain skipped: {display}")
               continue
            
            # All checks passed - add the link
            urls_seen.add(item['link'])
            link_collections.append(LinkCollection(
                displayLink=item["displayLink"],
                link=item["link"],
                search_engine=item["search_engine"],
                scenario=state["scenario_selected"],
                query_id=query_id_alias
            ))

        links_this_query = len(link_collections) - links_before
        logger.debug(f"[tool_execute_search_node] Query {query_id_alias} | Raw: {len(observation)} → Kept: {links_this_query} | lang: {query_lang}")
    
        time.sleep(1) 

    engine_counts = Counter(lc.search_engine for lc in link_collections)
    logger.info(f"[tool_execute_search_node] Done | New unique links: {len(link_collections)} | By engine: {dict(engine_counts)} | Total queries: {len(query_states)}")
    
    return {"messages": result ,  
            "search_results_raw":link_collections, 
            "scenario_used":[state["scenario_selected"]],
            "search_query_performance":query_states,
            "query_counter":query_id_alias
              }   


# #### Tavily for content extraction

# In[47]:


#Error generating reputation summary: Error code: 429 - {'error': {'code': 'RateLimitReached', 'message': 'Your requests to gpt-4.1 for gpt-4.1 in 
# West Europe have exceeded the token rate limit for your current AIServices S0 pricing tier. This request was for ChatCompletions_Create under OpenAI Language Model '
#'Instance API. Please retry after 59 seconds. To increase your default rate limit, visit: https://aka.ms/oai/quotaincrease.'}}

# Initialize client once at module level
# https://docs.tavily.com/documentation/api-reference/endpoint/extract

# Tavily API has a limit of 20 URLs per extraction request
# we can face error bad requiest max 20 is allowed

# Why: batches Tavily /extract calls (max 20 URLs per request) and applies the entity-name regex post-extract to drop pages that never mention the target.
# Used by: extract_content_node.
def tavily_content_extractor(
    name_pattern :str,    
    dummy_list: List[LinkCollection], # will be populated with new data from ClassSummary , argument value is search_results_raw  
    extract_depth: str = "advanced", # advanced  basic
    include_raw_content: bool = True
) -> List[LinkCollection]:

    # store urls values, order of output in threqad is differnet from original 
    urls = [link.link for link in dummy_list] # link_collection.link

    ## Extract content of URL in batches, tavily limits to 20 requests
    BATCH_SIZE = 20 
    url_to_raw_content = {} # need to store "url" and "raw_content"
    total_batches = (len(urls) + BATCH_SIZE - 1) // BATCH_SIZE

    tavily_client = make_tavily_client()
    logger.info(f"[tavily_content_extractor] Starting extraction | URLs: {len(urls)} | Batches: {total_batches} | depth: {extract_depth}")

    def first_n_words(text: str, n: int) -> str:
        return " ".join(text.split()[:n])

    for i in range(0, len(urls), BATCH_SIZE ):
        batch_urls = urls[i: ( i + BATCH_SIZE ) ]
        batch_num = i // BATCH_SIZE + 1
        logger.info(f"[tavily_content_extractor] Batch {batch_num}/{total_batches} | URLs: {len(batch_urls)}")

        # Retry transient failures per batch (timeout, 429, connection error, 5xx).
        # Invalid key / forbidden / plan limit / bad request is NOT retried. A failed batch is skipped, the flow continues.
        response = None
        for attempt in range(1, TAVILY_MAX_ATTEMPTS + 1):
            try:
                response = tavily_client.extract(urls=batch_urls, extract_depth=extract_depth, format='markdown', timeout=TAVILY_EXTRACT_TIMEOUT)
                if attempt > 1:
                    logger.info(f"[tavily_content_extractor] Batch {batch_num}/{total_batches} succeeded on attempt {attempt}/{TAVILY_MAX_ATTEMPTS}")
                break
            except ForbiddenError as e:   # HTTP 403 / 432 plan limit / 433 pay-as-you-go limit
                logger.error(f"[tavily_content_extractor] Batch {batch_num}/{total_batches} TAVILY OUT OF CREDITS or access denied, not retrying, batch skipped: {e}")
                break
            except TAVILY_FATAL_ERRORS as e:
                logger.error(f"[tavily_content_extractor] Batch {batch_num}/{total_batches} {type(e).__name__}, not retrying, batch skipped: {e}")
                break
            except Exception as e:
                if attempt == TAVILY_MAX_ATTEMPTS:
                    logger.error(f"[tavily_content_extractor] Batch {batch_num}/{total_batches} failed after {attempt} attempts, batch skipped: {type(e).__name__}: {e}")
                    break
                pause = 5 * attempt   # 5 s, then 10 s
                logger.warning(f"[tavily_content_extractor] Batch {batch_num}/{total_batches} attempt {attempt}/{TAVILY_MAX_ATTEMPTS} failed: {type(e).__name__}: {e} | retrying in {pause}s")
                time.sleep(pause)

        if response is None:
            for url in batch_urls:
                if url not in url_to_raw_content:
                    url_to_raw_content[url] = ""
            continue

        results = response.get("results", [])
        for item in results:
            url = item.get("url", "")
            content = item.get("raw_content", "")
            url_to_raw_content[url] = first_n_words(content , 2500)

        failed = response.get("failed_results", [])
        if failed:
            failed_urls = [(f.get("url", "") if isinstance(f, dict) else str(f))[:60] for f in failed]
            logger.warning(f"[tavily_content_extractor] Batch {batch_num}/{total_batches} — Tavily could not extract {len(failed)} URLs: {failed_urls}")

        empty_content = [url for url, content in url_to_raw_content.items() if url in batch_urls and not content.strip()]
        if empty_content:
            logger.debug(f"[tavily_content_extractor] Batch {batch_num} — empty content for {len(empty_content)} URLs: {[u[:60] for u in empty_content]}")

        logger.debug(f"[tavily_content_extractor] Batch {batch_num} extracted {len(results)} results")

    ## Filter based on KW: We need to keep only non empty content which contains name of the company                    
    # url_to_raw_content to url_to_raw_content_filtered

    name_re = re.compile(name_pattern, re.IGNORECASE) 
    # https://blog.cursuribursa.ro/statistici-si-fapte-interesante-privind-activitatile-de-spalarea-banilor/ does not contain Lukoil in readable section however its present in developers view
    # as part of links
    
    # filter with list comprehension
    url_to_raw_content_filtered = {
    url: content 
    for url, content in url_to_raw_content.items() 
       if name_re.search(content)
               }

    # debug 
    dropped = len(url_to_raw_content) - len(url_to_raw_content_filtered)
    logger.info(f"[tavily_content_extractor] Entity filter | Extracted: {len(url_to_raw_content)} | Matched: {len(url_to_raw_content_filtered)} | Dropped: {dropped}")

    # now unpack the dictionary which contains 3 attributes which we need to add to Link Collection
    # claim_type='other' date_published='Unknown' summary='No financial crime or compliance information is present in the content.'
    updated_collections = []
    for link_obj in dummy_list: 
        link = link_obj.link
        if link in url_to_raw_content_filtered: # eligibel for update
            updated_object = link_obj.model_copy( update = {"raw_content":url_to_raw_content_filtered[link]} )
            updated_collections.append(updated_object)

    logger.info(f"[tavily_content_extractor] Done | Returned {len(updated_collections)} enriched LinkCollections from {len(dummy_list)} input")
    return updated_collections



# In[48]:


# the function above will not be implemented as tool 
# we will update search_results component for each LinkCollection
# We need to avoid returning duplicated data, same approach must be extended to node which assignes scores and filters data 

# Why: takes the URLs found by the current scenario, runs Tavily /extract on them, and keeps only pages where the entity-name regex actually matches.
# Used by: graph node after tool_execute_search_node and generate_custom_payload_execute_node.
def extract_content_node(state: UnifiedResearchState) -> UnifiedResearchState:
    """Node function to extract content from URLs in search results"""

    scenario_selected = state["scenario_selected"]
    dummy_list = state["search_results_raw"].copy()
    links_to_process = [link for link in dummy_list if link.scenario == scenario_selected]
    skipped = len(dummy_list) - len(links_to_process)

    logger.info(f"[extract_content_node] Scenario: {scenario_selected} | Total raw: {len(dummy_list)} | To process: {len(links_to_process)} | Skipped (other scenarios): {skipped}")

    if not links_to_process:
        logger.warning(f"[extract_content_node] No links to process for scenario '{scenario_selected}' — returning empty")
        return {"search_results_entity_filtered": []}

    updated_search_result = tavily_content_extractor(
        dummy_list=links_to_process,
        name_pattern=state["query_components"].names_search_regexp
    )

    logger.info(f"[extract_content_node] Completed | Input: {len(links_to_process)} | Entity-filtered: {len(updated_search_result)} | Dropped: {len(links_to_process) - len(updated_search_result)}")
    return {"search_results_entity_filtered": updated_search_result}



# In[49]:


## now we requre separe node which applies topic filter to search_results_entity_filtered
# unlike raw links, search_results_entity_filtered

# we pass data from search_results_entity_filtered to search_results_kterms_filtered
# search_results_kterms_filtered here uses add operator to add new data

# Why: drops extracted content that does not contain any of the ~1.6k topic keywords (financial / corruption / organized-crime in en/ro/ru) so we do not waste LLM tokens on off-topic pages.
# Used by: graph node after extract_content_node.
def filter_key_terms_node(state: UnifiedResearchState) -> UnifiedResearchState:
    """Filter links that contain at least one keyword from topic_keywords"""
    
    # Unnest keywords
    list_keywords = []
    for topic in topic_keywords:
        for lang in topic_keywords[topic]:
            list_keywords.extend(topic_keywords[topic][lang])
    
    # Compile regex patterns with word boundaries
    # For multi-word phrases, match exact phrase
    patterns = []
    for keyword in list_keywords:
        # Escape special regex characters
        escaped = re.escape(keyword) # add backspaced before special characters so they are treated as literal
        
        # For multi-word: require exact phrase with word boundaries
        if ' ' in keyword:
            pattern = rf'\b{escaped}\b'
        else:
            # Single word: word boundaries
            pattern = rf'\b{escaped}\b'
        
        patterns.append(re.compile(pattern, re.IGNORECASE))
    
    # apply filter only for the new links
    scenario_selected = state["scenario_selected"]
    links_to_process = [
        link for link in state["search_results_entity_filtered"] 
        if link.scenario == scenario_selected
    ]
    
    filtered_links = []
    for link in links_to_process:
        raw_content = link.raw_content or ""
        
        # Check if any pattern matches
        if any(pattern.search(raw_content) for pattern in patterns):
            filtered_links.append(link)
    
    logger.info(f"[filter_key_terms_node] Scenario: {scenario_selected} | Patterns: {len(patterns)} | Input: {len(links_to_process)} | Passed: {len(filtered_links)} | Filtered out: {len(links_to_process) - len(filtered_links)}")
    return {"search_results_kterms_filtered": filtered_links}




# #### Unite Research and HyDe agent into 1

# In[50]:


## reads from main state, we require subgraph only as a function to append data to main state
# Why: short-circuits the expensive HyDe subgraph if hyde_list is already populated, so re-runs in the same session do not regenerate the same reference articles.
# Used by: builder.add_conditional_edges on buffer_node_hyde_generation.
def route_expert_generation(state: UnifiedResearchState):
    """Route which skips hyde generation"""
    hyde_list = state.get("hyde_list")
    if hyde_list is None or len(hyde_list) == 0:
        logger.info("[route_expert_generation] Routing to Expert Generation")
        return "continue"
    else:
        logger.info("[route_expert_generation] HyDE already generated — skipping")
        return "skip_hyde_generation"
    

## define subgraph
class SubgraphHyDeGenerate(TypedDict):
    # read from main state
    search_topic:str   # read from main state
    entity_name:str
    max_journalists: int

    # returs to main state:
    journalists: List[Journalist]
    hyde_list: List[str]        
    


# In[51]:


# Why: loads pre-generated journalist personas for the current topic instead of calling an LLM, keeping the subgraph deterministic and cost-free in normal runs.
# Used by: HyDe subgraph as its first node.
def create_expert_node(state: SubgraphHyDeGenerate):
    """Decorative no-op — personas removed (articles are pre-baked).
    Kept only to preserve the 2-node subgraph structure."""
    return {}
    


# In[52]:


# Why: loads the ~100 pre-baked HyDe reference articles per topic and substitutes the entity name into the <|company_name|> placeholder.
# Used by: HyDe subgraph after create_expert_node.
def generate_hyde_document_node(state: SubgraphHyDeGenerate):
      """Load pre-baked HyDe articles and substitute the entity name → hyde_list."""
      entity_name = state["entity_name"]
      articles = Config.hyde_articles()              # ← was the global hyde_articles_by_topic

      hyde_list = [
          article.replace("<|company_name|>", entity_name)
          for arts in articles.values()
          for article in arts
      ]
      logger.info(f"[generate_hyde_document_node] Loaded {len(hyde_list)} articles for '{entity_name}'")
      return {"hyde_list": hyde_list}


# In[53]:


## now compile graph 
subgraph_builder = StateGraph(SubgraphHyDeGenerate)

subgraph_builder.add_node("create_expert_node" , create_expert_node)
subgraph_builder.add_node("generate_hyde_document_node" , generate_hyde_document_node)

subgraph_builder.add_edge(START, "create_expert_node")
subgraph_builder.add_edge("create_expert_node" , "generate_hyde_document_node" )
subgraph_builder.add_edge( "generate_hyde_document_node", END )

subgraph = subgraph_builder.compile()


# In[54]:


# Why: invokes the HyDe subgraph from the main graph and folds its hyde_list output back into the parent state.
# Used by: graph node on the "enrich with HyDe" branch.
def call_subgraph(state: UnifiedResearchState): 

    """Run the HyDe subgraph to populate `journalists` and `hyde_list`.

      Today the subgraph reads pre-baked personas + HyDe articles from JSON
      (agents_personas.json, agents_hyde_articles.json), so this is effectively a
      loader — no live LLM generation happens in the normal path.

      NB: `search_topic` is passed as `list(hyde_articles_by_topic.keys())[0]` —
      a dummy that pins the subgraph to whichever topic happens to be first in
      the JSON. The subgraph doesn't currently use this value, but if it ever
      starts to, this needs to be replaced with the real scenario topic.

      Returns:
          {"hyde_list": [...], "journalists": [...]}.
      """
    
    logger.info(f"[call_subgraph] Executing | Raw links: {len(state['search_results_raw'])}")

    subgraph_output = subgraph.invoke({
          "entity_name": state["query_components"].entity_name,
          "search_topic": ""        # dummy; create_expert_node ignores it
      })
    
    logger.info(f"[call_subgraph] Done | Articles: {len(subgraph_output['hyde_list'])}")

    return {"hyde_list": subgraph_output["hyde_list"]}


# #### Prepare module for cosine similarity

# In[55]:


## test chunking tool 
#Tavily extraction includes navigation, ads, related articles, footer links
#2000 words of which maybe only 400-600 are actual article content
#The rest dilutes the embedding

text_splitter =  RecursiveCharacterTextSplitter(
    chunk_size = 1000 , # number of characters
    chunk_overlap = 200, 
    length_function = len,
    is_separator_regex = False
)

# text_splitter.split_text(text_sample) # list with strings 

# function will be iteravely applied over each content 
# It assign max values of cosine score

# Why: scores one extracted page against every HyDe reference article via cosine similarity and keeps the max as that page's relevance signal.
# Used by: filter_semantic_similarity_node per URL.
def check_content_similarity(
    search_results: List[LinkCollection], 
    hyde_list: npt.NDArray[np.float64]
) -> List[LinkCollection]:
    
    """
    Assign max HyDE similarity score to each URL using chunked content
    
    Args:
        search_results: List of LinkCollection objects with raw_content
        hyde_list: numpy array of shape (n_hyde, 3072) containing Hyde embeddings
    
    Returns:
        List of LinkCollection with hyde_score populated (same length as input)
    """
    
    # Store max similarity score for each URL
    url_to_hyde_map = {}

    for link in search_results:
        url_content = link.raw_content

        ## split into list of chunks 
        url_content_split = text_splitter.split_text(url_content)

        try: # generate embeddings for each chunk
            url_content_embed = np.array(embedding_cross_lang.embed_documents(url_content_split))

            # calculate similarity scores against all Hyde embeddings
            similarity_matrix = cosine_similarity(url_content_embed, hyde_list)

            # Find and store max similarity for this URL
            overall_max = round(float(similarity_matrix.max()), 2)
            url_to_hyde_map[link.link] = overall_max

        except Exception as e:
            raise RuntimeError("Error generating embeddings for text chunks {}".format(str(e)[:100]))        
            
    # now update the values in the list of objects         
    updated_collections = []    
    for link_obj in search_results:
        # Raises KeyError if link missing - exactly what you want
        score = url_to_hyde_map[link_obj.link]  
        updated_collection = link_obj.model_copy(update={"hyde_score": score})
        updated_collections.append(updated_collection)

    return updated_collections


# Tools are for external API, actions which requires LLM to decide parameters , sityation whcih required dynamic parameters
# Here we process data which already in a state, no external api, state modification, sequential processing 


# In[56]:


# Why: assigns each page a hyde_score, drops anything below RELEVANCE_THRESHOLD=0.45, and keeps the top TOP_K=30 so the expensive summary LLM only sees the most relevant content.
# Used by: graph node after HyDe generation (or skip path).

def filter_semantic_similarity_node(state: UnifiedResearchState) -> UnifiedResearchState:
    """Filter and sort search results by HyDE relevance score"""
    
    logger.info(f"[filter_semantic_similarity_node] Raw links: {len(state['search_results_raw'])}")

    # Calcualte hyde score
    list_collection = copy.deepcopy(state["search_results_kterms_filtered"])
    links_to_process = [ link for link in list_collection if link.scenario == state["scenario_selected"]  ]
    hyde_content_list = state.get("hyde_list", None)

    hyde_embeddings = embedding_cross_lang.embed_documents(hyde_content_list)
    hyde_content_list_embed = np.array(hyde_embeddings)
    # this return class, we need to extract link and hyde score
    updated_search_result = check_content_similarity(search_results=links_to_process , hyde_list = hyde_content_list_embed )

    # Assign Hide score
    url_hyde_map = {item.link: item.hyde_score for item in updated_search_result}

    ## now replace values in the original 
    for item in list_collection: 
        if item.link in url_hyde_map: 
           item.hyde_score = url_hyde_map.get(item.link)
    
    # Filter items with scores
    scored_items = [ link for link in list_collection if link.hyde_score is not None]
    
    # Sort by score (descending)
    sorted_items = sorted( scored_items,  key=lambda x: x.hyde_score, reverse=True)
    
   # threshold filtering
    threshold_filtered = [it for it in sorted_items if it.hyde_score >= HYDE_RELEVANCE_THRESHOLD]
    sm_filtered = threshold_filtered[:HYDE_TOP_K]
    logger.info(f"[filter_semantic_similarity_node] Scored: {len(scored_items)} | Above threshold ({HYDE_RELEVANCE_THRESHOLD}): {len(threshold_filtered)} | Returned (TOP_K={HYDE_TOP_K}): {len(sm_filtered)}")

    ## now we need to update collected links in QueryPerformance , search_query_performance: Annotated[List[QueryPerformance],  operator.add ]
    # state["search_query_performance"]

    # create dictionary with query_id and count of links
    # count all links by query_id (single pass)
    query_link_counts = {}
    for link_obj in sm_filtered:  # 30 iterations
        query_id = link_obj.query_id
        query_link_counts[query_id] = query_link_counts.get(query_id, 0) + 1

    # update QueryPerformance objects
    queries_collection = copy.deepcopy(state["search_query_performance"])
    for query_obj in queries_collection:  # 5 iterations
        query_id = query_obj.query_id
        query_obj.links_after_sm_filter = query_link_counts.get(query_id, 0)     

    if sm_filtered: # generate statistics for hyde
        scores = [item.hyde_score for item in sm_filtered]
        logger.info(f"[filter_semantic_similarity_node] Score range: {min(scores):.2f} - {max(scores):.2f} | Top 5: {[item.displayLink[:30] for item in sm_filtered[:5]]}")
    else:
        logger.warning("[filter_semantic_similarity_node] No articles passed relevance threshold!")
    
    logger.info("[filter_semantic_similarity_node] Completed")
    
    return {"search_results_sm_filter": sm_filtered, "search_query_performance":queries_collection}


# #### Generate Summary

# In[57]:


# Why: produces one structured ContentSummary (claim_type, severity, date, summary) per URL using gpt-4.1, returned as {url: summary} so the caller can map results back.
# Used by: generate_url_summary_node inside asyncio.gather.
async def generate_url_summary(
    raw_content: str, 
    llm_instance, 
    entity_name: str, 
    url: str
) -> Dict[str, Optional[ContentSummary]]:
    
    """Summarise one page into a ContentSummary. Never raises.

      Network errors, timeouts, 429 and 5xx are retried inside the OpenAI client (max_retries=3).
      This loop retries only bad answers, with the next limit from SUMMARY_TOKEN_LIMITS:
        - output cut off (LengthFinishReasonError),
        - invalid values in the answer (ValidationError),
        - empty summary.
      Any other error (wrong key, bad request, content filter, client retries exhausted) is not retried.

      Empty/whitespace input short-circuits to a Level_1 placeholder ContentSummary (no LLM call).

      Args:
          raw_content: Page text to summarise.
          llm_instance: ChatOpenAI — copied with max_tokens per attempt and wrapped with
              .with_structured_output(ContentSummary).
          entity_name: Substituted into the system prompt via url_summary_instructions.
          url: Used as the returned dict's key and in log lines.

      Returns:
          {url: ContentSummary} on success, {url: None} on failure — the page keeps summary=None,
          is counted by generate_url_summary_node and retried if another search round runs.
      """
    
    logger.debug(f"[generate_url_summary] Starting: {url[:60]}")

    # Input validation
    if not raw_content or not raw_content.strip():
        logger.warning(f"[generate_url_summary] No raw_content for {url[:60]}")
        return { url:ContentSummary(
                    summary=f"No content for {url}",
                    claim_type="other",
                    severity_level="Level_1",
                    date_published="Unknown"
                ) }
    
    system_message = url_summary_instructions.format(entity_name=entity_name)
    messages = [SystemMessage(content=system_message), HumanMessage(content=raw_content)]
    attempts = len(SUMMARY_TOKEN_LIMITS)

    # Network errors, timeouts, 429 and 5xx are retried inside the OpenAI client (max_retries=3).
    # This loop retries only bad answers: cut off (more tokens next time), invalid values, empty summary.
    for attempt, max_tokens in enumerate(SUMMARY_TOKEN_LIMITS, 1):
        structured_llm = (llm_instance
                          .model_copy(update={"max_tokens": max_tokens})
                          .with_structured_output(ContentSummary))
        try:
            response = await structured_llm.ainvoke(messages)
        except (LengthFinishReasonError, ValidationError) as e:
            reason = f"{type(e).__name__} (max_tokens={max_tokens})"
        except Exception as e:   # wrong key, bad request, content filter, client retries exhausted
            logger.error(f"[generate_url_summary] Failed, not retrying: {type(e).__name__}: {str(e)[:150]} | url: {url[:80]}{out_of_credits(e)}")
            return {url: None}
        else:
            if response is not None and response.summary and response.summary.strip():
                if attempt > 1:
                    logger.info(f"[generate_url_summary] Succeeded on attempt {attempt}/{attempts} | url: {url[:80]}")
                return {url: response}
            reason = "empty summary"
        logger.warning(f"[generate_url_summary] Attempt {attempt}/{attempts} failed: {reason} | url: {url[:80]}")

    logger.error(f"[generate_url_summary] No valid summary after {attempts} attempts | url: {url[:80]}")
    return {url: None}


# In[58]:


# Why: parallelises generate_url_summary in batches of 10 across all filtered URLs so total runtime scales with batch count, not URL count.
# Used by: graph node after filter_semantic_similarity_node.
async def generate_url_summary_node(state: UnifiedResearchState) -> dict:
    
    link_collection = state["search_results_sm_filter"]
    entity_name = state['query_components'].entity_name

    link_collection_to_process = [
            link for link in link_collection 
            if not link.summary or not link.summary.strip()
                  ]

    url_to_link_obj = {link.link: link for link in link_collection_to_process}

    logger.info(f"[generate_url_summary_node] Entity: {entity_name} | Processing {len(link_collection_to_process)} URLs")
    
    BATCH_SIZE = 10

    async def process_all_batches():
        all_outcomes = {}

        for i in range(0 , len(link_collection_to_process),BATCH_SIZE): 
            batch_links = link_collection_to_process[i:i + BATCH_SIZE]
            tasks = [
                        generate_url_summary(
                            raw_content=link.raw_content,
                            llm_instance=llm_url_content_summary,
                            entity_name=entity_name,
                            url=link.link
                        )
                        for link in batch_links
                    ]
            batch_results = await asyncio.gather(*tasks)    

            for result_dict in batch_results:
                all_outcomes.update(result_dict)
        
        return all_outcomes   
    
    all_outcomes = await process_all_batches()
    
    updated_collections = []
    for url, outcome in all_outcomes.items():
        if not isinstance(outcome, ContentSummary):   # None = page failed, counted below
            continue
        
        original_link = url_to_link_obj[url]
        updated_collection = original_link.model_copy(update={
            "summary": outcome.summary,
            "claim_type": outcome.claim_type,
            "severity_level": outcome.severity_level,
            "date_published": outcome.date_published
        })
        updated_collections.append(updated_collection)

    failed_urls = [url for url, outcome in all_outcomes.items() if outcome is None]
    logger.info(f"[generate_url_summary_node] Summarised {len(updated_collections)}/{len(all_outcomes)} pages")
    if failed_urls:
        logger.warning(f"[generate_url_summary_node] {len(failed_urls)} page(s) NOT summarised, retried if another search round runs: {[u[:60] for u in failed_urls]}")
    
    return {"search_results_sm_filter": updated_collections}


# #### Evidence Validation

# In[59]:


# Why: consolidates the per-URL summaries into a deduplicated list of EvidenceClaims, preserving the supporting_urls provenance for every claim.
# Used by: graph node after generate_url_summary_node, with a token-escalation retry loop.
def extract_evidence_claims_node(state:UnifiedResearchState) -> Dict: 
    
    """Extract claims from summaries and consolidate duplicate claims across sources.

      No summaries → returns [] without an LLM call.
      Network errors, timeouts, 429 and 5xx are retried inside the OpenAI client; this node retries
      only bad answers (cut off / invalid / empty) using CLAIMS_TOKEN_LIMITS. Any other error, or no
      valid answer after all attempts, raises.
      """
    
    logger.info(f"[extract_evidence_claims_node] Extracting claims | Raw links: {len(state['search_results_raw'])}")

    summaries_data = []

    if state.get("url_feedback"):
        modifier = state.get("url_feedback").content
        logger.info("[extract_evidence_claims_node] Retry attempt - incorporating feedback")
    else:    
        modifier = "No previous feedback - first attempt"

    for link_obj in state["search_results_sm_filter"]: 
        if link_obj.summary and link_obj.summary.strip(): 
            summaries_data.append(
                {
                "url": link_obj.link,
                "source": link_obj.displayLink,
                "summary": link_obj.summary[:],
                "claim_type": link_obj.claim_type,
                "date_published":link_obj.date_published,
                "severity_level": link_obj.severity_level
                }
            )

    if not summaries_data:
        logger.info("[extract_evidence_claims_node] No summaries, no LLM call | 0 claims")
        return {"evidence_claims": []}

    entity_name = state["query_components"].entity_name
    summaries_data_string = json.dumps(summaries_data, indent=2)

    extraction_prompt = extract_evidence_claims_prompt.format(
        summaries_data_string=summaries_data_string, 
        entity_name=entity_name,
        modifier=modifier
    )

    messages = [SystemMessage(content=f"""
                                You are a forensic Financial Crimes Analysts of company {entity_name}

                                Follow these constraints:
                                - Use only information contained in the user message; do not add outside knowledge.
                                - Do not guess missing details (dates, amounts, agencies).
                                - Produce consolidated, non-duplicative claims and include provenance (supporting URLs).
                                - Choose exactly one claim_type per claim, using the schema provided by the user.
                                - If nothing qualifies under these constraints, return an empty list.
                                
                                """), HumanMessage(content=extraction_prompt)]    
    
    # Network errors, timeouts, 429 and 5xx are retried inside the OpenAI client (max_retries=3).
    # This loop retries only bad answers: cut off (more tokens next time), invalid values, empty response.
    # Any other error, or no valid answer after all attempts, raises → the company is logged as FAILED
    # (a report without consolidated evidence could look clean).
    attempts = len(CLAIMS_TOKEN_LIMITS)
    for attempt, max_tokens in enumerate(CLAIMS_TOKEN_LIMITS, 1):
        structured_llm = (llm_agg_summaries
                          .model_copy(update={"max_tokens": max_tokens})
                          .with_structured_output(ClaimsFromSummaries))
        try:
            response = structured_llm.invoke(messages)
        except (LengthFinishReasonError, ValidationError) as e:
            reason = f"{type(e).__name__} (max_tokens={max_tokens})"
        except Exception as e:   # wrong key, bad request, content filter, client retries exhausted
            logger.error(f"[extract_evidence_claims_node] Consolidation failed, not retrying: {type(e).__name__}: {str(e)[:150]}{out_of_credits(e)}")
            raise
        else:
            if response is not None:
                if attempt > 1:
                    logger.info(f"[extract_evidence_claims_node] Succeeded on attempt {attempt}/{attempts}")
                logger.info(f"[extract_evidence_claims_node] Extracted {len(response.evidence_claims)} unique claims from {len(summaries_data)} summaries")
                logger.debug(f"[extract_evidence_claims_node] Claims: {response.evidence_claims}")
                return {"evidence_claims": response.evidence_claims}
            reason = "empty response"
        logger.warning(f"[extract_evidence_claims_node] Attempt {attempt}/{attempts} failed: {reason}")

    logger.error(f"[extract_evidence_claims_node] No valid consolidation after {attempts} attempts")
    raise RuntimeError(f"Evidence consolidation failed for {entity_name}: no valid answer after {attempts} attempts")


# In[60]:


# Why: checks that every URL cited in supporting_urls is actually present in the pool of extracted URLs, blocking the LLM from inventing sources.
# Used by: graph node after extract_evidence_claims_node; feeds back into the extract step until clean.
def url_verification_node(state: UnifiedResearchState):

    """Detect URLs cited in evidence_claims that don't exist in search_results_raw.

      Compares the set of `supporting_urls` across all evidence_claims to the set of
      URLs actually collected by the search/extraction nodes. Any URL in the former
      but not the latter is a hallucination by the consolidation LLM.

      On hallucinations: returns {"url_feedback": AIMessage(<corrective prompt>)} so
      the conditional edge `route_evidence_validation` loops back to
      extract_evidence_claims_node with explicit instructions to use only real URLs.

      If URLs are still invalid after that one retry (url_feedback already set), they are
      removed from the claims in code (claims left without a URL are dropped), logged as
      WARNING, and {"evidence_claims": cleaned, "url_feedback": None} routes "continue".

      On clean output: returns {"url_feedback": None} → routes "continue".
      """

    logger.info(f"[url_verification_node] Executing | Raw links: {len(state['search_results_raw'])}")

    url_in_evidence = []
    for evidence in state.get("evidence_claims", []):
        url_in_evidence.extend(evidence.supporting_urls)

    url_in_collection = [link.link for link in state.get("search_results_raw", [])]

    url_in_evidence_set = set(url_in_evidence)
    url_in_collection_set = set(url_in_collection)

    hallucinated = url_in_evidence_set - url_in_collection_set

    logger.info(f"[url_verification_node] Evidence URLs: {len(url_in_evidence)} | Collection URLs: {len(url_in_collection)} | Hallucinated: {len(hallucinated)}")

    if hallucinated and state.get("url_feedback") is not None:
        # Already retried once this round → remove the invalid URLs instead of looping again.
        cleaned = []
        for claim in state.get("evidence_claims", []):
            valid = [u for u in claim.supporting_urls if u in url_in_collection_set]
            if valid:
                cleaned.append(claim.model_copy(update={"supporting_urls": valid}))
        dropped = len(state.get("evidence_claims", [])) - len(cleaned)
        logger.warning(f"[url_verification_node] Still {len(hallucinated)} invalid URL(s) after 1 retry, removed them | claims dropped (no valid URL left): {dropped} | {sorted(hallucinated)}")
        return {"evidence_claims": cleaned, "url_feedback": None}

    if hallucinated:
        for url in hallucinated:
            logger.warning(f"[url_verification_node] Hallucinated URL: {url}")
        
        hallucinated_urls_str = ", ".join(hallucinated)
        
        prompt = f"""You were given the task to consolidate evidence claims. However, your final conclusion 
referenced URLs which do not exist in the provided evidence collection.

**Hallucinated URLs ({len(hallucinated)}):**
{hallucinated_urls_str}

**Instructions:**
- ONLY use URLs that are present in the evidence collection
- Review each evidence claim and ensure all supporting_urls are valid
- Remove or replace any hallucinated URLs with actual URLs from the collection
- Maintain the same level of detail and quality in your evidence claims

Please repeat the consolidation process with strict focus on using ONLY the URLs that were actually provided in the search results."""
        
        ai_message = AIMessage(content=prompt)
        
        return {"url_feedback": ai_message}
    
    else:
        logger.info("[url_verification_node] No hallucinations detected")
        return {"url_feedback": None}
    

# Why: conditional-edge function that loops back to claim extraction when URL provenance is wrong, or moves on once everything verifies.
# Used by: builder.add_conditional_edges on url_verification_node.
def route_evidence_validation(state: UnifiedResearchState):
    """Route based on evidence feedback"""
    if state.get("url_feedback") is None:
        logger.info("[route_evidence_validation] ✓ Validation passed - proceeding")
        return "continue"
    else:
        logger.warning("[route_evidence_validation] ✗ Hallucinations detected - repeating consolidation")
        return "consolidate_evidence"


# In[61]:


## we need node which checks if we collected enough evidences
# Why: asks an LLM, framed as a cautious business partner, whether the collected evidence is strong enough to make a decision or whether another search loop is justified.
# Used by: graph node after url_verification_node.
def reflect_evidence_quality_node(state: UnifiedResearchState):

    """Decide whether the accumulated evidence is strong enough to stop searching.

      Frames the question to llm_agg_summaries as a cautious long-term customer of
      `entity_name` weighing whether to continue the business relationship given
      media findings on the configured `search_topics`. Evidence is grouped by
      claim_type, numbered, and inlined into the prompt.

      Uses structured output (AssessEvidenceQuality). If the call fails, defaults to
      repeat_search (logged as WARNING) so a routing decision never fails the company.

      Returns:
          {"evidence_feedback": AssessEvidenceQuality} where
          .evidence_quality ∈ {"convinced", "repeat_search"} and .reasoning is the LLM's
          justification — consumed by route_should_run_tool to decide loop vs. finalise.
    """

    logger.info(f"[reflect_evidence_quality_node] Executing | Raw links: {len(state['search_results_raw'])}")

    entity_name = state['query_components'].entity_name

    topics_raw = state.get("query_components").search_topics
    topics_list = []
    for topic in topics_raw: 
        for lang in topics_raw[topic]:
            if lang == "en":
              topics_list.extend(topics_raw[topic][lang])

    search_topics = " , ".join(topics_list)   

    types = {c.claim_type for c in state["evidence_claims"]}  # its a set 
    claims_by_type = {t: [] for t in types}
    logger.debug(f"[reflect_evidence_quality_node] Claim types: {sorted(types)}")

    ## now prepare input for final summarisation prompt 
    for claim in state["evidence_claims"]:
        claims_by_type[claim.claim_type].append({
            "text": claim.claim_text,
            "sources": ", ".join(claim.supporting_urls),
            "number of evidences": len(claim.supporting_urls), 
            "date_publish": claim.date_publish
        })

    sections = []
    for t in sorted(claims_by_type.keys()):
        items = claims_by_type[t]
        if not items:
            continue
        title = t.replace("_", " ").upper()
        
        # Add evidence numbering
        numbered_items = []
        for idx, item in enumerate(items, 1):
            numbered_items.append(f"EVIDENCE {idx}:\n{json.dumps(item, indent=2, ensure_ascii=False)}")
        
        sections.append(f"{title}:\n" + "\n\n".join(numbered_items))

    evidence_block = "\n\n".join(sections) if sections else "No evidence available."    
    logger.debug(f"[reflect_evidence_quality_node] Evidence block:\n{evidence_block}")

    system_prompt = f"""You are a long-time customer of {entity_name}. You have a business relationship with them that you value.
However , you are very cautions about company reputation as its directly imparct your business. 
Recently, you came across concerning information online suggesting {entity_name} may be connected to "{search_topics}".

This worries you because:
- Your reputation could be affected by association
- You need to know if this is credible or just rumors
- You want to make an informed decision about continuing the relationship

Below is the evidence you've gathered so far:

{evidence_block}

WHAT WOULD CONVINCE YOU:
- **Multiple credible sources** (official records preferred)
- **Specific details** (dates, amounts, agencies, jurisdictions)
- **Recent or ongoing matters** (active investigations, current sanctions)
- Evidence from authoritative sources (government agencies, regulators, established media)

WHAT WOULD MAKE YOU WANT MORE INFORMATION:
- Only one weak source
- Extremely vague claims with no details
- Sources appear unreliable or fabricated
- No information at all about the topic

Be honest: Does this evidence give you enough information to form a confident opinion about whether {entity_name} is truly involved in "{search_topics}", or do you need to dig deeper?

Return "convinced" if you have enough to make a decision (whether to stay or leave).
Return "repeat_search" if you're still uncertain and need more solid evidence."""

    structured_llm = llm_agg_summaries.with_structured_output(AssessEvidenceQuality)

    # Network errors, timeouts, 429 and 5xx are retried inside the OpenAI client (max_retries=3).
    # This is only a routing decision: if it fails, keep searching instead of failing the company.
    # The scenario pool is finite, so the loop still ends at exit_scenario.
    try:
        response = structured_llm.invoke([SystemMessage(content=system_prompt)])
    except Exception as e:
        logger.warning(f"[reflect_evidence_quality_node] Assessment failed, continuing with repeat_search: {type(e).__name__}: {str(e)[:150]}{out_of_credits(e)}")
        response = None
    if response is None:
        response = AssessEvidenceQuality(evidence_quality="repeat_search",
                                         reasoning="Assessment unavailable (LLM call failed), searching further by default.")
    
    logger.info(f"[reflect_evidence_quality_node] Assessment: {response.evidence_quality} | Reasoning: {response.reasoning}")
    logger.info("[reflect_evidence_quality_node] Done")
    
    return {"evidence_feedback":response }


# In[62]:


## condition node to try another tool 
# Why: conditional-edge function that ends the search loop and triggers the final report once the reflection LLM says "convinced".
# Used by: builder.add_conditional_edges on reflect_evidence_quality_node.
def route_should_run_tool(state:UnifiedResearchState):
    """Conditional-edge: end the loop on 'convinced', otherwise run another scenario.

      Reads `state["evidence_feedback"]` (set by reflect_evidence_quality_node).
      Returns "generate_risk_assessment_node" on 'convinced', else
      "scenario_selection_node" to pick the next scenario in the pool.

      A missing evidence_feedback (None) is treated as repeat_search.
      """
      
    evidence_feedback  = state.get("evidence_feedback", None)
    if evidence_feedback is not None and evidence_feedback.evidence_quality == 'convinced':
        return "generate_risk_assessment_node"
    else:
        return "scenario_selection_node"


# In[63]:


# Why: produces the final FinalReport (Romanian fields, 0-100 scor_risc) from the consolidated evidence claims so decision-makers get a structured verdict.
# Used by: terminal graph node before END.
def generate_risk_assessment_node(state: UnifiedResearchState):
    """Generate comprehensive AML risk assessment from evidence claims"""
    
    logger.info(f"[generate_risk_assessment_node] Executing | Raw links: {len(state['search_results_raw'])}")
    
    types = {c.claim_type for c in state["evidence_claims"]}
    claims_by_type = {t: [] for t in types}

    logger.debug(f"[generate_risk_assessment_node] Claim types: {sorted(types)}")
    
    ## now prepare input for final summarisation prompt 
    for claim in state["evidence_claims"]:
        claims_by_type[claim.claim_type].append({
            "text": claim.claim_text,
            "sources": ", ".join(claim.supporting_urls),
            "date_publish":claim.date_publish
        })
    
    entity_name = state["query_components"].entity_name

    idno = state.get("registration_number", "")

    summary_prompt = final_summary_prompt.format(
        current_date_alias = datetime.now().strftime('%Y-%m-%d') ,
        entity_name_alias = entity_name
    )
    
    system_message = SystemMessage(content=summary_prompt)
    
    # Prepare evidence for analysis
    # Build evidence_prompt dynamically (no hardcoded claim types)
    sections = []
    for t in sorted(claims_by_type.keys()):
        items = claims_by_type[t]
        if not items:
            continue
        title = t.replace("_", " ").upper()  # simple, generic prettifier
        sections.append(f"{title}:\n{json.dumps(items, indent=2, ensure_ascii=False)}")

    evidence_block = "\n\n".join(sections) if sections else "No evidence available."

    evidence_prompt = f"""Evidence for {entity_name} (IDNO {idno}):

            Public-source evidence:
            {evidence_block}

            Analyze this evidence according to the framework provided."""
    

    logger.debug(f"[generate_risk_assessment_node] Evidence prompt:\n{evidence_prompt}")
    logger.info("[generate_risk_assessment_node] Requesting final conclusion from LLM")
    
    messages = [system_message, HumanMessage(content=evidence_prompt)]

    # Network errors, timeouts, 429 and 5xx are retried inside the OpenAI client (max_retries=3).
    # This loop retries only bad answers: report cut off (more tokens next time), invalid values, empty response.
    # Any other error, or no valid report after all attempts, raises → the company is logged as FAILED.
    attempts = len(FINAL_REPORT_TOKEN_LIMITS)
    analysis_response = None
    for attempt, max_tokens in enumerate(FINAL_REPORT_TOKEN_LIMITS, 1):
        structured_llm = (llm_evaluation
                          .model_copy(update={"max_tokens": max_tokens})
                          .with_structured_output(FinalReport))
        try:
            analysis_response = structured_llm.invoke(messages)
        except (LengthFinishReasonError, ValidationError) as e:
            reason = f"{type(e).__name__} (max_tokens={max_tokens})"
        except Exception as e:   # wrong key, bad request, content filter, client retries exhausted
            logger.error(f"[generate_risk_assessment_node] Final report failed, not retrying: {type(e).__name__}: {str(e)[:150]}{out_of_credits(e)}")
            raise
        else:
            if analysis_response is not None:
                if attempt > 1:
                    logger.info(f"[generate_risk_assessment_node] Succeeded on attempt {attempt}/{attempts}")
                break
            reason = "empty response"
        logger.warning(f"[generate_risk_assessment_node] Attempt {attempt}/{attempts} failed: {reason}")

    if analysis_response is None:
        logger.error(f"[generate_risk_assessment_node] No valid final report after {attempts} attempts")
        raise RuntimeError(f"Final report failed for {entity_name}: no valid report after {attempts} attempts")

    # Mechanically build search-results list
    media_urls = sorted({
        url
        for claim in state["evidence_claims"]
        for url in claim.supporting_urls
    })
    logger.info(f"[generate_risk_assessment_node] Search sources: {len(media_urls)} media URLs")

    
    logger.info("[generate_risk_assessment_node] Risk assessment completed")
    
    return {"final_conclusion": analysis_response , "search_result_assets":media_urls}



# In[64]:


## We have to join data before assigning scores
# state is updates with data from both flows but without interaction
#def join_node(state: UnifiedResearchState):
#    """
#    Simple pass-through node that waits for both paths to complete
#    """
#    print("Both paths completed, proceeding to scoring...")
#    return state
# !! joining 2 states means duplicating data is part of information is shared
# it can cause error both thet we repeat same information, but it also required all data to have reducers
# The join node receives two separate state updates from the parallel paths, and when it tries to merge them, some keys have different values or duplicated data

# Why: empty pass-through node used purely as a join point between the parallel keyword-filter branch and the HyDe-generation branch, avoiding state-merge conflicts.
# Used by: graph node before route_expert_generation.
def buffer_node_hyde_generation(state: UnifiedResearchState):

    """
    Used to maintain split between deterministic and conditional nodes
    """
    #print("Total numner of raw links: " ,  len(state["search_results_raw"]) )
    
    return {} # dont use state or we will duplicate all data 

# we will isolate conditional logic i have iisue with nodes timing 


# #### Custom payload node

# In[65]:


# prepare input for the search Evaluator
# Why: serialises search_query_performance into a compact text block the lead-researcher LLM can read when deciding the next custom query.
# Used by: generate_custom_payload_execute_node prompt assembly.
def format_query_performance(query_performance: list) -> str:
    lines = []
    for q in query_performance:
        lines.append(
            f"[Query {q.query_id}] "
            f"Tool: {q.search_engine} | "
            f"Language: {q.query_lang} | "
            f"Scenario: {q.scenario} | "
            f"Links Initial: {q.links_initial} | "
            f"Links After Filter: {q.links_after_sm_filter} | "
            f"Query: {q.query_text[0]}..."
        )
    return "\n".join(lines)


# prepare input for the search Evaluator
# Why: serialises the current evidence_claims list into a compact text block so the lead researcher can see what gaps remain.
# Used by: generate_custom_payload_execute_node prompt assembly.
def format_evidence_claims(evidence_claims: list) -> str:
    lines = []
    for i, claim in enumerate(evidence_claims, 1):
        lines.append(
            f"[Claim {i}] "
            f"Type: {claim.claim_type} | "
            f"Date: {claim.date_publish} | "
            f"Text: {claim.claim_text}"
        )
    return "\n\n".join(lines)
	


# In[66]:


# Generate queries as possible direction
class ConductSearch(BaseModel):
    """Tool for defining the search direction."""
    reason_for_selection: str
    query: str

# node which runs qieries generated by supervisor 
# inlcudes URL collection and execution

# Why: alternative search node where a "lead researcher" LLM generates targeted Tavily queries based on what has been found, then executes them sequentially.
# Used by: graph node on the custom_scenario branch.
def generate_custom_payload_execute_node(state: UnifiedResearchState):
    """Generates Tavily search queries via reflection and executes them.

      If query generation fails (or returns no queries), the custom round is skipped (WARNING);
      evidence from earlier rounds is kept and the company continues.
      """
    logger.info(f"[custom_search_node] Scenario: {state.get('scenario_selected')} | Raw links so far: {len(state['search_results_raw'])}")

    # === GENERATE PAYLOAD ===
    lead_researcher_prompt_template = ChatPromptTemplate.from_messages([
        ("system", lead_researcher_prompt)
    ])

    # generate search queries
    chain = lead_researcher_prompt_template | llm_lead_researcher

    try:
        response = chain.invoke({
            "query_performance": format_query_performance(state["search_query_performance"]),
            "evidence_claims": format_evidence_claims(state["evidence_claims"])
        })

        if not response.tool_calls:
            logger.warning(f"[custom_search_node] No tool calls generated")
            raise ValueError("No tool_calls generated by lead researcher")
        
         # Cap queries to keep cost predictable
        if len(response.tool_calls) > CUSTOM_MAX_QUERIES:
            logger.info(f"[custom_search_node] Capping {len(response.tool_calls)} → {CUSTOM_MAX_QUERIES} queries")
            response.tool_calls = response.tool_calls[:CUSTOM_MAX_QUERIES]

        logger.info(f"[custom_search_node] Generated {len(response.tool_calls)} queries")

    except Exception as e:   # includes "no tool_calls" — this round is optional, earlier evidence is kept
        logger.warning(f"[custom_search_node] Payload generation failed, skipping custom searches: {type(e).__name__}: {str(e)[:150]}{out_of_credits(e)}")
        return {"scenario_used": [state["scenario_selected"]], "query_counter": state.get("query_counter", 0)}


    # Exucute search queries
    result = []
    link_collections = []
    query_states = []
    query_id_alias = state.get("query_counter", 0)

    urls_seen = set()
    if state["search_results_raw"]:
        for link in state["search_results_raw"]:
            urls_seen.add(link.link)

    # run sequentially 
    for tool_call in response.tool_calls:
        observation = tool_tavily_search.invoke({
            "query": tool_call["args"].get("query"),
            "max_results":CUSTOM_RESULTS_PER_QUERY
        })

        if not observation:
            logger.warning(f"[custom_search_node] Tavily returned 0 results for: {tool_call['args'].get('query', '')[:80]}")
    # collect results
        result.append(ToolMessage(
            content=str(observation),
            tool_call_id=tool_call["id"],
            name=tool_call["name"]
        ))

        query_id_alias += 1

        query_state = QueryPerformance(
            query_text=[tool_call["args"].get("query")],
            query_lang=None,
            query_id=query_id_alias,
            search_engine="tool_tavily_search",
            scenario=state["scenario_selected"],
            links_initial=len(observation) if observation else 0
        )
        query_states.append(query_state)

        for item in observation:
            if not isinstance(item, dict):
                logger.error(f"[custom_search_node] Expected dict, got {type(item)}")
                continue

            required_keys = ['displayLink', 'link', 'search_engine']
            missing_keys = [key for key in required_keys if key not in item]
            if missing_keys:
                logger.error(f"[custom_search_node] Missing keys: {missing_keys}")
                continue

            if item['link'] in urls_seen:
                logger.debug(f"[custom_search_node] Duplicate skipped: {item['link']}")
                continue

            display = extract_domain(item['link'])
            if display in Config.LOW_VALUE_DOMAINS:
                logger.debug(f"[custom_search_node] Low-value domain skipped: {display}")
                continue

            urls_seen.add(item['link'])

            link_collections.append(LinkCollection(
                displayLink=item['displayLink'],
                link=item['link'],
                search_engine=item["search_engine"],
                scenario=state["scenario_selected"],
                query_id=query_id_alias
            ))

        time.sleep(1)

    logger.info(f"[custom_search_node] Collected {len(link_collections)} new unique links")

    return {
        "messages": result,
        "search_results_raw": link_collections,
        "scenario_used": [state["scenario_selected"]],
        "search_query_performance": query_states,
        "query_counter": query_id_alias
    }
    
# vs 
#tool_execute_search_node
#def tool_execute_search_node(state: UnifiedResearchState):

#return {"messages": result ,  
#            "search_results_raw":link_collections, 
#            "scenario_used":[state["scenario_selected"]],
#            "search_query_performance":query_states,
#            "query_counter":query_id_alias
#              }   



# #### Assemble Graph

# In[67]:


## add node to the workflow 

# assemble agent
builder = StateGraph(UnifiedResearchState)

# content extraction
builder.add_node("scenario_selection_node", scenario_selection_node)
builder.add_node("search_payload_generate_node", search_payload_generate_node)
builder.add_node("tool_execute_search_node", tool_execute_search_node)
builder.add_node("extract_content_node", extract_content_node)

# custom node 
builder.add_node("generate_custom_payload_execute_node",generate_custom_payload_execute_node)

# content semantic filtering 
builder.add_node("filter_key_terms_node", filter_key_terms_node)
builder.add_node("filter_semantic_similarity_node", filter_semantic_similarity_node)
builder.add_node("generate_url_summary_node", generate_url_summary_node)

# evidence generation and conclusion
builder.add_node("generate_risk_assessment_node", generate_risk_assessment_node)
builder.add_node("extract_evidence_claims_node", extract_evidence_claims_node)

# evidence quality assesment 
builder.add_node("url_verification_node", url_verification_node)
builder.add_node("reflect_evidence_quality_node", reflect_evidence_quality_node)

# hyde generation
builder.add_node("call_subgraph", call_subgraph)
builder.add_node("buffer_node_hyde_generation",buffer_node_hyde_generation)



##########  logic 
# content extraction
builder.add_edge(START, "scenario_selection_node") # intiate process

builder.add_conditional_edges(
    source="scenario_selection_node", 
    path=route_trigger_search, 
    path_map={
        "END": END, 
        "search_payload_generate_node": "search_payload_generate_node",
        "generate_custom_payload_execute_node": "generate_custom_payload_execute_node",
        "generate_risk_assessment_node": "generate_risk_assessment_node",   # ← add this
    }
)


# what if tools are avaible and payload generated
# we can either generate hyde or skip it
builder.add_edge("search_payload_generate_node", "tool_execute_search_node")

builder.add_edge("generate_custom_payload_execute_node", "extract_content_node")
builder.add_edge("tool_execute_search_node", "extract_content_node")
builder.add_edge("extract_content_node", "filter_key_terms_node")

builder.add_edge("filter_key_terms_node", "buffer_node_hyde_generation") # buffer node is ised to isolate logic and narrow 2 directions
                                                                # enrich with hyde or skip it 

builder.add_conditional_edges(
    source="buffer_node_hyde_generation",
    path=route_expert_generation,
    path_map={
        "continue": "call_subgraph",
        "skip_hyde_generation": "filter_semantic_similarity_node"
    }
)

builder.add_edge("call_subgraph", "filter_semantic_similarity_node")

## genete summaries
builder.add_edge("filter_semantic_similarity_node","generate_url_summary_node")

# evidence validation loop
builder.add_edge("generate_url_summary_node", "extract_evidence_claims_node")
builder.add_edge("extract_evidence_claims_node", "url_verification_node")

builder.add_conditional_edges(
    source="url_verification_node", 
    path=route_evidence_validation, 
    path_map={
        "continue": "reflect_evidence_quality_node",
        "consolidate_evidence": "extract_evidence_claims_node"  # loops back
    }
)

builder.add_conditional_edges(
    source="reflect_evidence_quality_node", 
    path=route_should_run_tool, 
    path_map={
        "generate_risk_assessment_node": "generate_risk_assessment_node",
        "scenario_selection_node": "scenario_selection_node"  # loops back
    }
)


# final step
builder.add_edge("generate_risk_assessment_node", END)

# assemble agent
graph = builder.compile()


# #### Define main pipeline

# #### Test Multiple Cases

# In[ ]:


# ── Pipeline tuning constants — MODULE LEVEL (nodes read these as globals) ──
#NUM_RESULTS_PER_QUERY    = 5
MAX_JOURNALISTS          = 10 # please ignore 
RECURSION_LIMIT          = 50
HYDE_RELEVANCE_THRESHOLD = 0.45     # read by filter_semantic_similarity_node
HYDE_TOP_K               = 30       # read by filter_semantic_similarity_node
CUSTOM_MAX_QUERIES       = 5        # read by generate_custom_payload_execute_node
CUSTOM_RESULTS_PER_QUERY = 5        # read by generate_custom_payload_execute_node
SERPAPI_TIMEOUT          = 30       # read by google_search — seconds per request (library default 60000 s)
SERPAPI_MAX_ATTEMPTS     = 2        # read by google_search — one retry after a 5 s pause
SERPAPI_FATAL_ERRORS     = ("invalid api key", "run out of searches")  # read by google_search — never retried
TAVILY_MAX_ATTEMPTS      = 3        # read by tavily_search / tavily_content_extractor — pauses 5 s, then 10 s
TAVILY_EXTRACT_TIMEOUT   = 60       # read by tavily_content_extractor — seconds per batch (library default 30, max 120)
TAVILY_FATAL_ERRORS      = (InvalidAPIKeyError, ForbiddenError, BadRequestError, MissingAPIKeyError)  # never retried
SUMMARY_TOKEN_LIMITS     = [3500, 5000]   # read by generate_url_summary — 2nd attempt only if the answer was cut off / invalid
FINAL_REPORT_TOKEN_LIMITS = [6000, 10000]  # read by generate_risk_assessment_node — 2nd attempt only if the report was cut off / invalid
CLAIMS_TOKEN_LIMITS      = [8000, 16000]  # read by extract_evidence_claims_node — 2nd attempt only if the answer was cut off / invalid

scenario_pool = list(get_args(ScenarioPool))   # pure, stays




# In[ ]:


async def main(contragents, openai_url, openai_api_key, tavily_url, tavily_api_key, serp_api_key,NUM_RESULTS_PER_QUERY=10):
    # ── globals the GRAPH NODES read (must be set before graph.ainvoke) ──
    global key_vault
    global llm_with_tools, embedding_cross_lang, llm_url_content_summary
    global llm_agg_summaries, llm_evaluation, llm_lead_researcher

    setup_logging()
    logger.info("New batch run started")

    # 1. key_vault for the search tools (serp + tavily)
    key_vault = APIVault()
    key_vault.add_key("serp_google_key", serp_api_key)
    key_vault.add_key("tavily_key", tavily_api_key)
    key_vault.add_key("tavily_url", tavily_url or "https://api.tavily.com")   # APISIX gateway URL; None = direct to Tavily

    # 2. models + wire node globals
    models = Models(openai_url, openai_api_key)
    embedding_cross_lang    = models.embedding_cross_lang
    llm_url_content_summary = models.llm_url_content_summary
    llm_agg_summaries       = models.llm_agg_summaries
    llm_evaluation          = models.llm_evaluation
    llm_with_tools = models.llm_search_tool_payload.bind_tools(
        tools, parallel_tool_calls=True, max_tokens=2000)
    llm_lead_researcher = models.llm_lead_researcher.bind_tools([ConductSearch])   # client retries (max_retries=3) only
    tool_list_deployed = [e["function"]["name"] for e in llm_with_tools.kwargs["tools"]]

    # 4. inputs (rows passed in by the caller: [{"ID": ..., "IDENTIFYCODE": ..., "SNAME": ...}, ...])
    test_companies = []
    seen_idno = set()
    seen_id = set()
    for row in contragents:
        sname = str(row.get("SNAME") or "").strip()
        idno  = str(row.get("IDENTIFYCODE") or "").strip()
        if sname.lower() in ("", "nan", "none", "null") or idno.lower() in ("", "nan", "none", "null"):
            logger.warning(f"[BATCH] Skipped row with missing SNAME or IDNO: {row}")
            continue
        try:
            contragent_id = int(float(row.get("ID")))       # DWH contragent ID — not the IDNO
        except (TypeError, ValueError):                     # None, NaN, non-numeric
            logger.warning(f"[BATCH] Skipped row with missing contragent ID: {row}")
            continue
        if idno in seen_idno:
            logger.warning(f"[BATCH] Skipped duplicate IDNO {idno} ({sname})")
            continue
        if contragent_id in seen_id:                        # the output file is named by contragent ID
            logger.warning(f"[BATCH] Skipped duplicate contragent ID {contragent_id} ({sname}, IDNO={idno})")
            continue
        seen_idno.add(idno)
        seen_id.add(contragent_id)
        test_companies.append((sname, idno, contragent_id))
    if not test_companies:
        raise RuntimeError("No usable companies in contragents (each row needs SNAME and IDENTIFYCODE).")
    logger.info(f"[BATCH] Built test_companies: {len(test_companies)} of {len(contragents)} rows")

    # 5. wipe output dir
    if Config.AD_MEDIA_DIR.exists():
        shutil.rmtree(Config.AD_MEDIA_DIR)
    Config.AD_MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    # 6. batch loop  (unchanged except the marked lines)
    for company, idno, contragent_id in test_companies:
        logger.info("=" * 60)
        logger.info(f"[BATCH] Starting: {company} (IDNO={idno}, contragent ID={contragent_id})")
        logger.info("=" * 60)

        try:
            parts, name_re = generate_name_regex(
                entity_name=company, llm_instance=models.llm_names_variation)   # ← models.*
            qc = QueryComponentsCalc(company, DOMAIN_EXCLUDE, parts)
            qc.build_all()
            batch_input = qc.to_state().model_copy(update={"names_search_regexp": name_re.pattern})

            result = await graph.ainvoke(
                {
                    "messages": [],
                    "max_journalists": MAX_JOURNALISTS,
                    "journalists": [],
                    "hyde_list": [],
                    "num_results_alias": NUM_RESULTS_PER_QUERY,
                    "query_components": batch_input,
                    "registration_number": idno,
                    "tool_pool": tool_list_deployed,
                    "scenario_pool": scenario_pool,
                    "scenario_used": []
                },
                {"recursion_limit": RECURSION_LIMIT}
            )

            out_path = Config.AD_MEDIA_DIR / f"{contragent_id}.json"   # one file per contragent ID; company names can repeat

            final = result.get("final_conclusion")
            final_payload = final.model_dump() if hasattr(final, "model_dump") else final
            if final_payload is not None:                            # ← keep this None guard
                final_payload["rezultate_cautare"] = result.get("search_result_assets") or []

            row = {
                "contragentid": contragent_id,       # input ID (DWH contragent ID)
                "idno": idno,                        # input IDENTIFYCODE, from the same input row
                "arcdate": (date.today() - timedelta(days=1)).isoformat(),
                "runtime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "aml_report": final_payload,
            }

            tmp = out_path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(row, f, ensure_ascii=False, indent=2)
            tmp.replace(out_path)

            logger.info(f"[BATCH] Saved: {out_path}")

        except Exception as e:
            logger.exception(f"[BATCH] FAILED for {company}{out_of_credits(e)}")

def run_osint_agent(contragents, openai_url, openai_api_key, tavily_url, tavily_api_key, serp_api_key,NUM_RESULTS_PER_QUERY=10):
    """Sync entry point — what Airflow's PythonOperator calls. contragents = rows read from Oracle by the caller."""
    asyncio.run(main(contragents, openai_url, openai_api_key, tavily_url, tavily_api_key, serp_api_key,NUM_RESULTS_PER_QUERY=NUM_RESULTS_PER_QUERY))


if __name__ == "__main__":
    # Local CLI: secrets come from the environment (.env loaded at the top).
    # In Airflow, the DAG reads the rows and calls run_osint_agent() directly.

    # Load contragents from Oracle; if the DB is unreachable, WARN and fall back to 5 known test companies.
    try:
        engine = make_engine(
            os.getenv("ORACLE_SQL_USERNAME"),
            os.getenv("ORACLE_SQL_PASSWORD"),
            os.getenv("ORACLE_SQL_CONNECTION_STRING"),
            os.getenv("ORACLE_SQL_SERVICE_NAME"),
        )
        df_contragents = load_contragents(engine, Config.SOURCE_TABLE)
    except Exception as e:
        engine = None                                       # fallback test run: nothing is uploaded
        logger.warning(f"DB UNAVAILABLE ({type(e).__name__}: {e}) — FALLING BACK to a hardcoded sample of 5 test companies")
        df_contragents = pd.DataFrame([
            {"ID": 1, "IDENTIFYCODE": "1021600048015", "SNAME": "AIRROCK SOLUTIONS"},
            {"ID": 2, "IDENTIFYCODE": "1011600019764", "SNAME": "PRO IMOBIL GRUP"},
            {"ID": 3, "IDENTIFYCODE": "1004601004787", "SNAME": "BEMOL RETAIL"},
            {"ID": 4, "IDENTIFYCODE": "1004600068795", "SNAME": "AUTOFRAME-FM"},
            {"ID": 5, "IDENTIFYCODE": "1003600005654", "SNAME": "VITASANMAX"},
        ])

    run_osint_agent(
        contragents=df_contragents.to_dict("records"),
        openai_url=os.getenv("OPENAI_URL"),            # APISIX gateway URL; not set locally = direct to OpenAI
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        tavily_url=os.getenv("TAVILY_URL"),            # APISIX gateway URL; not set locally = direct to Tavily
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        serp_api_key=os.getenv("SERP_GOOGLE_API_KEY"),
        NUM_RESULTS_PER_QUERY = 5
    )

    # Upload the reports to Oracle; a fallback test run (no DB) is never uploaded.
    if engine is not None:
        engine.dispose()                               # the read is done; the upload opens its own connection
        upload_reports_to_oracle(
            reports_dir=Config.AD_MEDIA_DIR,
            ORACLE_SQL_USERNAME=os.getenv("ORACLE_SQL_USERNAME"),
            ORACLE_SQL_PASSWORD=os.getenv("ORACLE_SQL_PASSWORD"),
            ORACLE_SQL_CONNECTION_STRING=os.getenv("ORACLE_SQL_CONNECTION_STRING"),
            ORACLE_SQL_SERVICE_NAME=os.getenv("ORACLE_SQL_SERVICE_NAME"),
            table_target=Config.TARGET_TABLE,
        )
    else:
        logger.warning("Fallback test run: reports NOT uploaded to Oracle")


# 
