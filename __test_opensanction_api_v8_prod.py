#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!jupyter nbconvert --to script __test_opensanction_api_v8_prod.ipynb 


# In[14]:


#!/usr/bin/env python
 # coding: utf-8

 # ──────────────────────────────────────────────────────────────
 # Dependencies — install with:
 #
 #   pip install requests pandas pydantic python-dotenv sqlalchemy oracledb openai langchain-openai langchain-core
 #
 # Notes:
 #   - oracledb         : Oracle thin-mode driver, no Oracle Instant Client needed
 #   - langchain-core   : pulled in transitively by langchain-openai, listed for clarity
 #   - openai           : used by preflight_check; langchain-openai also depends on it
 #
 # Local module (NOT installable via pip — copy the folder from this project):
 #   - agent_components/logger.py    provides get_logger()
 #   - agent_components/sanc_prog_dict.json
 #   - agent_components/opensanctions-sources-2026-04-26.csv
 #
 # Python: 3.10+ (for the `dict | None` type annotation syntax)
 # ──────────────────────────────────────────────────────────────


# In[15]:


# ──────────────────────────────────────────────────────────────
# External components & dependencies
# ──────────────────────────────────────────────────────────────
#
# EXTERNAL APIs
#   1. OpenSanctions API           https://api.opensanctions.org
#        - POST /match/default     entity match by registration number
#        - GET  /entities/{id}     fetch directors/owners for BO screening
#        - GET  /healthz           preflight reachability check
#        Auth: ApiKey header,  env var OPENSANC_API_KEY
#
#   2. OpenAI API                  https://api.openai.com
#        - gpt-4o                  AML risk verdict (RED/YELLOW/GREEN) via langchain
#        - models.list             preflight reachability check
#        Auth: Bearer,             env var OPENAI_API_KEY
#
# EXTERNAL DATABASE
#   3. Oracle DWH (MICB)           db1prodDWH.cs.mcb.md:1521 / service PWH
#        - Login schema:           MDWH
#        - Source table:           DM_NM.AR_NOTA_MONITORIZARE_PJ_DAILY_CONTRAGENT_INFO
#        - Columns read:           ARCDATE, ID, IDENTIFYCODE, SNAME
#        Auth: user/pass,          env vars ORACLE_SQL_USERNAME / ORACLE_SQL_PASSWORD
#                                          ORACLE_SQL_CONNECTION_STRING
#                                          ORACLE_SQL_SERVICE_NAME
#
# LOCAL FILES (must exist next to this script)
#   - .env                                    secrets, see env vars listed above
#   - agent_components/logger.py              shared get_logger() factory
#   - agent_components/sanc_prog_dict.json    OpenSanctions program-id → title map
#   - agent_components/opensanctions-sources-2026-04-26.csv
#                                             dataset-id → title map (refresh periodically)
#
# OUTPUTS WRITTEN BY THIS SCRIPT
#   - output_sanctions/{IDNO}.json            one screening result per contragent
#   - sanctions_screening.log                 append-mode log (see agent_components/logger.py)
#



# #### Set up Env

# In[37]:


import os
import json
import time
import requests
import pandas as pd
from pprint import pprint
from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
#from langchain_core.runnables import RunnableParallel, RunnableLambda
#from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from sqlalchemy import create_engine, text
from pathlib import Path

from .agent_components.logger import get_logger

from .agent_components.config import Config


# In[29]:


# Local/CLI convenience: load .env if present (no-op if absent). In Airflow the
 # secrets are passed as arguments to run_screening(), not read from .env.
load_dotenv(Config.PROJECT_DIR / ".env", override=True) #  builds the path to the .env file at your project root
#  It's purely a local-dev convenience — on an Airflow worker with no .env, it's a harmless no-op

logger = get_logger()


# #### connect to DB

# In[30]:


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

      engine = create_engine(
          f"oracle+oracledb://{ORACLE_SQL_USERNAME}:{ORACLE_SQL_PASSWORD}@{ORACLE_SQL_CONNECTION_STRING}/?service_name={ORACLE_SQL_SERVICE_NAME}",
          pool_size=pool_size,
          max_overflow=max_overflow,
      )
      return engine



# In[31]:


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


# In[26]:


## test apis
def preflight_check(opensanc_api_key, openai_api_key) -> bool:
    """Verify OpenSanctions /healthz and OpenAI /models are reachable. Returns True if both OK."""
    errors = []

    # OpenSanctions
    try:
        r = requests.get(
            "https://api.opensanctions.org/healthz",
            headers={"Authorization": f"ApiKey {opensanc_api_key}"},   # ← param
            timeout=5,
        )
        r.raise_for_status()
        logger.info("OpenSanctions API reachable")
    except Exception as e:
        errors.append(f"- OpenSanctions API failed: {e}")

    # OpenAI
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_api_key)                        # ← param
        client.models.list()
        logger.info("OpenAI API reachable")
    except Exception as e:
        errors.append(f"- OpenAI API failed: {e}")

    for err in errors:
        logger.error(err)

    return len(errors) == 0



# #### First HTTP request by company id

# In[33]:


## Find company with IDNO and Jurisdiction

def _request_with_retry(method: str, url: str, session: requests.Session,MAX_RETRIES = 5 , RETRY_DELAY = 5,  **kwargs):
      """
      HTTP wrapper with retry on transient failures.

      Retries on:
        - Network errors (ConnectionError, Timeout) — connection died before any response
        - 429 Too Many Requests — OpenSanctions rate-limits. NOT retrying 429 means
          transient throttling becomes a silent 'no match' downstream, which in turn
          becomes a false-negative AML verdict (safe_to_engage=True for an unscreened entity).
        - 5xx server errors — server-side faults, almost always transient

      Fails fast on:
        - All other 4xx (400, 401, 403, 404, ...) — client-side bugs, no point retrying

      Honors the server's `Retry-After` header when present, otherwise uses exponential
      backoff starting at RETRY_DELAY seconds.
      """
      delay = RETRY_DELAY
      for attempt in range(1, MAX_RETRIES + 1):
          try:
              response = session.request(method, url, timeout=30, **kwargs)
          except (requests.ConnectionError, requests.Timeout) as e:
              # No response was received — retry until the budget runs out.
              if attempt == MAX_RETRIES:
                  raise
              logger.warning(
                  f"{method} {url} network error (attempt {attempt}/{MAX_RETRIES}): {e}. "
                  f"Retrying in {delay}s"
              )
              time.sleep(delay)
              delay *= 2
              continue

          # Retry on rate-limit (429) and server faults (5xx).
          if response.status_code == 429 or 500 <= response.status_code < 600:
              if attempt == MAX_RETRIES:
                  response.raise_for_status()

              # If the server tells us how long to wait, respect it.
              # OpenSanctions returns Retry-After as integer seconds on 429.
              retry_after = response.headers.get("Retry-After")
              wait = int(retry_after) if retry_after and retry_after.isdigit() else delay

              logger.warning(
                  f"{method} {url} status {response.status_code} "
                  f"(attempt {attempt}/{MAX_RETRIES}). Retrying in {wait}s"
              )
              time.sleep(wait)
              delay *= 2
              continue

          # Any other 4xx — fail fast, retrying won't help.
          response.raise_for_status()
          return response


# #### Data Extraction steps

# In[ ]:


## ──────────────────────────────────────────────────────────────
## Step 1 — Match a company against OpenSanctions by registration number
## ──────────────────────────────────────────────────────────────

def match_company_by_registration(
    session: requests.Session,
    registration_number: str,
    jurisdiction: str,
) -> dict:
    """
    POST a single-query match request to OpenSanctions /match/default.

    Returns one of three shapes — the caller MUST distinguish them:
    1. Successful match:    {"results": [...], "total": {...}, ...}   normal payload
    2. Genuine 'no match':  {"results": [], "total": {"value": 0}, ...}
    3. API failure:         {"_api_error": "<error message>"}         sentinel

    The sentinel exists because production must NOT collapse 'API call failed'
    into 'no match found'. Both used to return None, which downstream became
    is_target=False → safe_to_engage=True — a silent false-negative AML verdict
    for an entity that was never actually screened.

    Why a sentinel key (`_api_error`) and not an exception:
    the pipeline is batch-oriented (loops over rows). One row's API failure
    should not abort the whole run; it should mark that row as un-screened
    and let the loop continue.
    """
    query_key = f"reg_{registration_number}"

    body = {
        "queries": {
            query_key: {
                "schema": "Company",
                "properties": {
                    "registrationNumber": [registration_number],
                    "jurisdiction": [jurisdiction],
                },
            }
        }
    }

    try:
        response = _request_with_retry("POST", Config.OS_MATCH_URL, session, json=body, params=Config.MATCH_PARAMS)
        # The /match endpoint returns {"responses": {<query_key>: {...}}}.
        # We unwrap to the single-query payload the rest of the pipeline expects.
        payload = response.json()["responses"][query_key]
        # A 200 without a "results" list is NOT a clean no-match — treat it as an API error,
        # otherwise it would fall through to GREEN / safe_to_engage=True.
        if not isinstance(payload.get("results"), list):
            raise ValueError("unexpected /match payload: no 'results' list")
        return payload
    except Exception as e:
        # Reached when _request_with_retry has exhausted its budget (transient failure that
        # survived all retries, or a non-retryable 4xx) or the response is not a valid /match payload.
        logger.error(f"Match request failed for reg={registration_number} jur={jurisdiction}: {e}")
        return {"_api_error": str(e)}




# In[ ]:


## ──────────────────────────────────────────────────────────────
## Step 2 — Flatten a single OpenSanctions match into LLM-ready shape
## ──────────────────────────────────────────────────────────────

def extract_for_llm(result: dict) -> dict:
    """
    Take ONE result dict from `results["results"][i]` (the OpenSanctions /match
    payload) and reshape it into the flat schema the rest of the pipeline uses.

    Pre-conditions (enforced by the caller, screen_company):
    - `result` is a real match, not None and not an API-error sentinel.
    - The caller passes `results["results"][0]` (the top match, since PARAMS has limit=3
        but we only act on the highest-scoring one).

    Three reshaping decisions worth flagging:

    1. Multi-valued props are unwrapped to scalar.
        OpenSanctions returns `registrationNumber`, `jurisdiction`, etc. as LISTS
        (an entity can legitimately have multiple). For our use case (one IDNO per
        row in DM_NM), we take the first value. The `(props.get(x) or [None])[0]`
        idiom handles three cases in one line: key missing, key present with []
        (empty list), key present with values.

    2. Topics are split into `risks` (known/described) vs `other_topics` (unknown).
        The LLM reasons better when it sees human-readable descriptions, not bare
        slugs like 'sanction.linked'. RISK_TOPICS supplies those descriptions.
        Unmapped topics still go into `other_topics` so nothing is silently dropped —
        this is the audit trail when OpenSanctions adds a new topic we haven't
        mapped yet.

    3. `sanctions_programs` and `datasets` become {id: title} dicts, not lists.
        Same reason as topics: the LLM (and the final JSON output) needs the human
        names ('EU Russia Sanctions', 'OFAC SDN List'), not opaque IDs.
        Unknown IDs fall back to a placeholder rather than being dropped —
        preserves traceability if our local maps fall out of sync with the API.
    """
    props = result.get("properties", {})
    topics = props.get("topics", [])

    risk_topics = Config.RISK_TOPICS                 # constant
    prog_map    = Config.sanctions_programs_map()    # lazy + cached
    dataset_map = Config.datasets_description_map()  # lazy + cached

    return {
          "entity_id": result.get("id"),
          "registration_number": (props.get("registrationNumber") or [None])[0],
          "entity_name": result.get("caption"),
          "match_score": result.get("score"),
          "is_target": result.get("target"),
          "jurisdiction": (props.get("jurisdiction") or [None])[0],

          "risks": {t: risk_topics[t] for t in topics if t in risk_topics},
          "other_topics": [t for t in topics if t not in risk_topics],

          "sanctions_programs": {
              prog_id: prog_map.get(prog_id, "Unknown Program")
              for prog_id in props.get("programId", [])
          },
          "datasets": {
              ds_id: dataset_map.get(ds_id, "Unknown Dataset")
              for ds_id in result.get("datasets", [])
          },

          "notes": props.get("notes", []),
          "last_seen": result.get("last_seen"),
      }

 


# In[ ]:


## ──────────────────────────────────────────────────────────────
## Step 3 — Fetch the full entity record (needed for relationships)
## ──────────────────────────────────────────────────────────────

def get_entity(session: requests.Session, entity_id: str) -> dict | None:
      """
      Fetch a single entity by ID from OpenSanctions /entities/{id}.

      Why this second call exists:
        The /match endpoint (Step 1) returns the matched entity's own properties
        but does NOT expand its relationships (directorshipOrganization,
        ownershipAsset, etc.). To enumerate directors and owners — which is what
        Step 4 (extract_people_from_entity) needs for the related-persons graph —
        we have to make a separate /entities/{id} call.

      Failure semantics — DIFFERENT from match_company_by_registration:
        Returns None on failure (not a sentinel dict).

        This is deliberate: an /entities failure is a PARTIAL failure of the
        overall screening, not a total failure.
          - We still have the /match result (entity is/isn't a sanctions target,
            its own risk topics, programs, datasets).
          - We just lose the related-persons graph (directors, owners).

        The downstream consumer (extract_people_from_entity) already guards against
        None and returns an empty people list. The final screening output is
        therefore degraded but not invalid — the primary entity verdict is intact,
        only the related-persons enrichment is missing.

      Caveat for production:
         Failure is surfaced upstream:
          `screen_company` checks for entity_data is None and sets
          `entity_fetch_error` on the screening result. `assess_aml_risk` then
          routes that result to ERROR / safe_to_engage=False (branch 4), so a
          BO-fetch failure NEVER produces a clean verdict.
      """
      try:
          response = _request_with_retry("GET", f"{Config.OS_BASE_URL}/entities/{entity_id}", session)
          return response.json()
      except Exception as e:
          # Reached only after _request_with_retry exhausts retries (5x with backoff
          # for network/429/5xx) or hits a non-retryable 4xx.
          # We log with entity_id so batch runs are debuggable from the log alone.
          logger.error(f"Entity fetch failed for entity_id={entity_id}: {e}")
          return None



# In[35]:


# Step 4 , Extract related people (directors + owners)
# Note: parent/child company tracing is not possible 
# the API does not expose cross-entity graph traversal
# ex. Lukoil RU does not point to Lukoil MD

def extract_people_from_entity(entity_data: dict) -> dict:
    """
      Walk a /entities response and produce a flat list of directors and owners,
      plus the subset that are sanctioned (is_target=True).

      Sources:
        - directorshipOrganization → directors (no schema filter applied)
        - ownershipAsset            → owners (filtered to schema Person/LegalEntity)

      Returns a dict with:
        - entity_id, entity_name : copied from the input
        - people                 : deduplicated list (by id) of all directors+owners
        - sanctioned_people      : subset where is_target=True

      Guards None input — returns the same shape with empty lists, because
      /entities failure is a partial failure of the screening, not a fatal one.
    """

    # Guard against None — get_entity returns None when all retries fail.
    # Without this, .get("properties") raises AttributeError and kills the whole pipeline.
    if not entity_data:
        return {"entity_id": None, "entity_name": None, "people": [], "sanctioned_people": []}

    props = entity_data.get("properties", {})
    people = []

    for directorship in props.get("directorshipOrganization", []):
        d_props = directorship.get("properties", {})
        for director in d_props.get("director", []):
            if isinstance(director, dict):
                dir_props = director.get("properties", {})
                people.append({
                    "source": "director",
                    "id": director.get("id"),
                    "name": director.get("caption"),
                    "role": d_props.get("role", []),
                    "start_date": d_props.get("startDate", []),
                    "end_date": d_props.get("endDate", []),
                    "status": "former" if d_props.get("endDate") else "current",
                    "is_target": director.get("target"),
                    "topics": dir_props.get("topics", []),
                    "sanctions_programs": dir_props.get("programId", []),
                })

    for ownership in props.get("ownershipAsset", []):
        o_props = ownership.get("properties", {})
        for owner in o_props.get("owner", []):
            if isinstance(owner, dict):
                owner_props = owner.get("properties", {})
                if owner.get("schema") in ("Person", "LegalEntity"):
                    people.append({
                        "source": "owner",
                        "id": owner.get("id"),
                        "name": owner.get("caption"),
                        "role": o_props.get("role", []),
                        "percentage": o_props.get("percentage", []),
                        "start_date": o_props.get("startDate", []),
                        "end_date": o_props.get("endDate", []),
                        "status": "former" if o_props.get("endDate") else "current",
                        "is_target": owner.get("target"),
                        "topics": owner_props.get("topics", []),
                        "sanctions_programs": owner_props.get("programId", []),
                    })

    seen = set()
    unique_people = []
    for p in people:
        if p["id"] not in seen:
            seen.add(p["id"])
            unique_people.append(p)

    return {
        "entity_id": entity_data.get("id"),
        "entity_name": entity_data.get("caption"),
        "people": unique_people,
        "sanctioned_people": [p for p in unique_people if p["is_target"]],
    }


# #### Main Pipeline

# In[36]:


# ──────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────

def screen_company(
      registration_number: str,
      jurisdiction: str,
      opensanc_api_key: str,
  ) -> dict | None:
    
    """
    Screen one company against OpenSanctions by registration number + jurisdiction.

    One API call per row:
    1. /match/default      — does the entity itself appear in any sanctions list?
    2. /entities/{id}      — DISABLED (commented out in Step 3+4 below). Was: fetch the
                                full entity to enumerate directors/owners for
                                beneficial-owner sanctions screening (FATF R.24 / NBM AML).
                                sanctioned_persons is still returned, always [].

    Occupation inference removed — was enrichment only, no compliance impact.

    Returns a flat dict with one of four outcomes (same shape in all cases):
        1. Successful match
        2. No match found
        3. Validation error
        4. API failure
            
    """

    # ── Input validation: skip the API call if inputs are missing ──
    reg = (registration_number or "").strip()
    jur = (jurisdiction or "").strip()
    # A NULL IDNO from the DB arrives as the string "nan"/"None" after astype(str) — reject it.
    if not reg or not jur or reg.lower() in ("nan", "none", "null"):
        error_msg = f"Invalid input: registration_number={registration_number!r}, jurisdiction={jurisdiction!r}"
        logger.warning(error_msg)
        return {
            "entity_id": None,
            "registration_number": registration_number,
            "entity_name": None,
            "match_score": None,
            "is_target": False,
            "jurisdiction": jurisdiction,
            "risks": {},
            "other_topics": [],
            "sanctions_programs": {},
            "datasets": {},
            "notes": [],
            "last_seen": None,
            "total_matches": 0,
            "sanctioned_persons": [],
            "validation_error": error_msg,
        }

    session = requests.Session()
    session.headers["Authorization"] = f"ApiKey {opensanc_api_key}"

    try:
        # ── Step 1: Match by registration number ──
        results = match_company_by_registration(session, reg, jur)

        # API failure path — is_target=None distinguishes from genuine 'no match'
        if results and results.get("_api_error"):
            return {
                "entity_id": None,
                "registration_number": registration_number,
                "entity_name": None,
                "match_score": None,
                "is_target": None,
                "jurisdiction": jurisdiction,
                "risks": {},
                "other_topics": [],
                "sanctions_programs": {},
                "datasets": {},
                "notes": [],
                "last_seen": None,
                "total_matches": 0,
                "sanctioned_persons": [],
                "api_error": results["_api_error"],
            }

        # Genuine 'no match in OpenSanctions' path
        if not results or not results.get("results"):
            return {
                "entity_id": None,
                "registration_number": registration_number,
                "entity_name": None,
                "match_score": None,
                "is_target": False,
                "jurisdiction": jurisdiction,
                "risks": {},
                "other_topics": [],
                "sanctions_programs": {},
                "datasets": {},
                "notes": [],
                "last_seen": None,
                "total_matches": 0,
                "sanctioned_persons": [],
            }

        # ── Step 2: Reshape the top match into LLM-ready flat dict ──
        llm_data = extract_for_llm(results["results"][0])
        llm_data["total_matches"] = results.get("total", {}).get("value")

        # ── Step 3+4 DISABLED: second call /entities/{id} (directors/owners) ──
        # Only ONE OpenSanctions call per company (/match). Director/owner screening
        # is not performed. get_entity(), extract_people_from_entity() and the
        # entity_fetch_error branch in assess_aml_risk are kept — uncomment to re-enable.
        #
        # # /entities failure is non-fatal — extract_people_from_entity guards against None
        # # and returns an empty people list, so the primary entity verdict is preserved.
        # if llm_data.get("entity_id"):
        #     entity_data = get_entity(session, llm_data["entity_id"])
        # else:
        #     entity_data = None
        #
        # people_data = extract_people_from_entity(entity_data)
        #
        # # Flag /entities fetch failure separately from 'no people on file'.
        # # Without this, "we never fetched the ownership graph" looks identical
        # # to "this entity has no directors or owners".
        # if entity_data is None and llm_data.get("entity_id"):
        #     llm_data["entity_fetch_error"] = (
        #         f"Failed to fetch /entities/{llm_data['entity_id']} — "
        #         f"director/owner sanctions screening incomplete"
        #     )
        #
        # llm_data["sanctioned_persons"] = [
        #     {
        #         "name":   p["name"],
        #         "role":   (p["role"] or [None])[0],
        #         "source": p["source"],
        #     }
        #     for p in people_data.get("sanctioned_people", [])
        # ]

        # Field kept so the output shape stays identical across all outcomes.
        # Empty because the /entities lookup above is disabled — NOT a "no sanctioned persons" result.
        llm_data["sanctioned_persons"] = []

        return llm_data
    finally:
        session.close()


# #### Generate AML report based on Sanction Data

# In[38]:


class AMLAssessment(BaseModel):
    risk_level: Literal["RED", "YELLOW", "GREEN"] = Field(
        description="RED=prohibited, YELLOW=EDD required, GREEN=standard DD"
    )
    reason: str = Field(description="2-3 sentence justification citing specific risks/programs")
    sanction_conclusion: str = Field(description="3-5 sentence compliance conclusion")
    safe_to_engage: bool = Field(description="False for RED/YELLOW, True for GREEN")




def build_aml_chain(openai_api_key):
      """Build the AML assessment chain. Call ONCE per run, reuse across companies."""

      aml_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the AML/Compliance analysis engine for Moldindconbank.

        Assess the entity and produce a risk classification + conclusion.

        RISK LEVELS:
        - RED: Direct prohibitive risk. Engagement NOT permitted.
        (direct sanctions, terrorism, war crimes, asset freezes, trafficking, export controls on entity)
        - YELLOW: Requires enhanced due diligence before engagement.
        (sanction.linked, debarment only, reg warnings, shell/offshore, counter-sanctions, moderate scores)
        - GREEN: No significant risk. Standard DD sufficient.

        Rules:
        - RED → safe_to_engage=false, cite specific topics and programs
        - YELLOW → safe_to_engage=false, recommend specific EDD steps
        - GREEN → safe_to_engage=true
        - Reference entity name, risk topics, sanctions programs, datasets explicitly
        - Do NOT invent risks not present in the data"""),
            ("user", "Screening data:\n{data}")
        ])
      
      llm_aml = ChatOpenAI(
          model=Config.AML_MODEL,          # ← from Config
          temperature=0.1,
          timeout=60,
          max_retries=3,
          api_key=openai_api_key,          # ← passed in, not os.environ
      ).with_structured_output(AMLAssessment)
    
      return aml_prompt | llm_aml


def assess_aml_risk(screening_result: dict, aml_chain) -> dict:
    """
    Take screen_company() output, return it enriched with risk verdict fields.

    Branch order is load-bearing — each guard catches a specific shape from
    screen_company and the next branch assumes the previous one didn't fire.

        1. validation_error    → ERROR / unsafe   (bad input, never hit the API)
        2. api_error           → ERROR / unsafe   (API call failed; entity unscreened)
        3. is_target == False  → GREEN / safe     (authoritative clean no-match)
        4. entity_fetch_error  → ERROR / unsafe   (/entities failed; BO screening incomplete)
        5. entity_id missing   → ERROR / unsafe   (defensive guard; should be unreachable)
        6. is_target == True   → LLM verdict      (the actual sanctioned-entity path)

    NEVER mark safe_to_engage=True for any path that didn't get an authoritative
    'no match' from the API. False negatives in AML are audit findings.
    """
    now = datetime.now(timezone.utc).isoformat()

    # ── 1. Bad input — never called the API ──
    if screening_result.get("validation_error"):
        return {
            "risk_level": "ERROR",
            "safe_to_engage": False,
            "reason": screening_result["validation_error"],
            "sanction_conclusion": "Input failed validation, no screening performed.",
            "analysis_timestamp": now,
            **screening_result,
        }

    # ── 2. API failure — entity was NOT screened, do NOT fast-path to GREEN ──
    # This branch must come BEFORE the is_target check, because screen_company
    # sets is_target=None on API failure (None is falsy and would otherwise
    # collapse into the 'no match' GREEN path).
    if screening_result.get("api_error"):
        return {
            "risk_level": "ERROR",
            "safe_to_engage": False,
            "reason": f"Sanctions API call failed: {screening_result['api_error']}",
            "sanction_conclusion": (
                "Sanctions screening could not be completed due to an API failure. "
                "Engagement must NOT proceed on the basis of this result. "
                "Retry the screening, or perform manual verification before any decision."
            ),
            "analysis_timestamp": now,
            **screening_result,
        }

    # ── 3. Genuine 'no match in OpenSanctions' — safe fast-path ──
    # Note: is_target is explicitly False here (not None), because the api_error
    # branch above already filtered out None. So this is an authoritative 'clean'.
    if screening_result.get("is_target") is False:
        return {
            "risk_level": "GREEN",
            "safe_to_engage": True,
            "reason": "No target match found in sanctions databases.",
            "sanction_conclusion": (
                f"Entity '{screening_result.get('entity_name') or screening_result.get('registration_number')}' "
                f"did not match as a target in any sanctions database. Standard due diligence is sufficient."
            ),
            "analysis_timestamp": now,
            **screening_result,
        }
    
    # ── 4. BO-screening incomplete — /entities fetch failed, refuse to verdict ──
    # Without this, an entity that matched but had its director/owner graph
    # un-fetched would silently fall through to the LLM with sanctioned_persons=[],
    # producing a clean verdict on incomplete data.
    if screening_result.get("entity_fetch_error"):
      return {
          "risk_level": "ERROR",
          "safe_to_engage": False,
          "reason": "Beneficial-owner screening incomplete — /entities fetch failed.",
          "sanction_conclusion": (
              "The entity matched in /match but director/owner sanctions screening "
              "could not be completed. Engagement must NOT proceed until BO data is "
              "retrieved and re-screened."
          ),
          "analysis_timestamp": now,
          **screening_result,
      }

    # ── 5. Defensive guard — shouldn't reach here without entity_id ──
    # If branches 1-3 are intact this is unreachable. Kept as a safety net so
    # any future change to screen_company that yields is_target=True but
    # entity_id=None fails loudly rather than poisoning the LLM call.
    if screening_result.get("entity_id") is None:
        return {
            "risk_level": "ERROR",
            "safe_to_engage": False,
            "reason": "is_target=True but entity_id is missing — inconsistent screening result.",
            "sanction_conclusion": "Screening data is inconsistent. Manual verification required before engagement.",
            "analysis_timestamp": now,
            **screening_result,
        }

    # ── 6. Target match — full LLM-driven assessment ──
    assessment_input = {
        k: screening_result.get(k) for k in (
            "entity_name", "entity_id", "match_score", "is_target", "jurisdiction",
            "risks", "other_topics", "sanctions_programs", "datasets",
            "notes", "last_seen", "sanctioned_persons",
        )
    }

    try:
        result = aml_chain.invoke({"data": json.dumps(assessment_input, indent=2, ensure_ascii=False)})
    except Exception as e:
        # OpenAI/LangChain failure after retries — surface as ERROR rather than crashing
        # the batch loop. The /match data we already have is preserved in the output.
        logger.error(f"AML LLM assessment failed for entity_id={screening_result.get('entity_id')}: {e}")
        return {
            "risk_level": "ERROR",
            "safe_to_engage": False,
            "reason": f"AML reasoning step failed: {e}",
            "sanction_conclusion": (
                "Entity matched as a sanctions target but the AML reasoning step failed. "
                "Manual review of the screening data is required before engagement."
            ),
            "analysis_timestamp": now,
            **screening_result,
        }

    return {
        "risk_level": result.risk_level,
        "safe_to_engage": result.safe_to_engage,
        "reason": result.reason,
        "sanction_conclusion": result.sanction_conclusion,
        "analysis_timestamp": now,
        **screening_result,
    }


# In[16]:


# AIRROCK_SOLUTIONS 1021600048015  in sanctions 
# AUTOFRAME 1004600068795 not in sanctions
# BEMOL 1004601004787 not in sanctions
# PROIMOBIL 1015600008270 not in sanctions
# VITASANMAX 1003600005654 not in sanctions
# Lukoil 1027700035769 in sanctions 


# In[ ]:


#1 # Smoke test: mimics exactly what an Airflow PythonOperator would do —
#2 # import the module, then call run_screening() with arguments.
#3 # Uses an isolated output dir so it does NOT touch real output_sanctions/.
#4 import os
#5 import json11
#6 from pathlib import Path
#7 from __test_opensanction_api_v8_prod import run_screening
#8 from agent_components.config import Config

def run_screening(contragents, openai_api_key, opensanc_api_key,
                    jurisdiction="md", output_dir=None, clean_output=True):
      """
      Screen a batch of companies; write one JSON per company. Returns a summary.

      Args:
          contragents: list of dicts, each with "IDENTIFYCODE" + "SNAME" (opt. "ID").
                       Built by the caller (DAG / __main__) — Oracle is NOT touched here.
          openai_api_key, opensanc_api_key: secrets, supplied by the caller.
          jurisdiction: ISO code for the match query.
          output_dir:  defaults to Config.OUTPUT_DIR.
          clean_output: wipe existing *.json before the run.
      """
      # Validate secrets ONCE, before the loop (not per-company).
      missing = [n for n, v in {"OPENAI_API_KEY": openai_api_key,
                                "OPENSANC_API_KEY": opensanc_api_key}.items() if not v]
      if missing:
          raise RuntimeError(f"run_screening missing required keys: {missing}")

      out_dir = Path(output_dir) if output_dir else Config.OUTPUT_DIR
      out_dir.mkdir(parents=True, exist_ok=True)

      aml_chain = build_aml_chain(openai_api_key)        # built ONCE, reused

      if clean_output:
          for old_file in out_dir.glob("*.json"):
              old_file.unlink()
          logger.info(f"Cleaned {out_dir}")

      count = 0
      failed = 0
      for row in contragents:
          idno  = str(row["IDENTIFYCODE"]).strip()
          sname = str(row["SNAME"]).strip()
          logger.info(f"--- Screening {sname} (IDNO={idno}) ---")

          try:
              raw      = screen_company(idno, jurisdiction, opensanc_api_key)
              assessed = assess_aml_risk(raw, aml_chain)
          except Exception as e:
              # One bad record must not stop the batch — write an ERROR row and continue.
              failed += 1
              logger.error(f"{idno} → unexpected error, marked ERROR: {type(e).__name__}: {e}")
              assessed = {
                  "risk_level": "ERROR",
                  "safe_to_engage": False,
                  "reason": f"Unexpected screening error: {type(e).__name__}: {e}",
                  "sanction_conclusion": (
                      "Sanctions screening could not be completed due to an unexpected error. "
                      "Engagement must NOT proceed. Retry the screening, or perform manual verification."
                  ),
                  "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                  "entity_id": None,
                  "registration_number": idno,
                  "entity_name": None,
                  "match_score": None,
                  "is_target": None,
                  "jurisdiction": jurisdiction,
                  "risks": {},
                  "other_topics": [],
                  "sanctions_programs": {},
                  "datasets": {},
                  "notes": [],
                  "last_seen": None,
                  "total_matches": 0,
                  "sanctioned_persons": [],
                  "screening_error": f"{type(e).__name__}: {e}",
              }

          assessed["sname"] = sname                       # ← REQUIRED by Pipeline 2
          assessed["registration_number"] = idno          # ← REQUIRED by Pipeline 2

          filename = f"{idno or row.get('ID')}.json"
          with open(out_dir / filename, "w", encoding="utf-8") as f:
              json.dump(assessed, f, indent=2, ensure_ascii=False, default=str)

          count += 1
          logger.info(
              f"{idno} → risk={assessed.get('risk_level')} "
              f"safe={assessed.get('safe_to_engage')} "
              f"entity={assessed.get('entity_name')!r}"
          )

      logger.info(f"Done. Wrote {count} results to {out_dir} | unexpected errors: {failed}")
      return {"screened": count, "failed": failed, "output_dir": str(out_dir)}


# In[ ]:


if __name__ == "__main__":
      import sys

      # Local CLI: secrets come from the environment (.env loaded at top of file).
      # In Airflow, the DAG calls run_screening() directly with keys from Connections/Variables.
      openai_api_key   = os.getenv("OPENAI_API_KEY")
      opensanc_api_key = os.getenv("OPENSANC_API_KEY")

      if not preflight_check(opensanc_api_key, openai_api_key):
          logger.error("Preflight failed, aborting")
          sys.exit(1)

      # Load contragents from Oracle; fall back to a hardcoded sample if DB unreachable.
      try:
          engine = make_engine(
              os.getenv("ORACLE_SQL_USERNAME"),
              os.getenv("ORACLE_SQL_PASSWORD"),
              os.getenv("ORACLE_SQL_CONNECTION_STRING"),
              os.getenv("ORACLE_SQL_SERVICE_NAME"),
          )
          df_contragents = load_contragents(engine, Config.SOURCE_TABLE)
          logger.info(f"Loaded {len(df_contragents)} rows from {Config.SOURCE_TABLE}")
      except Exception as e:
          logger.warning(f"DB unavailable ({e}); using hardcoded sample of 6 known IDNOs")
          df_contragents = pd.DataFrame([
              {"ID": 1, "IDENTIFYCODE": "1021600048015", "SNAME": "AIRROCK_SOLUTIONS"},
              {"ID": 2, "IDENTIFYCODE": "1004600068795", "SNAME": "AUTOFRAME"},
              {"ID": 3, "IDENTIFYCODE": "1004601004787", "SNAME": "BEMOL"},
              {"ID": 4, "IDENTIFYCODE": "1015600008270", "SNAME": "PROIMOBIL"},
              {"ID": 5, "IDENTIFYCODE": "1003600005654", "SNAME": "VITASANMAX"},
              {"ID": 6, "IDENTIFYCODE": "1027700035769", "SNAME": "LUKOIL"},
          ])

      run_screening(
          contragents     = df_contragents.to_dict("records"),
          openai_api_key  = openai_api_key,
          opensanc_api_key= opensanc_api_key,
          jurisdiction    = "md",
      )

