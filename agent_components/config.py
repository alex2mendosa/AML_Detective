from pathlib import Path
import json
import pandas as pd

_PROJECT_DIR = Path(__file__).resolve().parent.parent   # project root (one level above agent_components/)

class Config:

    # llm classification support 
    _datasets_cache = None
    _sanc_prog_cache = None
    _hyde_articles_cache = None

    # ── Paths ──
    PROJECT_DIR    = _PROJECT_DIR
    COMPONENTS_DIR = _PROJECT_DIR / "agent_components"
    OUTPUT_DIR     = _PROJECT_DIR / "output_sanctions"
    LOG_PATH       = _PROJECT_DIR / "sanctions_screening.log"
    SANC_PROG_PATH = _PROJECT_DIR / "agent_components" / "sanc_prog_dict.json"
    DATASETS_CSV   = _PROJECT_DIR / "agent_components" / "opensanctions-sources-2026-04-26.csv"

     # ── Pipeline 2 (deep research) paths ──
    SANCTIONS_DIR = _PROJECT_DIR / "output_sanctions"   # INPUT (= Pipeline 1 output)
    AD_MEDIA_DIR  = _PROJECT_DIR / "output_ad_media"    # OUTPUT
    RESEARCH_LOG  = _PROJECT_DIR / "aml_research.log"


    # ── Oracle source ──
    SOURCE_TABLE = "DM_NM.AR_NOTA_MONITORIZARE_PJ_DAILY_CONTRAGENT_INFO"

    # ── OpenSanctions API ──
    OS_MATCH_URL = "https://api.opensanctions.org/match/default"
    OS_BASE_URL  = "https://api.opensanctions.org"
    MATCH_PARAMS = {
        "algorithm": "logic-v2",
        "limit": 3,
        "threshold": 0.7,
        "changed_since": "2015-01-01",
    }

    # ── LLM (AML assessment) ──
    AML_MODEL = "gpt-4o"

    # to add description in final result and help agent make summary
    # https://followthemoney.tech/explorer/types/topic/
    # ── Risk topic descriptions (FollowTheMoney topics) ──
      # https://followthemoney.tech/explorer/types/topic/

    RISK_TOPICS = {
        "sanction": "Entity is directly designated by a government, prohibiting specific interactions.",
        "sanction.linked": "Entity has a direct relationship with a sanctioned entity, including indirect subsidiaries.",
        "sanction.counter": "Counter-sanctioned by non-democratic countries, may include activists or journalists.",
        "debarment": "Excluded from public procurement, often due to fraud in government contracts.",
        "export.control": "Subject to trade/export restrictions.",
        "export.control.linked": "Linked to an export-controlled entity.",
        "export.risk": "Flagged as a trade risk.",
        "invest.risk": "Flagged as an investment risk.",
        "crime": "General criminal activity.",
        "crime.fraud": "Fraud.",
        "crime.cyber": "Cybercrime.",
        "crime.fin": "Financial crime.",
        "crime.env": "Environmental violations.",
        "crime.theft": "Theft.",
        "crime.war": "War crimes.",
        "crime.boss": "Criminal leadership.",
        "crime.terror": "Terrorism.",
        "crime.traffick": "Trafficking.",
        "crime.traffick.drug": "Drug trafficking.",
        "crime.traffick.human": "Human trafficking.",
        "forced.labor": "Forced labor.",
        "asset.frozen": "Assets have been frozen by authorities.",
        "wanted": "Wanted by law enforcement.",
        "reg.action": "Subject to enforcement action by an industry regulator.",
        "reg.warn": "Placed on a warning/alert list by an industry regulator.",
        "corp.disqual": "Disqualified corporate entity.",
        "corp.shell": "Shell company.",
        "corp.offshore": "Offshore entity.",
    }

    # requred to avoid search among low reliability resources 

    LOW_VALUE_DOMAINS = {
                    # Social media
                    "https://www.youtube.com",
                    "https://www.instagram.com",
                    "https://www.facebook.com",
                    "https://www.tiktok.com",
                    "https://www.twitter.com",
                    "https://x.com",
                    "https://www.linkedin.com",
                    "https://t.me",                  # Telegram
                    "https://vk.com",                # Russian social network, very common in RU queries
                    "https://ok.ru",                 # Odnoklassniki, same

                    # Low-value content
                    "https://www.reddit.com",
                    "https://medium.com",
                    "https://www.quora.com",

                    # Encyclopedias (already blocked via -site: in query but just in case)
                    "https://www.wikidata.org",
                }

    EXCLUDE_DOMAINS = [
        "youtube.com", "instagram.com", "facebook.com", "tiktok.com",
        "twitter.com", "x.com", "linkedin.com", "t.me", "vk.com", "ok.ru",
        "reddit.com", "medium.com", "quora.com", "wikidata.org",
    ]



    @classmethod
    def sanctions_programs_map(cls):
          """OpenSanctions program-id → human title. Loaded once, cached."""
          if cls._sanc_prog_cache is None:
              with open(cls.SANC_PROG_PATH, encoding="utf-8") as f:
                  cls._sanc_prog_cache = {
                      item["key"]: item["title"] for item in json.load(f)["data"]
                  }
          return cls._sanc_prog_cache
    
    @classmethod
    def datasets_description_map(cls):
          """OpenSanctions dataset-id → title. Loaded once, cached."""
          if cls._datasets_cache is None:
              df = pd.read_csv(cls.DATASETS_CSV, encoding="utf-8", usecols=["title", "identifier"])
              cls._datasets_cache = dict(zip(df["identifier"], df["title"]))
          return cls._datasets_cache
    
    @classmethod
    def hyde_articles(cls):
          """Pre-baked HyDe reference articles by topic (raw, with <|company_name|>
          placeholder). Loaded once, cached. Substitute the entity name at the call site."""
          if cls._hyde_articles_cache is None:
              with open(cls.COMPONENTS_DIR / "agents_hyde_articles.json", encoding="utf-8") as f:
                  cls._hyde_articles_cache = json.load(f)
          return cls._hyde_articles_cache