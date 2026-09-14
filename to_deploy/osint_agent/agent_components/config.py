from pathlib import Path
import json

_PROJECT_DIR = Path(__file__).resolve().parent.parent   # project root (one level above agent_components/)

class Config:

    _hyde_articles_cache = None

    # ── Paths ──
    PROJECT_DIR    = _PROJECT_DIR
    COMPONENTS_DIR = _PROJECT_DIR / "agent_components"

     # ── Pipeline 2 (deep research) paths ──
    AD_MEDIA_DIR  = _PROJECT_DIR / "output_ad_media"    # OUTPUT
    RESEARCH_LOG  = _PROJECT_DIR / "aml_research.log"


    # ── Oracle source and target ──
    SOURCE_TABLE = "DM_NM.AR_NOTA_MONITORIZARE_PJ_DAILY_CONTRAGENT_INFO"   # contragents read from here
    TARGET_TABLE = "DM_NM.NOTA_MONITORIZARE_PJ_DAILY_OSINT_AGENT"          # reports uploaded here


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
    def hyde_articles(cls):
          """Pre-baked HyDe reference articles by topic (raw, with <|company_name|>
          placeholder). Loaded once, cached. Substitute the entity name at the call site."""
          if cls._hyde_articles_cache is None:
              with open(cls.COMPONENTS_DIR / "agents_hyde_articles.json", encoding="utf-8") as f:
                  cls._hyde_articles_cache = json.load(f)
          return cls._hyde_articles_cache