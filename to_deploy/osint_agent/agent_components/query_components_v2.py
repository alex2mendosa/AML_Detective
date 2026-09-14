from .states_v2 import QueryComponentsInState, DOMAIN_EXCLUDE   # relative: works in the notebook and when Airflow imports osint_agent as a package
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import copy


###################
# QueryComponentsCalc
###################

class QueryComponentsCalc:
    """
    Calculates and manages query components for Google search across multiple languages and topics.
    
    Generates name variations, search modifiers, and topic-specific queries for AML screening.
    """

    def __init__(self, entity_name: str, DOMAIN_EXCLUDE: List[str],entity_names_variations:List[str]):
        """
        Initialize query components calculator.
        
        Args:
            entity_name: Company or person name to generate queries for
            DOMAIN_EXCLUDE: List of domain names to exclude from search results
        """


        ##### SHARED ATTRIBUTES
        self.entity_name = entity_name
        self.DOMAIN_EXCLUDE = DOMAIN_EXCLUDE
        self.entity_names_variations = entity_names_variations

        # support for query generation for each instrument
        self.search_topics = {

            # Financial crime — money flows, asset misappropriation
            "financial": {
                "en": ["money laundering", "fraud", "tax evasion", "embezzlement", "financial crime", "crime"],
                "ro": ["spălare de bani", "fraudă", "evaziune fiscală", "delapidare", "infracțiune financiară"],
                "ru": ["отмывание денег", "мошенничество", "уклонение от налогов", "хищение", "финансовое преступление"],
            },
            
            # Corruption — abuse of power, illicit payments
            "corruption": {
                "en": ["corruption", "bribery", "influence peddling", "kickback", "illicit enrichment"],
                "ro": ["corupție", "mită", "trafic de influență", "îmbogățire ilicită", "abuz în serviciu"],
                "ru": ["коррупция", "взятка", "злоупотребление влиянием", "откат", "незаконное обогащение"],
            },
            
            # Organised crime — criminal networks, smuggling operations
            "organized_crime": {
                "en": ["organized crime", "mafia", "cartel", "smuggling", "trafficking"],
                "ro": ["crimă organizată", "mafie", "cartel", "contrabandă", "trafic"],
                "ru": ["организованная преступность", "мафия", "картель", "контрабанда", "рэкет"],
            },

            # Sanctions & regulatory actions — formal enforcement by authorities
            "sanctions": {
                "en": ["sanctions", "blacklist", "OFAC", "penalty", "regulatory action", "license revoked"],
                "ro": ["sancțiuni", "listă neagră", "penalitate", "acțiune regulatorie", "licență revocată"],
                "ru": ["санкции", "чёрный список", "штраф", "регуляторные меры", "отзыв лицензии"],
            },

            # Legal proceedings — court cases, arrests, convictions
            "legal": {
                "en": ["lawsuit", "indictment", "arrest", "conviction", "criminal charges", "investigation"],
                "ro": ["dosar penal", "arest", "condamnare", "acuzații penale", "anchetă", "judecată"],
                "ru": ["судебный иск", "обвинительное заключение", "арест", "осуждение", "уголовное дело"],
            },

            "reputational": {
                "en": ["scandal", "controversy", "misconduct", "whistleblower", "adverse media", "negative news"],
                "ro": ["scandal", "controversă", "conduită necorespunzătoare", "avertizor", "presă negativă"],
                "ru": ["скандал", "спор", "нарушение", "информатор", "негативные новости"],
            }
        }
        

        ############ TAVILY ATTRIBUTES , shared with perplexity
        self.tavily_exclude_domains = DOMAIN_EXCLUDE
        self.tavily_search_queries = None
        self.tavily_modified_queries =  None 

        self.tavily_query_templates = {

            # Financial crime — money flows, asset misappropriation
            "financial": {
                "en": "Find money laundering, fraud, tax evasion, or embezzlement allegations against {entity_name}",
                "ro": "Găsește acuzații de spălare de bani, fraudă, evaziune fiscală sau delapidare împotriva {entity_name}",
                "ru": "Найти обвинения в отмывании денег, мошенничестве, уклонении от налогов против {entity_name}"
            },

            # Corruption — abuse of power, illicit payments
            "corruption": {
                "en": "Find corruption, bribery, influence peddling allegations against {entity_name}",
                "ro": "Găsește acuzații de corupție, mită, trafic de influență împotriva {entity_name}",
                "ru": "Найти обвинения в коррупции, взяточничестве против {entity_name}"
            },

            # Organised crime — criminal networks, smuggling operations
            "organized_crime": {
                "en": "Find organized crime, smuggling, trafficking allegations against {entity_name}",
                "ro": "Găsește acuzații de crimă organizată, contrabandă împotriva {entity_name}",
                "ru": "Найти обвинения в организованной преступности, контрабанде против {entity_name}"
            },

            # Sanctions & regulatory actions — formal enforcement by authorities
            "sanctions": {
                "en": "Find sanctions, blacklist, OFAC penalties, or regulatory actions against {entity_name}",
                "ro": "Găsește sancțiuni, liste negre, penalități sau acțiuni regulatorii împotriva {entity_name}",
                "ru": "Найти санкции, чёрные списки, штрафы или регуляторные меры против {entity_name}"
            },

            # Legal proceedings — court cases, arrests, convictions
            "legal": {
                "en": "Find lawsuits, indictments, arrests, or criminal charges against {entity_name}",
                "ro": "Găsește dosare penale, aresturi, condamnări sau acuzații penale împotriva {entity_name}",
                "ru": "Найти судебные иски, обвинительные заключения, аресты или уголовные дела против {entity_name}"
            },

            "reputational": {
                "en": "Find scandals, controversies, misconduct, or adverse media coverage about {entity_name}",
                "ro": "Găsește scandaluri, controverse, conduită necorespunzătoare sau presă negativă despre {entity_name}",
                "ru": "Найти скандалы, споры, нарушения или негативные новости о {entity_name}"
            }
        }

        ############ GOOGLE ATTRIBUTES
        
        # placeholders google 
        self.google_search_modifier_entity = None   # this exact words or phrase:
        self.google_search_modifier_exc_domain = None
        self.google_search_modifier_topics = None # any of this words
        self.google_search_queries = None # any of this words



    ####### GENERAL METHODS
    def build_all( self ) -> None: 
        """Generate all query components (your current workflow)"""

        self.build_modifiers()
        self.google_generate_search_queries()
        self.tavily_generate_search_queries()

    def to_state(self) -> QueryComponentsInState:
        """Convert builder attributes to state model"""
        
        # Organize queries by search engine
        search_queries = {
            "google": self.google_search_queries,
            "tavily": self.tavily_search_queries
        }
        
        return QueryComponentsInState(
            entity_name=self.entity_name,
            DOMAIN_EXCLUDE=self.DOMAIN_EXCLUDE,
            search_queries=search_queries,
            search_topics=self.search_topics,
            entity_names_variations=self.entity_names_variations,
            google_search_modifier_entity=self.google_search_modifier_entity,
            google_search_modifier_exc_domain=self.google_search_modifier_exc_domain,
            google_search_modifier_topics=self.google_search_modifier_topics,
            names_search_regexp=None  # populated externally after generate_name_regex()
        ) 

    ############ TAVILY METHODS 
       
    def tavily_generate_search_queries(self): 
        """
        Using prompt template, add company name instead of placeholder
        """
        buffer = {}

        for topic in self.tavily_query_templates:
            buffer[topic] = {}

            for lang in self.tavily_query_templates[topic]:
                template = self.tavily_query_templates[topic][lang]
                buffer[topic][lang] = template.format(entity_name=self.entity_name)
                
        self.tavily_search_queries = buffer


    ############ GOOGLE METHODS

    # using modifiers and names variation generate 1 query per language per topic
    def google_generate_search_queries(self):
        """
        Generate complete Google search queries for all topic and language combinations.
        
        Combines entity variations, topic keywords, and domain exclusions into
        ready-to-use Google search query strings.
        
        Returns:
            None. Sets instance attribute:
                - self.google_search_queries: Dict[topic][lang] = query_string
                
        Raises:
            RuntimeError: If build_modifiers() hasn't been called yet
        """
        
        # Check if all required modifiers exist
        if not self.google_search_modifier_entity:
            raise RuntimeError("Search modifiers not ready. Call build_modifiers() first.")
        
        # Build queries for all topic/language combinations
        self.google_search_queries = {}
        
        for topic in self.google_search_modifier_topics:
            self.google_search_queries[topic] = {}
            
            for lang in self.google_search_modifier_topics[topic]:
                keyword_clause = self.google_search_modifier_topics[topic][lang]
                
                # Combine components (AND is implicit in Google, no need to write it)
                query = (
                    f'{self.google_search_modifier_entity} '
                    f'({keyword_clause}) '
                    f'{self.google_search_modifier_exc_domain}'
                )
                
                self.google_search_queries[topic][lang] = query.strip()         


    def build_modifiers(self):
        """
        Build Google search query modifiers from name variations and topics.
        
        Constructs three types of modifiers:
        1. Entity modifier: OR-combined quoted name variations
        2. Domain exclusion modifier: -site: filters for blocked domains
        3. Topic modifiers: OR-combined keywords for each topic and language
        
        Must be called after generate_name_regex() to ensure name variations exist.
        
        Returns:
            None. Sets instance attributes:
                - self.google_search_modifier_entity: Entity name OR clause
                - self.google_search_modifier_exc_domain: Domain exclusion string
                - self.google_search_modifier_topics: Dict of topic/language keyword strings
                
        Prints:
            Warning message if name variations haven't been generated yet
        """
        if self.entity_names_variations: 
                # intext was removed 
            self.google_search_modifier_entity = " OR ".join(
                 ['"' + elem + '"' for elem in self.entity_names_variations]
            )
        
            # Domain exclusions
            self.google_search_modifier_exc_domain = " ".join(
                [f"-site:{i}" for i in self.DOMAIN_EXCLUDE]
            )

            buffer = copy.deepcopy(self.search_topics)
            for topic in self.search_topics:
                for lang in self.search_topics[topic]:
                    search_terms = self.search_topics[topic][lang]
                    
                    # Add quotes around multi-word terms (terms containing spaces)
                    quoted_terms = []
                    for term in search_terms:
                        if " " in term:  # Multi-word term
                            quoted_terms.append(f'"{term}"')
                        else:  # Single word
                            quoted_terms.append(term)
                    
                    search_string = " OR ".join(quoted_terms)
                    buffer[topic][lang] = search_string

            self.google_search_modifier_topics = buffer

            return

        else:
            raise RuntimeError("Names variations not generated. Call generate_name_regex() first.")  
            return
        # raise stops execution and propagates error 


