###################
# Structured Outputs
###################

from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, List, Dict, Optional, Set, Literal , Optional
from langchain_core.messages import BaseMessage, AnyMessage, ToolMessage,HumanMessage, AIMessage, SystemMessage
from langgraph.graph import add_messages , START, END , StateGraph
from langchain_core.language_models import BaseChatModel

import operator
import re

## Support for journalists and HyDe articles 
class Journalist(BaseModel): # generates multiple values in single call
    expertise: str = Field(
        description="Primary area of expertise and writing focus."
    )
    perspective: str = Field(
        description="Writing perspective and approach to the topic."
    )
    style: str = Field(
        description="Writing style, tone, and target audience."
    )

class HydePerspectives(BaseModel):
      journalists: List[ Journalist ] = Field( "Comprehensive list of analysts with their roles and affiliations." )




# Evidence Lineage tracking 

# We have to guarantee
# Accountability: Every claim is traced to specific source
# Credibility: Check if the claim is supported by evidence

AllowedClaimType = Literal["allegation","investigation","charge","conviction","settlement","sanction_listing","other"]

# will store consolidated evidences into single ongect with supportive LLM
# additinally supporting_urls, each of them must be present in a pool of all url indicated across all LinkCollection 

class EvidenceClaim(BaseModel):
    """Track claims extracted from summaries - focused on consolidation"""
    claim_text: str = Field(description="Specific claim made (e.g., 'FinCEN fined company €50M in 2023')")
    claim_type: AllowedClaimType = Field(description="Each article must fit specific only 1 claim type") # should be generated during classification
    supporting_urls: List[str] = Field(description="ALL URLs that mention this claim")
    date_publish: str = Field(description=" Publication date or event date ")

class ClaimsFromSummaries( BaseModel ): 
    evidence_claims: List[EvidenceClaim] = Field(description="All extractable claims from the summaries")




# Data and Meta for extracted content, per single link


ToolsPool = Literal["google_search","perplexity_search","tavily_search","exit_scenario"]

class Tool_Selected(BaseModel):
       search_engine:ToolsPool = Field(default="" , description="Name of tool used to suggest url link") # taken from tool call tool.name

class LinkCollection(BaseModel): # organise for better visual inspection 
       
       displayLink: str = Field(description="The display URL shown in search results (usually domain name)") # passeed from search tool
       link: str = Field(description="The full URL of the search result") # passeed from search tool

       claim_type: str = Field(default="other", description="Each article must fit specific only 1 claim type") # from ContentSummary
       date_published: Optional[str] = Field( default=None, description="Publication date extracted from content (format: YYYY-MM-DD, or 'Unknown' if not found)" ) # from ContentSummary
       search_engine: str = Field(default="" , description="Name of tool used to suggest url link") # taken from tool call tool.name
       severity_level: str = Field(default="", description="Each article must fit specific only 1 severity level") # from ContentSummary
       summary: str = Field(default="", description="Summary of content extracted from URL") # from ContentSummary

       hyde_score: Optional[float] = Field(default=None, description="Max value of similarity between content of web page and HyDe articles") # from similarity score
                                           
       raw_content: str = Field(default="", description="Content extracted from URL") # passed from content parser                                    
                                           



#ContentSummary (claim_type: AllowedClaimType) 
#    ↓ [extract/convert]
#LinkCollection (claim_type: str)    



AllowedSeverityLevel = Literal["Level_1", "Level_2", "Level_3", "Level_4", "Level_5"]   
       
# during article summarizations we generate both summary and assign specific claim  type    
class ContentSummary(BaseModel):
    claim_type: AllowedClaimType = Field(
        description="Primary financial crime classification from content"
    )

    severity_level: AllowedSeverityLevel = Field(
        description="Select exactly one of: Level_1, Level_2, Level_3, Level_4, Level_5"
    )

    date_published: Optional[str] = Field( default=None,
         description="Publication date extracted from content (format: YYYY-MM-DD, or 'Unknown' if not found)" 
    )

    summary: str = Field(
        description="Detailed summary of financial crime/compliance information present in the content, or statement that no relevant information exists"
    )
    





# QueryComponentsCalc -> populates QueryComponentsinState in first node of the Investigation Process 
# QueryComponentsinState must store content which is reusable across different tools 


def generate_name_regex(company_name: str, llm_instance:BaseChatModel) -> tuple[ list[str], re.Pattern]:
    """
    Returns:
        parts: [Original, Russian]
        name_re: compiled regex matching either variant (case-insensitive)
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

    Original name: {company_name}
    """.strip()


    resp = llm_instance.invoke(prompt)
    pipe_line = (getattr(resp, "content", "") or "").strip()

    # Parse "Original | Russian", fallback to original if needed
    parts = [p.strip() for p in pipe_line.split("|")]

    # Compile strict, case-insensitive pattern
    pattern_str = r"(?:%s)" % "|".join(map(re.escape, parts))
    name_re = re.compile(pattern_str, re.IGNORECASE)

    return parts, name_re



# domain names
DOMAIN_EXCLUDE = [
    # Social Media & Forums
    "facebook.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "reddit.com",
    "quora.com",
    "pinterest.com",
    "tiktok.com",
    "snapchat.com",
    "telegram.org",
    "whatsapp.com",

    # Wikipedia & Wikis (multiple languages)
    "wikipedia.org",
    "en.wikipedia.org",
    "ru.wikipedia.org",  # Russian
    "ro.wikipedia.org",  # Romanian
    "fr.wikipedia.org",  # French
    "de.wikipedia.org",  # German
    "es.wikipedia.org",  # Spanish
    "it.wikipedia.org",  # Italian
    "uk.wikipedia.org",  # Ukrainian
    "pl.wikipedia.org",  # Polish
    "wikimedia.org",
    "wikidata.org",
    "wikihow.com",
    "wikia.com",
    "fandom.com",  # Wikia rebranded

    # Video & Entertainment
    "youtube.com"
]



class QueryComponentsinState(BaseModel):
      entity_name: str
      search_topic: str
      query_collection: Dict[str, str] = Field(
        default_factory=lambda: {"ro": "", "ru": "", "fr": "", "en": "", "de": ""}
    )
      entity_names_variations: List[str] = Field(default_factory=list)
      google_search_modifier_entity: str = ""
      google_search_modifier_exc_domain: str = ""
 

class QueryComponentsCalc():

    def __init__(self, entity_name:str, search_topic:str, llm_names_variation:BaseChatModel , generate_name_regex_f =  generate_name_regex   ):
        self.entity_name =  entity_name
        self.search_topic = search_topic
        self.query_collection = {"ro":"" , "ru":"", "fr":"" , "en":"" ,  "de":""} 

        #self.google_search_modifier = '(intitle:"{entity_name}" OR inurl:"{entity_name}") AND intext:"{entity_name}" '.format(entity_name=self.entity_name)
        # in text has the highest priority , append query with variations generated by generate_name_regex
        self.entity_names_variations = generate_name_regex_f(self.entity_name , llm_names_variation)[0] # list and reg expression, here we need only list
        self.google_search_modifier_entity = " OR ".join( [ "intext:" + '"'+ elem + '"' for elem in self.entity_names_variations] )
        self.google_search_modifier_exc_domain = " ".join([f"-site:{i}" for i in DOMAIN_EXCLUDE ])  


    def translate_query_for_search(self,llm_instance) -> str:
    
        """
        Translate search query to target language
        Args:
            llm_instance: LLM instance for translation
        Returns:
            Dictionary of translated queries
        """

        # Access attributes using self
        entity = self.entity_name
        topic = self.search_topic
        query_collection = self.query_collection
        # we will directly modify the attributes

        language_names = {
            "en": "English","ru": "Russian", "fr": "French","ro": "Romanian","de":"German"
        }

        query = self.entity_name + " " + self.search_topic
        for lang in language_names.keys():
            target_language = language_names[lang]

            translation_prompt = f"""Translate this search query to {target_language}. 
                                    Keep it concise and search-engine friendly.
                                    Keep company names in their original form (do not translate company names)
                                    Use semantically correct terms for fraud, corruption, and reputation-related concepts
                                    Only return the translated query, nothing else:
            Query: {query}

            Translation:"""
                
            try:
                response = llm_instance.invoke(translation_prompt)
                translated = response.content.strip() if hasattr(response, 'content') else str(response).strip()
                print(f"Translated '{query}' -> '{translated}' ({target_language})")
                query_collection[lang] = translated

            except Exception as e:
                print(f"Translation error: {e}. Using original query.")
                query_collection[lang] = query  # Fallback to original

        # explicitly return modified attribute
        return self.query_collection         
            





## We need to access quality of the evidences 

AllowedEvidenceAssessment = Literal["repeat_search", "convinced"]

class AssessEvidenceQuality(BaseModel):
    evidence_quality: AllowedEvidenceAssessment = Field(description="Whether evidence is convincing or search should be repeated")
    reasoning: str = Field(description="Brief, up to 50 words explanation of the assessment")






###################
# State Definitions
###################





# custom reducers

def custom_reducer(existing:list , new:list) -> list: 
    links  = [item.link for item in new]
    scores = [item.hyde_score for item in new]
    url_hyde_map = {link: score for link, score in zip(links, scores)}

    ## now replace values in the original 
    for item in existing: 
        if item.link in url_hyde_map: 
           item.hyde_score =  url_hyde_map.get(item.link)

    return existing
# it was created for search_results_kw_filter, to append data however same key receieve input from another node
# we can not implement 2 reducers or indicate from which node information comes from , therefore, assign hide score will be moved to 
# filter on semantics 


class UnifiedResearchState(TypedDict):
    
    # From UnifiedResearchState
    messages: Annotated[List[AnyMessage], add_messages]
    search_results_raw: Annotated[List[LinkCollection] , operator.add]
    search_results_kw_filter: Annotated[List[LinkCollection] , operator.add] # applied during content extraction, filter via regexp
    search_results_sm_filter: Annotated[List[LinkCollection] , operator.add] # filtering done with hyde 
    query_components:List[QueryComponentsinState]
          #entity_name: str
          #expanded_query:str
    num_requests_per_lang : int
    
    # From GenerateAnalystsState
    max_journalists: int
    journalists: List[Journalist]
    hyde_list: List[str]    

    # from evidence collection 
    evidence_claims: List[EvidenceClaim]

    # evidence_feedback
    url_feedback: Optional[AIMessage] # used for evaluation optimizer workflow
    evidence_feedback: Optional[AssessEvidenceQuality]

    # support MCP approach 
    search_engines_used : Annotated[List[str] , operator.add]
    search_engines_selected: str # can cause issue related to missing reducer
    search_engines_pool: List[str]

    final_conclusion:str


#Inside a node's return statement: When you return "search_results_raw": new_element, LangGraph will automatically apply operator.add 
#(which concatenates lists). So if new_element is a LinkCollection, it gets wrapped in a list and appended to the existing list.

#For manual testing outside nodes: The Annotated[List[LinkCollection], operator.add] annotation only tells LangGraph how to handle 
#updates when the graph processes node returns. When you're manually constructing or modifying the state dictionary outside of the 
#graph execution (like in tests or debugging), you need to manually manage the list yourself using .append() or list concatenation.











