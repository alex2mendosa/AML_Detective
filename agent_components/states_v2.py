###################
# Structured Outputs
###################

from typing import TypedDict, Annotated, List, Dict, Optional, Set, Literal , Optional
from langchain_core.messages import BaseMessage, AnyMessage, ToolMessage,HumanMessage, AIMessage, SystemMessage
from langgraph.graph import add_messages , START, END , StateGraph
import copy 
from pydantic import BaseModel, Field

import operator


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

ScenarioPool = Literal["tool_google_search", "tool_tavily_search", "custom_scenario", "exit_scenario"]

class Scenario_Selected(BaseModel):
       search_engine:ScenarioPool = Field(default="" , description="Different scenarious for tool call or exit conditions") # taken from tool call tool.name

class LinkCollection(BaseModel):
    # Required fields (no default)
    displayLink: str = Field(description="The display URL shown in search results (usually domain name)")
    link: str = Field(description="The full URL of the search result")
    query_id: Optional[int] = Field(default=None, description="Unique id of query") 
    
    # Optional fields - MUST use Optional[str], not str with default=None
    claim_type: Optional[str] = Field(default=None, description="Each article must fit specific only 1 claim type")
    date_published: Optional[str] = Field(default=None, description="Publication date extracted from content (format: YYYY-MM-DD, or 'Unknown' if not found)")
    search_engine: Optional[str] = Field(default=None, description="Name of tool used to suggest url link")
    scenario: Optional[str] = Field(default=None, description = "Name of the search scenario" )
    severity_level: Optional[str] = Field(default=None, description="Each article must fit specific only 1 severity level")
    summary: Optional[str] = Field(default=None, description="Summary of content extracted from URL")
    hyde_score: Optional[float] = Field(default=None, description="Max value of similarity between content of web page and HyDe articles")
    raw_content: Optional[str] = Field(default=None, description="Content extracted from URL")                                                                  


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
    



# domain names
DOMAIN_EXCLUDE = [
    "facebook.com",
    "wikipedia.org",
    "wikimedia.org"
]

# current class store attributes usefull for initial stage of url collection 

class QueryComponentsInState(BaseModel):
    entity_name: str
    DOMAIN_EXCLUDE: List[str]
    search_queries: Dict[str,Dict[str,Dict[str,str]]]    # { google_queries : {topic: { lang:query } }  }
    search_topics: Dict[str, Dict[str, List[str]]]  # {topic: {lang: [keywords]}}
    entity_names_variations: List[str]
    google_search_modifier_entity: str
    google_search_modifier_exc_domain: str
    google_search_modifier_topics: Dict[str, Dict[str, str]]  # Not str!
    names_search_regexp: Optional[str] = None




## We need to access quality of the evidences 

AllowedEvidenceAssessment = Literal["repeat_search", "convinced"]

class AssessEvidenceQuality(BaseModel):
    evidence_quality: AllowedEvidenceAssessment = Field(description="Whether evidence is convincing or search should be repeated")
    reasoning: str = Field(description="Brief, up to 50 words explanation of the assessment")




###################
# State Definitions
###################


class FinalReport(BaseModel):
    """Final AML/Compliance Investigation Report"""
    
    rezumat_analiza: str = Field(
        ...,
        description="Executive summary of the entire investigation. Include: entity overview, total financial impact (fines/penalties), core compliance issues, geographic scope, and overall risk assessment. 3-5 paragraphs in Romanian."
    )
    
    scor_risc: int = Field(
        ...,
        ge=0,
        le=100,
        description="Risk score from 0-100. Use this scale: 0-25=Low Risk, 26-50=Medium Risk, 51-75=High Risk, 76-100=Critical Risk. Base on: violation severity, ongoing investigations, remediation quality, and timeline pattern."
    )
    
    analiza_suspiciuni: str = Field(
        ...,
        description="Detailed analysis of each violation/suspicion found. For each: type of violation, dates (ISO 8601), amounts involved, fines paid, investigating authority, resolution status. Calculate total fines. Quote specific evidence with sources. Write in Romanian."
    )
    
    situatie_actuala: str = Field(
        ...,
        description="Current status as of today. List: ongoing investigations (authority, charges, dates), resolved matters, current compliance posture, active restrictions/monitoring. Include assessment of whether risks are resolved or ongoing. Write in Romanian."
    )
    
    traiectorie: str = Field(
        ...,
        description="Risk trajectory: state clearly if IMPROVING (settlements, no new violations), STABLE (old issues resolved), DETERIORATING (new violations appearing), or UNKNOWN. Explain: compare earliest to most recent violation dates, trend analysis, evidence of remediation. Write in Romanian."
    )
    
    recomandare_relatie_afaceri: str = Field(
        ...,
        description="Clear business relationship recommendation. Choose one: TERMINATE (critical risk), SUSPEND (high risk pending resolution), ENHANCED_DUE_DILIGENCE (medium risk with strict monitoring), CONTINUE_WITH_MONITORING (low risk with standard monitoring), CONTINUE (minimal risk). Include rationale, specific conditions, required monitoring, and red lines. Write in Romanian."
    )
    
    concluzie_finala: str = Field(
        ...,
        description="Final conclusion in one clear paragraph: does entity meet compliance standards for partnership? State specific recommendation, key risks to monitor, and next review date recommendation. This is the definitive answer for decision-makers. Write in Romanian."
    )





class QueryPerformance(BaseModel):
    query_id: int
    query_text: List[str]
    query_lang: Optional[str] = None
    search_engine: str
    scenario:str
    links_initial: int
    links_after_sm_filter: Optional[int] = None


def merge_links_by_url(existing: List[LinkCollection], new: List[LinkCollection]) -> List[LinkCollection]:
    """
    Merge links by URL:
    - If link exists: update attributes (overwrite old + add new)
    - If link doesn't exist: append it
    """
    # Create map of existing links
    existing_map = {item.link: item for item in existing}

    # Update or add
    for new_item in new:
        if new_item.link in existing_map:
            # Update existing: copy all non-None attributes from new_item
            existing_item = existing_map[new_item.link]
            
            # Get all fields as dictionary
            new_data = new_item.model_dump()
            for field_name, new_value in new_data.items():
                if new_value is not None:  # Only update if value exists
                    setattr(existing_item, field_name, new_value)
        else:
            # Add new link
            existing.append(new_item)
    
    return existing
#URL doesn't exist → append new link
#URL exists AND new attribute is NOT None → assign/create/replace value in existing data
#URL exists AND new attribute is None → keep old value (don't change)



# we need custom reducer for query states to add information about kept links 
def merge_query_data_by_id(existing:List[QueryPerformance] , new:List[QueryPerformance]): 
    """
        Merge links by URL:
        - If qid exists: update attributes (overwrite old + add new)
        - If liqidnk doesn't exist: append it
        """
    # Create map of existing links
    existing_map = {item.query_id: item for item in existing}

    # Update or add
    for new_item in new:
        if new_item.query_id in existing_map:
            # Update existing: copy all non-None attributes from new_item
            existing_item = existing_map[new_item.query_id]
            
            # Get all fields as dictionary
            new_data = new_item.model_dump()
            for field_name, new_value in new_data.items():
                if new_value is not None:  # Only update if value exists
                    setattr(existing_item, field_name, new_value)
        else:
            # Add new link
            existing.append(new_item)
    
    return existing           



class UnifiedResearchState(TypedDict):
    
    # From UnifiedResearchState
    registration_number: str
    sanctions_data: dict
    messages: Annotated[List[AnyMessage], add_messages]
    search_results_raw: Annotated[List[LinkCollection] , operator.add]
    search_results_entity_filtered: Annotated[List[LinkCollection] ,  operator.add] # only appended with new data
    search_results_kterms_filtered: Annotated[List[LinkCollection] , merge_links_by_url] # appende with new data , or added raw content
    search_results_sm_filter: Annotated[List[LinkCollection] , merge_links_by_url] # filtering done with hyde , summary is added
    query_components:QueryComponentsInState
          #entity_name: str
          #expanded_query:str
    num_results_alias : int
    
    # From GenerateAnalystsState
    max_journalists: int
    journalists: List[Journalist]
    hyde_list: List[str]    
    # from evidence collection 
    evidence_claims: List[EvidenceClaim]

    # evidence_feedback
    url_feedback: Optional[AIMessage] # used for evaluation optimizer workflow
    evidence_feedback: Optional[AssessEvidenceQuality]

    # support MCP approach , tools will be added only to llm instance with tools, in
    # other cases we would refer to scenario 
    scenario_used : Annotated[List[str] , operator.add]
    scenario_selected: str # can cause issue related to missing reducer
    scenario_pool: List[str] # used for selection of scenario , we separate tools from scenario as tools mauy be repeated with different payloads 

    tool_pool:List[str] # used for payload generation 

    query_counter:int
    search_query_performance: Annotated[List[QueryPerformance],  merge_query_data_by_id ]

    search_result_assets:list[str]
    final_conclusion:FinalReport



#Inside a node's return statement: When you return "search_results_raw": new_element, LangGraph will automatically apply operator.add 
#(which concatenates lists). So if new_element is a LinkCollection, it gets wrapped in a list and appended to the existing list.

#For manual testing outside nodes: The Annotated[List[LinkCollection], operator.add] annotation only tells LangGraph how to handle 
#updates when the graph processes node returns. When you're manually constructing or modifying the state dictionary outside of the 
#graph execution (like in tests or debugging), you need to manually manage the list yourself using .append() or list concatenation.





