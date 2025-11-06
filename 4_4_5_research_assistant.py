#!/usr/bin/env python
# coding: utf-8

# In[1]:


# By chat model we mean LLM model which operates with chats 
import os 
import json

from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

from pydantic import BaseModel, Field

## check performance in agent workflow 
from typing import TypedDict, Annotated, List, Dict, Optional
from langchain_core.messages import BaseMessage, AnyMessage, ToolMessage,HumanMessage, AIMessage, SystemMessage
from langgraph.graph import add_messages , START, END , StateGraph
from IPython.display import Image, display 
from langgraph.checkpoint.memory import MemorySaver

from googleapiclient.discovery import build
from urllib.parse import urlparse

# prepare model for embeddings  
from langchain_openai import OpenAIEmbeddings 
from langchain_core.tools import tool, StructuredTool


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import numpy.typing as npt

from tavily import TavilyClient 

import subprocess
from IPython.display import Image, display


## OpenAI configuration
api_key_var = os.environ.get("OPENAI_API_KEY")
print("OpenAI API: " , api_key_var)


# keys for tavily 
tavily_api = os.environ.get("TAVILY_API_KEY")
print("Tavily API: " , tavily_api)

# api for goole
api_google = os.environ.get('GOOGLE_API_KEY')
print("Google API: " , api_google)

# define search engine
SEARCH_ENGINE_ID = '26dda816634bd4044'



# In[2]:


# test OpenAI api
# model as basis for tool payload creation 

llm = ChatOpenAI(
    model="gpt-4o-mini",           # BEST: Most reliable for tool calling
                              # Better function calling accuracy than gpt-4o-mini
                              # Handles complex tool sequences properly
    
    temperature=0,            # PERFECT: Maximum precision for tool calls
                              # Deterministic parameter selection
                              # Consistent function calling behavior
    
    max_retries=3,            # ESSENTIAL: Tool calls can fail, need retries
                              # Network issues more critical with function calling
    
    timeout=30,               # ADEQUATE: Tool calls need processing time
                              # Allows for complex parameter reasoning
    
    api_key=api_key_var
)


llm.invoke(input="Hello") # get AI message


# In[3]:


# Prepare embedding model
embedding_main = OpenAIEmbeddings(api_key=api_key_var, model="text-embedding-3-small")
embedding_cross_lang = OpenAIEmbeddings(api_key=api_key_var, model="text-embedding-3-large")

# For HyDE comparison we will employ large embedding model for richer semantic representation
# Larger model should be able to capture same concepts in different languages 
# Similar concepts should be positioned closer in the vector space

# Test with similar concepts + unrelated text
test_sentences = [
    "Искусственный интеллект меняет мир",        # AI is changing the world (Russian)
    "Artificial intelligence is changing the world",  # AI is changing the world (English) - fixed typo "word" -> "world"
    "Я люблю есть пиццу по вечерам"              # I love eating pizza in the evenings (unrelated)
]

# Get embeddings
embeddings = [np.array(embedding_cross_lang.embed_query(x)).reshape(1, -1) for x in test_sentences]

# Calculate similarities
print("Similarity between Russian and English (should be HIGH):")
print(f"  {cosine_similarity(embeddings[0], embeddings[1])[0][0]:.4f}")

print("\nSimilarity between Russian AI and Pizza (should be LOW):")
print(f"  {cosine_similarity(embeddings[0], embeddings[2])[0][0]:.4f}")

print("\nSimilarity between English AI and Pizza (should be LOW):")
print(f"  {cosine_similarity(embeddings[1], embeddings[2])[0][0]:.4f}")

del embeddings

# large vs small:
# better at capturing same meaning in different languages


# In[4]:


get_ipython().system('jupyter nbconvert --to python 4_4_5_research_assistant.ipynb')


# #### LLM models api definition

# In[5]:


# Translation and key terms extraction from short input
llm_translation_or_terms = ChatOpenAI(
    model="gpt-4o-mini",  # BEST: Cheapest model that excels at translation tasks
                          # Translation is pattern-matching, doesn't need reasoning power
                          # GPT-4o-mini handles languages perfectly at 10x lower cost
    
    temperature=0.1,      # BEST: Near-deterministic output for consistent translations
                          # Translation should be consistent, not creative
                          # 0.1 allows tiny variation while preventing hallucinations
    
    max_tokens=100,       # BEST: Perfect for short translations and key terms
                          # Prevents model from over-explaining or adding fluff
                          # Saves money by limiting output length
    
    timeout=5,           # BEST: Translation should be fast, 5s is generous
                          # If it takes longer, something's wrong with the request
                          # Prevents hanging requests from eating budget
    
    max_retries=2         # BEST: Translation usually works first try
                          # 2 retries handles temporary network issues
                          # More retries waste time and money on bad requests
)

# Summary of content extracted from URL with tavily  
llm_url_content = ChatOpenAI(
    model="gpt-4o-mini",  # BETTER CHOICE: Your original "gpt-4o" was budget-killer
                          # URL content analysis is mostly extraction/summarization
                          # GPT-4o-mini handles this 90% as well for 10% of the cost
                          # Save premium models for truly complex reasoning
    
    temperature=0.2,      # BEST: Factual analysis needs consistency, not creativity
                          # Prevents model from "making up" facts about content
                          # Ensures similar content gets similar analysis
    
    max_tokens=1500,      # GOOD: Enough for detailed analysis without waste
                          # Your original 1500 was excessive for most URL content
                          # Can capture key points without paying for verbosity
    
    top_p=0.95,          # GOOD: Slight randomness reduction for factual tasks
                          # Works with low temperature to ensure accuracy
                          # 0.95 allows some vocabulary variation while staying factual
    
    timeout=15,          # BETTER: Reduced from implicit 60s default
                          # Content analysis shouldn't take forever
                          # Faster feedback loop for debugging
)


# Final summarization of multiple content pieces
# more agents requires more tokens to be generated
llm_class_generation = ChatOpenAI(
    model="gpt-4o-mini",     # BETTER: Cheaper, perfectly capable for structured output
                             # Class generation is simple task, save budget
    
    temperature=0.2,         # BEST: Near-deterministic for consistent structure
                             # You want predictable field values
    
    max_tokens=600,          # CORRECT: 3 fields with descriptions = ~50-80 tokens
                             # Prevents rambling, forces concise responses
    
    timeout=60,             # SUFFICIENT: Simple generation shouldn't take long
    
    max_retries=2           # ADEQUATE: Structured output usually works first try
)
# LengthFinishReasonError: Could not parse response content as the length limit was reached - CompletionUsage(completion_tokens=200, prompt_tokens=426, total_
# APITimeoutError: Request timed out.


llm_hyde_generation = ChatOpenAI(
    model="gpt-4o-mini",     # OPTIMAL: Cost-effective for creative writing
                             # Hyde articles don't need complex reasoning
                             # Good at generating diverse writing styles
    
    temperature=0.5,         # BEST: Creative variation for different journalist styles
                             # Each journalist should write distinctively
                             # Higher temp = more stylistic diversity
    
    max_tokens=500,          # CORRECT: 200-300 word articles need ~350-400 tokens
                             # Allows for complete thoughts and proper structure
                             # Prevents truncation mid-sentence
    
    timeout=60,             # ADEQUATE: Article generation needs more time than classes
                             # Creative writing takes longer than structured output
    
    max_retries=2           # SUFFICIENT: Article generation usually succeeds first try
)




# Final summarization of multiple content pieces
llm_summarization = ChatOpenAI(
    model="gpt-4o",       # CORRECT: You need the reasoning power here!
                          # Multi-document synthesis requires intelligence
                          # Finding connections across articles needs advanced reasoning
                          # 128K context window handles multiple long articles
                          # This is where you should spend your budget
    
    temperature=0.1,      # BEST: Even complex analysis should be consistent
                          # You want reliable insights, not creative interpretations
    
    max_tokens=3000,      # CORRECT: Complex multi-article analysis needs space
                          # Executive summaries of multiple sources require detail
                          # You need room for nuanced insights and connections
    
    top_p=0.95,          # GOOD: Allows sophisticated language while staying accurate
    
    timeout=45,          # INCREASED: Complex reasoning takes time
                          # Multi-document analysis can't be rushed
                          # 45s allows for thoughtful processing
    
    max_retries=3        # CORRECT: Complex tasks may need retries
                          # Network issues more likely with longer processing
)




# #### Main classes for workflow

# In[6]:


## Support for journalists and HyDe articles 
class Journalist(BaseModel):
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
class EvidenceClaim(BaseModel):
    """Track claims extracted from summaries - focused on consolidation"""
    claim_text: str = Field(description="Specific claim made (e.g., 'FinCEN fined company €50M in 2023')")
    claim_type: str = Field(description="Type: fine|investigation|allegation|clearance|remediation")
    supporting_urls: List[str] = Field(description="ALL URLs that mention this claim")
    date_publish: str = Field(description=" Publication date or event date ")


class ClaimsFromSummaries( BaseModel ): 
    evidence_claims: List[EvidenceClaim] = Field(
    description="All extractable claims from the summaries"
      )

# Accountability: Every claim is traced to specific source
# Credibility: Check if the claim is supported by evidence
# Gap identification: We know what information is missing from the data
# Iteration: Check how many times we researched missing evidences 



# Support of google search arguments 
class candidate_names(BaseModel):
       comp_names_variation: List[str] = Field(description="Different possible variations of company name")

class search_terms_extraction(BaseModel):
       or_terms: List[str] = Field(description="List of alternative terms and variations for OR search in the specified language")




# Data and Meta for extracted content
class LinkCollection(BaseModel):
       displayLink: str = Field(description="The display URL shown in search results (usually domain name)")
       link: str = Field(description="The full URL of the search result")
       raw_content: str = Field(default="", description="Content extracted from URL")
       summary: str = Field(default="", description="Summary of content extracted from URL")
       hyde_score: Optional[float] = Field(
        default=None,
        description="Max value of similarity between content of web page and HyDe articles"
    )



# main state
class UnifiedResearchState(TypedDict):
    # From UnifiedResearchState
    messages: Annotated[List[AnyMessage], add_messages]
    search_results: List[LinkCollection]
    filtered_results: List[LinkCollection]
    entity_name: str
    expanded_query:str
    
    # From GenerateAnalystsState
    max_journalists: int
    journalists: List[Journalist]
    hyde_list: List[str]    

    # from evidence collection 
    evidence_claims: List[EvidenceClaim]



# #### Supportive tools definition 

# In[7]:


## Function to generate different names of the company
# can be used later

entity_name = "Lukoil"


def extract_name_variations(company_name: str, llm) -> candidate_names:
    """Generate different variations of how a company name might appear in news articles"""
    
    prompt = PromptTemplate(
        input_variables=["company_name"],
        template='''
You are a text analysis expert. Given the original company name: "{company_name}"

Generate 4-6 possible ways this company name could appear in news articles, press releases, and public documents.

Consider these common variations:
1. Full official name with legal entity type (PJSC, LLC, Inc., Corp., Ltd.)
2. Shortened version without legal entity type
3. Common abbreviations or acronyms
4. Alternative spellings or transliterations
5. How it might appear in headlines (often shortened)
6. Brand name vs. legal entity name
7. Regional variations or subsidiaries

For example:
- Original: "PJSC Lukoil Oil Company"
- Variations: ["PJSC Lukoil", "Lukoil Oil Company", "Lukoil", "Lukoil Oil"]

- Original: "Apple Inc."
- Variations: ["Apple Inc", "Apple", "Apple Computer", "AAPL", "Apple Corporation"]

Original Company Name: {company_name}

Generate ONLY the list of name variations as a JSON array of strings.
        '''
    )
    
    structured_llm = llm.with_structured_output(candidate_names)
    chain = prompt | structured_llm
    response = chain.invoke({"company_name": company_name})
    
    return response

# Usage:
result = extract_name_variations("PJSC Lukoil Oil Company", llm)
variations_list = result.comp_names_variation
variations_list.append(entity_name)
variations_list
" ".join( variations_list)


# In[8]:


# define argument for google search 

def extract_orterms_from_query(search_query: str, llm, ln: str) -> str:
    """Extract relevant terms from search query for Google orTerms parameter in the specified language
    
    Args:
        search_query (str): The original search query
        llm: Language model instance
        ln (str): Language code ('en', 'ru', 'fr', 'ro', 'de')
        
    Returns:
        str: Space-separated string of OR terms in the specified language
    """
    
    # Language code to full name mapping
    language_mapping = {
        'en': 'English',
        'ru': 'Russian',
        'fr': 'French', 
        'ro': 'Romanian',
        'de': 'German'
    }
    
    # Get full language name or default to English
    full_language_name = language_mapping.get(ln.lower(), 'English')
    
    prompt = PromptTemplate(
        input_variables=["search_query", "language"],
        template='''
You are a search optimization expert. Analyze this search query and extract alternative terms that should be used in Google's orTerms parameter.

IMPORTANT: Generate ALL terms in {language} language. The response must be entirely in the specified language.

Original Query: "{search_query}"

Extract 6-10 alternative terms covering:
1. Company name variations (abbreviations, alternative spellings, transliterations), both in english and  {language}
2. Topic/subject variations (synonyms, related terms, technical terms)
3. Activity variations (different ways to describe the same concept)

For money laundering searches, consider terms like:
- Financial crimes, compliance violations, sanctions evasion
- AML violations, illicit finance, financial misconduct
- Regulatory violations, corruption, fraud

For company names, consider:
- Official names with/without legal entity types
- Common abbreviations, brand names
- Alternative spellings or transliterations

IMPORTANT: Keep each term SHORT - single words or very short phrases work best for OR logic.
CRITICAL: All terms must be in {language} language.

Example for English:
Query: "Search for news about Apple Inc related to tax evasion"
orTerms: ["Apple", "AAPL", "Apple-Inc", "tax-evasion", "tax-avoidance", "taxation", "fiscal", "treasury", "IRS", "revenue"]

Example for Romanian:
Query: "Căutați știri despre Apple Inc legate de evaziunea fiscală"
orTerms: ["Apple", "AAPL", "Apple-Inc", "evaziune-fiscală", "fraudă-fiscală", "taxare", "fiscal", "trezorerie", "venituri", "impozite"]

Example for Russian:
Query: "Поиск новостей об Apple Inc связанных с уклонением от налогов"
orTerms: ["Apple", "AAPL", "Apple-Inc", "уклонение-налогов", "налоговое-мошенничество", "налогообложение", "фискальный", "казначейство", "доходы", "налоги"]

Example for French:
Query: "Rechercher des nouvelles sur Apple Inc liées à l'évasion fiscale"
orTerms: ["Apple", "AAPL", "Apple-Inc", "évasion-fiscale", "fraude-fiscale", "taxation", "fiscal", "trésorerie", "revenus", "impôts"]

Example for German:
Query: "Suchen Sie nach Nachrichten über Apple Inc bezüglich Steuerhinterziehung"
orTerms: ["Apple", "AAPL", "Apple-Inc", "Steuerhinterziehung", "Steuerumgehung", "Steuervermeidung", "Steuerdelikte", "Steuerrecht", "Finanzbehörden", "Steuerprüfung", "Umsätze", "Gewinne", "Steuern"]

Query: "{search_query}"
Language: {language}

Generate ONLY a list of SHORT alternative terms as a JSON array of strings in {language} language.
        '''
    )
    
    structured_llm = llm.with_structured_output(search_terms_extraction)
    chain = prompt | structured_llm
    
    response = chain.invoke({
        "search_query": search_query, 
        "language": full_language_name
    })
    
    # Join the terms with spaces for orTerms parameter
    orterms_string = " ".join(response.or_terms)
    
    return orterms_string

# Usage examples:
# English
#search_query = "Search for news and information about TESLA Company company related to money laundering."
#orterms_for_search = extract_orterms_from_query(search_query, llm_translation_or_terms, "en")

#print(f"Original query: {search_query}")
#print(f"Generated orTerms: {orterms_for_search}")

# Romanian
#search_query = "Căutați știri și informații despre compania PJSC Lukoil Oil Company legate de spălarea banilor."
#orterms_for_search = extract_orterms_from_query(search_query, llm_translation_or_terms, "ro")

#print(f"Original query: {search_query}")
#print(f"Generated orTerms: {orterms_for_search}")

# Russian
#search_query = "Поиск новостей и информации о компании PJSC Lukoil Oil Company связанных с отмыванием денег."
#orterms_for_search = extract_orterms_from_query(search_query, llm_translation_or_terms, "ru")

#print(f"Original query: {search_query}")
#print(f"Generated orTerms: {orterms_for_search}")

# French
#search_query = "Rechercher des nouvelles et informations sur la société PJSC Lukoil Oil Company liées au blanchiment d'argent."
#orterms_for_search = extract_orterms_from_query(search_query, llm_translation_or_terms, "fr")

#print(f"Original query: {search_query}")
#print(f"Generated orTerms: {orterms_for_search}")

# German
#search_query = "Suchen Sie nach Nachrichten und Informationen über die Gesellschaft PJSC Lukoil Oil Company im Zusammenhang mit Geldwäsche."
#orterms_for_search = extract_orterms_from_query(search_query, llm_translation_or_terms, "fr")

#print(f"Original query: {search_query}")
#print(f"Generated orTerms: {orterms_for_search}")



# In[9]:


## Function to translate query 
## payload will be adjusted to the language but we also need to aligh language of request 

def translate_query_for_search(query: str, target_lang: str, llm_instance) -> str:
    """
    Translate search query to target language
    
    Args:
        query: Original search query
        target_lang: Target language code (en, ru, fr, ro, de)
        llm_instance: LLM instance for translation
        
    Returns:
        Translated query string
    """
    language_names = {
        "en": "English","ru": "Russian", "fr": "French","ro": "Romanian","de":"German"
    }

    target_language = language_names[target_lang]

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
        return translated
    except Exception as e:
        print(f"Translation error: {e}. Using original query.")
        return query
    

result = translate_query_for_search(" Information on Money laundering by Lukoil company " , "de" , llm_translation_or_terms)
result 


# ####  Google tool definition

# In[10]:


# common error BadRequestError: Max 20 URLs are allowed.
# https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list#try-it 

## define function for tool  schema
def google_search(query,  num_results = 10, hl="en", lr="lang_en", dateRestrict="y2", wikipedia_lang="en"):

    """
    Search Google using Custom Search API with configurable parameters.
    
    This function performs Google searches and returns structured results including titles, URLs, and snippets.
    The agent can customize language settings, time restrictions, and Wikipedia language exclusions to optimize search results for specific needs.
    
    Parameters:
    -----------
    query : str
        The search query string to execute
    api_key : str  
        Google Custom Search API key
    cse_id : str
        Google Custom Search Engine ID
        
    Agent Customizable Parameters:
    -----------------------------

    num_results : int, always included and defined by user
    Number of results to return (default: 5, max: 10 per API call), Value must not exceed 10, NOT customisable


    hl : str, always included (default: "en")
        Interface language - controls UI language and affects search quality
        Examples: "en", "ru", "de", "fr","ro"
        
    lr : str, always included (default: "lang_en") 
        Content language restriction - filters results by document language
        Examples: "lang_en", "lang_ru", "lang_de", "lang_fr", "lang_es"
        
    dateRestrict : str, always included (default: "d365")
        Time-based filtering for results freshness
        Examples: "d1" (past day), "w1" (past week), "m1" (past month), 
                 "m3" (past 3 months), "m6" (past 6 months), "y1" (past year)
    
    wikipedia_lang : str, always included (default: "en")
        Wikipedia language version to exclude from results
        Possible values: "en" (English), "ru" (Russian), "fr" (French), "ro" (Romanian), "de" (German)
        Maps to domains: en.wikipedia.org, ru.wikipedia.org, fr.wikipedia.org, ro.wikipedia.org, de.wikipedia.org
    
    Returns:
    --------
    list
        List of dictionaries containing search results with keys:
        - 'title': Result title
        - 'link': Result URL  
        - 'snippet': Result description/excerpt
        - Additional metadata from Google API
        
    Example Agent Usage:
    -------------------
    # For Russian content excluding Russian Wikipedia
    results = google_search(query, api_key, cse_id, hl="ru", lr="lang_ru", dateRestrict="y3", wikipedia_lang="ru")
    
    # For French content excluding French Wikipedia
    results = google_search(query, api_key, cse_id, hl="fr", lr="lang_fr", dateRestrict="y3", wikipedia_lang="fr")
    
    # For Romanian content excluding Romanian Wikipedia
    results = google_search(query, api_key, cse_id, hl="ro", lr="lang_ro", dateRestrict="y3", wikipedia_lang="ro")

    # For German content excluding German Wikipedia
    results = google_search(query, api_key, cse_id, hl="de", lr="lang_de", dateRestrict="y3", wikipedia_lang="de")

    """

    print("Executing google_search_payload ...")

    api_key = api_google
    cse_id = SEARCH_ENGINE_ID

    service = build("customsearch", "v1", developerKey=api_key)
    
    # Map language codes to Wikipedia domains
    wikipedia_domains = {
        "en": "en.wikipedia.org",
        "ru": "ru.wikipedia.org", 
        "fr": "fr.wikipedia.org",
        "ro": "ro.wikipedia.org",
        "de": "de.wikipedia.org"
    }
    
    wikipedia_site = wikipedia_domains.get(wikipedia_lang, "en.wikipedia.org")
    
    ## store search results 
    all_results = []
   
    ## query language must be aligned with search parameters
    translated_query = translate_query_for_search(query , hl , llm_translation_or_terms)

    ## orterms
    orterms_for_search = extract_orterms_from_query(translated_query, llm_translation_or_terms, hl )
    print("Additional search terms: " , orterms_for_search)
    

    try:
        result1 = service.cse().list(
            q=translated_query,
            cx=cse_id, # search engine name defined in the console 
            dateRestrict=dateRestrict,  # Agent customizable
            start=1,
            filter='1', # turn on duplicated filter content 
            hl=hl, # Agent customizable - user interface language
            lr=lr,  # Agent customizable - language of content             
            num=num_results, 
            orTerms=orterms_for_search, # each document must contain at least one of the additional terms 
            safe='off',
            siteSearch=wikipedia_site,  # Agent customizable - Wikipedia language version
            siteSearchFilter="e"  # Exclude Wikipedia
        ).execute()
        
        if 'items' in result1:
                # Extract only relevant information from each result
                filtered_items = []
                for item in result1['items']:
                    essential_data = {
                        'query':translated_query,
                        'title': item.get('title', ''),
                        'link': item.get('link', ''),
                        'snippet': item.get('snippet', ''),
                        'displayLink': item.get('displayLink', '')
                    }
                    filtered_items.append(essential_data)

                all_results.extend(filtered_items)
    
    except Exception as e:
        print(f"Error getting results: {e}")

    print("Executing google_search_payload DONE ")    

    return all_results



# Message for llm to create payload 
system_message_google_search = SystemMessage(content="""
You are a multi-language search assistant. When asked to search for information, 
call the google_search tool ONCE for EACH of these 5 languages:

1. **Romanian** (ro):
   - hl="ro" (Romanian interface)
   - lr="lang_ro" (Romanian content)
   - wikipedia_lang="ro" (exclude Romanian Wikipedia)

2. **English** (en):
   - hl="en" (English interface)  
   - lr="lang_en" (English content)
   - wikipedia_lang="en" (exclude English Wikipedia)

3. **Russian** (ru):
   - hl="ru" (Russian interface)
   - lr="lang_ru" (Russian content) 
   - wikipedia_lang="ru" (exclude Russian Wikipedia)

4. **French** (fr):
   - hl="fr" (French interface)
   - lr="lang_fr" (French content)
   - wikipedia_lang="fr" (exclude French Wikipedia)
                            
5. **German** (de):
   - hl="de" (German interface)
   - lr="lang_de" (German content)
   - wikipedia_lang="de" (exclude German Wikipedia)

IMPORTANT: Make exactly 5 tool calls (one per language). 
For each call, use num_results=5 to retrieve 5 search results per language.
Use dateRestrict="y1" (last year) for recent information.

"""
)


# In[11]:


## Test function 
tool_google_search = StructuredTool.from_function( google_search, name = "google_search" )

entity_name = "Lukoil"
query = f"{entity_name} money laundering compliance violations investigations"

# join instructions and entity name
messages = [ system_message_google_search ,  HumanMessage(content=query) ]

## create list of tools, which be called externally with  tool_call['args']
tools = [ tool_google_search  ]
tools_by_name = {tool.name:tool for tool in tools}

# bind first tool to 4o mnodel
llm_with_tools = llm.bind_tools(tools)

# test tool call without payload 
#payload = llm_with_tools.invoke( messages )


# In[12]:


## check payload values
#for tool_call in payload.tool_calls:
#    print(f"Tool: {tool_call['name']}")
#    print(f"ID: {tool_call['id']}")
#    print(f"Args: {tool_call['args']}")
#    print("---")

#print( payload.tool_calls[0])    


# In[13]:


# run the function  using single payload
#tool_call = payload.tool_calls[0]
#tool_google_search.invoke( tool_call['args'] )
#print("##-----------------##")

## Now run through app payloads  
#for tool_call in payload.tool_calls: # AI message with payload , last ai message with too l calls
#        tool = tools_by_name[tool_call["name"]]
#        observations = tool.invoke(tool_call["args"])
#        print(observations)


# In[14]:


## google search tool execution and allocation of seatch results
def tool_node_chunk_selection_exec(state: UnifiedResearchState):
    """Performs the tool call"""
    print("Executing tool_node_chunk_selection_exec...")
    
    result = []
    link_collections = []

    for tool_call in state["messages"][-1].tool_calls: # AI message with payload , last ai message with too l calls
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
           
        result.append(ToolMessage( 
                       content = observation, 
                       tool_call_id=tool_call["id"] , 
                       name = tool_call['name']))    
        
        # Extract LinkCollection data from observation
        for item in observation:
            if isinstance(item, dict):
                # Check if required keys exist
                if 'displayLink' in item and 'link' in item:
                    link_collection = LinkCollection(
                        displayLink=item['displayLink'],
                        link=item['link']
                    )
                    link_collections.append(link_collection)
                else:
                    # This will only print for items that are missing keys
                    available_keys = list(item.keys())
                    print(f"Item missing required keys. Available keys: {available_keys}")
                    print(f"Item content: {item}")
            else:
                print(f"Item is not a dict, it's: {type(item)}")
        
    # we need to populate LinkCollection class and store in  search_results 

    print("Executing tool_node_chunk_selection_exec END")
    return {"messages": result , "search_results":link_collections }   


## Node to create payload 
def llm_node(state: UnifiedResearchState):
    """LLM node that generates tool calls"""
    print("Executing llm_node...")
    
    # Get the last message from state
    entity_name  = state["messages"][-1] # human message
    query = f"{entity_name.content} money laundering criminal activity bribery"
    
    ## modified content 
    human_message_updated = HumanMessage(content=query, id=entity_name.id)
    print("Updated human message: " , human_message_updated.content)

    ## Replace content of the original human imput 
    system_message = system_message_google_search

    messages = [ system_message ,  human_message_updated ]
    
    # Call LLM with tools
    response = llm_with_tools.invoke( messages )
    print(f"LLM response: {response}")
    print("State fields " , state)
    
    # return all up to last message, replace last human input , company name(for prompts), expanded query for debugging in English
    return {"messages": state["messages"][:-1] + [human_message_updated, response], "entity_name": entity_name.content , "expanded_query":query }



# In[15]:


## function to display flow using cli

def render_mermaid_graph(mermaid_code, output_filename='graph.png', width=700, height=700, cleanup=True):
    """
    Render Mermaid diagram using CLI and display in Jupyter
    """
    # Create temporary mermaid file
    temp_mmd = output_filename.replace('.png', '.mmd').replace('.svg', '.mmd').replace('.pdf', '.mmd')
    
    try:
        # Write mermaid code to file
        with open(temp_mmd, 'w', encoding='utf-8') as f:
            f.write(mermaid_code)
        
        # Build command based on output format
        # Use the Windows .cmd version
        cmd = [
            'C:\\Users\\Admin\\AppData\\Roaming\\npm\\mmdc.cmd',  # Full path to Windows version
            '-i', temp_mmd,
            '-o', output_filename,
            '-w', str(width),
            '-H', str(height)
        ]
        
        # Add size parameters only for PNG/PDF
        if output_filename.endswith(('.png', '.pdf')):
            cmd.extend(['-w', str(width), '-H', str(height), '--scale', '2'])
        
        result = subprocess.run(cmd, capture_output=True, text=True) # we simulate running command like
        
        if result.returncode == 0:
            print(f"Graph rendered successfully: {output_filename}")
            try:
                display(Image(output_filename))
            except:
                print(f"Image saved to {output_filename} but could not display inline")
            return True
        else:
            print(f"Mermaid CLI error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("Mermaid CLI not found. Install with: npm install -g @mermaid-js/mermaid-cli")
        return False
    except Exception as e:
        print(f"Error rendering graph: {e}")
        return False
    finally:
        if cleanup and os.path.exists(temp_mmd):
            os.remove(temp_mmd)


# #### Prepare Tavily for content extraction

# In[16]:


## Generate summary of extracted content

def generate_url_summary(raw_content: str, llm, entity_name: str) -> str:
    """
    Generate a focused summary of raw content with emphasis on company reputation 
    and criminal activity using dedicated LLM.
    
    Args:
        raw_content: Text content to analyze
        llm: Language model instance for processing
        entity_name: Company name for context
        
    Returns:
        str: Focused summary or error message
        
    Raises:
        None: All exceptions are caught and handled gracefully
    """
    
    # Input validation
    if not raw_content or not raw_content.strip():
        return "Error: No content provided for analysis"
    
    if not entity_name or not entity_name.strip():
        return "Error: Entity name is required"
    
    # Truncate content if too long (prevent token limit issues)
    max_content_length = 8000  # Adjust based on your model's limits
    if len(raw_content) > max_content_length:
        raw_content = raw_content[:max_content_length] + "...[truncated]"
    
    prompt = f"""
            You are analyzing an article about {entity_name} company for anti-money laundering due diligence.

            Extract and summarize ANY information related to:

            FINANCIAL CRIMES (primary focus):
            - Money laundering allegations, investigations, charges, or convictions
            - AML program deficiencies or regulatory violations
            - Suspicious transaction monitoring failures
            - Know Your Customer (KYC) violations

            RELATED VIOLATIONS (also capture):
            - Sanctions violations (OFAC, EU, UN sanctions lists)
            - Tax evasion or aggressive tax avoidance schemes
            - Fraud, embezzlement, or misappropriation of funds
            - Corruption, bribery, or kickbacks
            - Securities violations involving misrepresentation

            COMPLIANCE ACTIONS (positive or negative):
            - Regulatory fines, penalties, or enforcement actions
            - Legal proceedings, settlements, or court rulings
            - Compliance program implementations or improvements
            - Internal investigations or audits
            - Management changes related to compliance issues

            CRITICAL INSTRUCTIONS:
            1. **Distinguish allegation vs. fact**: 
            - "Company accused of..." (allegation)
            - "Company convicted of..." (proven fact)
            - "Investigation closed without charges" (exoneration)

            2. **Include legal status**:
            - Ongoing investigation
            - Settled with fine
            - Dismissed by court
            - Under appeal

            3. **Capture both negative AND positive**:
            - Violations AND remediation efforts
            - Fines paid AND compliance improvements implemented

            4. **For irrelevant articles**: 
            "Article discusses {entity_name} [topic] but contains no financial crime or compliance information."

            5. **Always include**:
            - Date of article or events (if available)
            - Source publication
            - Specific amounts (fines, transactions, etc.)
            - Regulatory bodies involved (FinCEN, OFAC, DOJ, etc.)

            FORMAT:
            Date: [publication date or event date] | Source: [publication] | Legal Status: [ongoing/settled/dismissed/alleged] | [factual summary]

            EXAMPLES:

            Good - captures nuance:
            "Date: 2023-05-15 | Source: Reuters | Legal Status: Settled | FinCEN fined company €50M for AML violations during 2018-2020 related to Estonian branch operations. Settlement included admission of inadequate transaction monitoring. Company implemented new AML program in 2022 with independent monitor."

            Good - shows exoneration:
            "Date: 2024-01-10 | Source: Financial Times | Legal Status: Dismissed | DOJ investigation into alleged sanctions violations closed after 18-month probe. No charges filed. Company cleared of all allegations."

            Good - shows ongoing risk:
            "Date: 2024-03-20 | Source: WSJ | Legal Status: Ongoing | Company executives under investigation by EU authorities for potential money laundering related to Russian transactions 2021-2022. Company denies wrongdoing. Investigation ongoing, no charges filed yet."

            Good - irrelevant article:
            "Date: 2024-02-15 | Source: Bloomberg | Article discusses {entity_name}'s Q4 earnings and CEO succession but contains no financial crime or compliance information."

            Bad - too vague:
            "Company has compliance issues." 
            (Missing: dates, specifics, amounts, legal status, remediation)

            Content to analyze:
            {raw_content}

            Summary (follow format above):
            """
           
    try:
        response = llm.invoke(prompt)
        
        # Validate response
        if not response or not hasattr(response, 'content'):
            return "Error: Invalid response from language model"
        
        summary = response.content.strip()
        
        # Ensure we got actual content
        if not summary:
            return "Error: Empty response from language model"
            
        return summary
        
    except Exception as e:
        # More specific error handling
        error_msg = f"Error generating reputation summary: {str(e)}"
        print(error_msg)  # For debugging
        return "Error: Could not generate reputation summary due to processing issues"


# Fixed test call
result = generate_url_summary(
    "Lukoil was accused of money laundering in 2023. The company faced several lawsuits and fines related to financial misconduct. However, it has not been involved in any major scandals or controversies recently.",
    llm_url_content,
    "Lukoil"  # Added missing entity_name parameter
)
print(result)



# In[17]:


# Initialize client once at module level
tavily_client = TavilyClient(api_key=tavily_api)

# function will rin inside the node, tool is not generated 
# as application is strainghforward 

def tavily_content_extractor(
    dummy_state: List[LinkCollection],    
    extract_depth: str = "basic",
    include_raw_content: bool = True,
    entity_name:str = ""

) -> List[LinkCollection]:
    
    print("Executing content extraction...")
    
    urls = [link.link for link in dummy_state]
    response = tavily_client.extract(urls=urls, extract_depth=extract_depth)
    
    # Create URL to content mapping, as tavily may return data non sequentially
    url_to_content = {
        item.get("url", ""): item.get("raw_content", "")
        for item in response.get('results', [])
    }
    
    # Create new LinkCollection objects with matched raw_content
    updated_collections = []
    for link_collection in dummy_state:
        raw_content = url_to_content.get(link_collection.link, "")
        
        if raw_content:  # may not be present for pdf 

            # generate summary
            print(f"Generate summary for URL: {link_collection.displayLink}")
            raw_content_summary = generate_url_summary(raw_content, llm_url_content , entity_name) 
        else:
            print(f"No content found for URL: {link_collection.displayLink}")
            raw_content_summary = ""

        updated_collection = LinkCollection(
            displayLink=link_collection.displayLink,
            link=link_collection.link,
            raw_content=raw_content,
            summary=raw_content_summary
        )
        
        updated_collections.append(updated_collection)

       
    print("Executing content extraction Done")

    return updated_collections

# Test 
#dummy = tavily_content_extractor(dummy_state=results["search_results"],  entity_name= results["entity_name"] )


# In[18]:


# the function above will not be implemented as tool 
# we will update search_results component for each LinkCollection

def extract_content(state: UnifiedResearchState) -> UnifiedResearchState :
    """Node function to extract content from URLs in search results"""
    
    print("Executing extract_content...")
    
    dummy_state = state["search_results"].copy()
    updated_search_result = tavily_content_extractor(dummy_state=dummy_state , entity_name = state["entity_name"])
    
    print("Extract_content completed Done")

    return {"search_results": updated_search_result} # replace the searchresult field
 


# #### Unite Research and HyDe agent into 1

# In[19]:


## input comes from expanded query formed after llm node

## now define nodes 

journalist_instructions = """You are tasked with creating exactly {max_journalists} AI 
journalist personas who will analyze the same financial crime topic from different critical perspectives for banking risk assessment. 

Topic under investigation: {crime_topic}

Create {max_journalists} journalists who each bring a unique lens to evaluate whether a bank should collaborate with a B2B company:

1. **Regulatory Compliance Analyst**: Focuses on regulatory violations, sanctions, and legal compliance issues that would trigger regulatory action against the bank.

2. **Financial Due Diligence Investigator**: Examines the company's financial integrity, transaction patterns, and money flow irregularities that pose reputational risk.

3. **Operational Risk Assessor**: Analyzes operational red flags, corporate governance failures, and business practice irregularities that indicate systemic risk.

4. **Reputational Risk Evaluator**: Investigates media coverage, public scandals, and association risks that could damage the bank's reputation through partnership.

5. **KYC/AML Specialist**: Focuses on Know Your Customer failures, suspicious beneficial ownership, and anti-money laundering concerns that create compliance liability.

For each journalist:
- expertise: Their specific risk assessment specialization
- perspective: How they evaluate partnership risks from their expert viewpoint  
- style: Their analytical approach to uncovering risks relevant to banking partnerships

Return all journalists in the HydePerspectives format."""



def create_journalists( state:UnifiedResearchState  ) :
     
     """ create journalist """ 

     print("Executing create_journalists ... ")

      # prompts arguments
     crime_topic=state['expanded_query']
     max_journalists=state['max_journalists']
     
     # llm to generate persona 
     structured_llm = llm_class_generation.with_structured_output( HydePerspectives )

     system_message = journalist_instructions.format( crime_topic = crime_topic , 
                                                      max_journalists= max_journalists
                                                     )
     print( "System Instructions: " , system_message[:100])
     
     # populate values  
     journalists = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content="Generate the set of journalists.")])

     print("Executing create_journalists Done ")

     return { "journalists": journalists.journalists }



## function to write article 
def generate_hyde_document(state:UnifiedResearchState): 

    """Generate Hyde articles using journalist personas"""

    print("Executing generate_hyde_document ... ")

    article_template = """You are a journalist with the following profile:

            Expertise: {expertise}
            Perspective: {perspective}
            Style: {style}

            Topic: {crime_topic}

            Write a hypothetical article about this crime_topic that authentically reflects your expertise, 
            perspective, and writing style. The article should be 200-300 words and demonstrate your unique approach to covering this story."""
            
    hyde_list = []

    print("Unpacking journalists classes")

    ## unpack values in journalists field
    journalists =  [ state["journalists"] ] 

    for elem in state['journalists']:
        prompt = article_template.format(
            expertise=elem.expertise,
            perspective=elem.perspective, 
            style=elem.style,
            crime_topic=state["expanded_query"]
        )

        article = llm_hyde_generation.invoke([HumanMessage(content=prompt)])
        hyde_list.append(article.content)

    print("Executing generate_hyde_document Done ")    

    return {"hyde_list": hyde_list}



# #### Prepare module for cosine similarity

# In[20]:


## ok test comparison algo , each article is compared against
## HeDe docs

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# we compare 1 article agains max_journalist , the format is ( n_row , n_journalists ) 

text_samples = [
    "The financial institution faces significant regulatory compliance challenges and potential money laundering violations that could impact its banking partnerships.",
    
    "Recent investigations reveal operational risk factors including inadequate anti-money laundering controls and suspicious transaction monitoring failures.",
    
    "Market analysts express concerns about reputational damage from ongoing legal proceedings and regulatory scrutiny affecting stakeholder confidence."
]

url_content = text_samples[1]

## form embedding of original 

vector_url_content = np.array( embedding_cross_lang.embed_query(url_content) )
vector_url_content = vector_url_content.reshape(1,-1)
print(vector_url_content.shape)


# In[21]:


# function will be iteravely applied over each content 

def check_content_similarity(
    search_results: List[LinkCollection], 
    hyde_list: npt.NDArray[np.float64]
) -> List[LinkCollection]:
    
    """
    Check if URL content has high similarity to any Hyde document
    
    Args:
        search_results: List of LinkCollection objects with raw_content
        hyde_list: numpy array of shape (n, 1536) containing Hyde embeddings
    
    Returns:
        float: Maximum cosine similarity score rounded to hundredth (2 decimal places)
    """
    
    # Store max similarity score for each URL
    updated_collections = []  # stroe new values for the collection, later replace, solution is redundant
    for link in search_results:

        url_content = link.raw_content
        
        # Embed URL content, cap is 300 000 tokens, 1 token is aproximately 4 characters, Embedding error: Error code: 400 - {'error': {'message': 'Requested 307281 tokens, max 300000 tokens per request', 't
        try:
           url_content_embed = np.array(
               embedding_cross_lang.embed_query(url_content[:800000])
           ).reshape(1, -1)
        except Exception as e:
           print(f"Embedding error: {str(e)[:100]}")
           url_content_embed = np.zeros((1, 3072))  # Default embedding for text-embedding-3-large
        
        # calculate similarity scores against all Hyde embeddings
        similarity_scores = cosine_similarity(url_content_embed, hyde_list)
        
        # Find and store max similarity for this URL
        max_score = round(float(np.max(similarity_scores)), 2)
        
       
        # update the state values , fix later 
        updated_collection = LinkCollection(
            displayLink=link.displayLink,
            link=link.link,
            raw_content=link.raw_content,
            summary=link.summary,
            hyde_score=max_score
        )
        
        updated_collections.append(updated_collection)
    
    return updated_collections

# Usage:
#results["search_results"] = check_content_similarity(
#    results["search_results"], 
#    hyde_content_list_embed
#)

# Tools are for external API, actions which requires LLM to decide parameters , sityation whcih required dynamic parameters
# Here we process data which already in a state, no external api, state modification, sequential processing 


# In[22]:


## Node to update  state["search_results"] and keep only relevant material for summary 

def filter_relevant_content(state: UnifiedResearchState) -> UnifiedResearchState:
    """Filter and sort search results by HyDE relevance score"""
    
    print("Executing filter_relevant_content...")
    
    RELEVANCE_THRESHOLD = 0.3  # Configurable
    TOP_K = 15  # Maximum articles to keep
    
    # Filter items with scores
    scored_items = [
        link for link in state["search_results"]
        if link.hyde_score is not None
    ]
    
    # Sort by score (descending)
    sorted_items = sorted(
        scored_items, 
        key=lambda x: x.hyde_score, 
        reverse=True
    )
    
    # Apply threshold and limit
    filtered = [
        item for item in sorted_items
        if item.hyde_score >= RELEVANCE_THRESHOLD
    ][:TOP_K]
    
    # Diagnostic output
    print(f"Total articles: {len(state['search_results'])}")
    print(f"Articles with scores: {len(scored_items)}")
    print(f"Above threshold ({RELEVANCE_THRESHOLD}): {len(filtered)}")
    
    if filtered:
        scores = [item.hyde_score for item in filtered]
        print(f"Score range: {min(scores):.2f} - {max(scores):.2f}")
        print(f"Top 5 sources: {[item.displayLink[:30] for item in filtered[:5]]}")
    else:
        print("WARNING: No articles passed relevance threshold!")
    
    print("Executing filter_relevant_content Done")
    
    return {"filtered_results": filtered}


# In[23]:


## now prepare function to integrate into main flow 

# with cosine we should distingusin 
#company accused in money laundering, mentioned in article about laundering
# denying accusations, complying with anti money policy

def assign_score_vs_hyde(state: UnifiedResearchState) -> UnifiedResearchState :
    """Node function to estimate similarity score agains HyDe documents"""
    
    print("Executing assign_score_vs_hyde...")

    ## prepare HydeEmbeddings 
    hyde_content_list = state.get("hyde_list", None)
    # hyde_content_list_embed = np.vstack([  np.array( embedding_cross_lang.embed_query(article) ).reshape(1,-1) for article in hyde_content_list ])

    hyde_embeddings = embedding_cross_lang.embed_documents(hyde_content_list)
    hyde_content_list_embed = np.array(hyde_embeddings)
    
    ## prepare input for score estimation
    dummy_state = state["search_results"].copy()
    updated_search_result = check_content_similarity(search_results=dummy_state , hyde_list = hyde_content_list_embed )
    
    print(" Executing assign_score_vs_hyde  Done")

    return {"search_results": updated_search_result} # replace the searchresult field


# #### Evidence Validation

# In[24]:


# We need verifiation that final assessment clains are based on the evidence 
# collected as summaries 
# We need to emplly evidence therefore add modules to trace the process

# Eventually, evidences must be collected into reusable sharable format 
# It should help to avoid hallusination and back up arguments with evidences 

#llm_claim_extraction = ChatOpenAI(
#    model="gpt-5",        # gpt 5 has higher latency, frequent limit hits 
#    temperature=0.1,      # Zero for deterministic extraction
#    max_tokens=7000,      # Enough for multiple claims
#    timeout=60,           # Complex consolidation takes time
#    api_key=api_key_var
#)
# we require consistent deduplication, inproved output structure with more discinplined extraction, lower hallucination
# LengthFinishReasonError: Could not parse response content as the length limit was reached - CompletionUsage(completion_tokens=4000, prompt_tokens=3174, total_tokens=7174,


## test the class 

def extract_evidence_claims(state:UnifiedResearchState) -> Dict: 
    
    """Extract claims from summaries and consolidate duplicate claims across sources"""
    
    print("Extracting and consolidating evidence claims...")

    # prepare summaries 
    summaries_data = [] # each element is dictionary wish summary data 

    for link in state["filtered_results"]: 
        if link.summary.strip(): 
            summaries_data.append(
                {
                "url": link.link,
                "source": link.displayLink,
                "summary": link.summary, # contains date
                "hyde_score": link.hyde_score
                }
            )

    extraction_prompt = f"""
    Extract financial crime claims about {state["entity_name"]} from news summaries.
    
    CONTEXT: These summaries were filtered using HyDE (Hypothetical Document Embeddings) scoring.
    The hyde_score indicates relevance (0-1 scale) by comparing each article against hypothetical 
    articles written by financial crime experts about {state["entity_name"]}'s money laundering risks.
    Higher scores (>0.5) suggest strong relevance to AML/compliance topics.
    
    FILTERING RULES:
    1. ONLY extract claims about {state["entity_name"]} - ignore any other companies mentioned
    2. If a summary discusses another company's investigation/scandal, skip it entirely
    3. Skip summaries stating "no information about {state["entity_name"]}"
    4. Split multi-date summaries into separate claims (some contain 5-8 different events)
    
    CONSOLIDATION RULES:
    When multiple sources report the same event (even with different wording):
    - Create ONE unified claim
    - Combine ALL supporting URLs into one list
    - Preserve ALL source summary texts
    - Count independent sources correctly
    
    For each UNIQUE claim, provide:
    
    claim_text: 
    - Include specific details: amounts, dates, agencies involved
    - Example: "Company paid €50M fine to FinCEN in May 2023 for AML violations"
    
    claim_type: Select exactly one:
    - "fine" → monetary penalties (paid or agreed to pay)
    - "investigation" → active regulatory/criminal probes
    - "allegation" → unproven accusations or claims
    - "clearance" → dismissed cases, exonerations, closed without charges
    - "remediation" → compliance improvements, new controls, reforms implemented
    
    supporting_urls: List of ALL URLs mentioning this specific claim
    
    date_publish: Date of claim publication

    EXAMPLE CONSOLIDATION:
    Three sources mention same fine with variations:
    - Source A: "€1.5 billion settlement with authorities"
    - Source B: "agreed to pay €1.5B for AML failures"  
    - Source C: "€1.5 billion fine for money laundering"
    
    Result → ONE claim with:
    - claim_text: "{state["entity_name"]} paid €1.5 billion fine to settle AML violations"
    - claim_type: "fine"
    - supporting_urls: [urlA, urlB, urlC]
    - date_publish: "2023-12-23"
    
    EXTRACTION PRIORITIES:
    - Focus on the most serious violations and largest fines first
    - Capture the full timeline of events (earliest to latest)
    - Include both negative findings AND positive remediation efforts
    - Distinguish between different regulatory jurisdictions (US, EU, national)
    
    Summaries to analyze:
    {json.dumps(summaries_data, indent=2)}
    
    Return only consolidated, unique claims about {state["entity_name"]}.
    """
    # debug
    #print(json.dumps(summaries_data, indent=2))


    structured_llm = llm_summarization.with_structured_output( ClaimsFromSummaries )

    response = structured_llm.invoke([
        SystemMessage(content="You are a forensic analyst consolidating evidence."),
        HumanMessage(content=extraction_prompt)
    ])

    print(f"Extracted {len(response.evidence_claims)} unique claims from {len(summaries_data)} summaries")
    # json dums return string which is applied to promot via f syntax

    print(response.evidence_claims)
    return {"evidence_claims" : response.evidence_claims}

#buff = extract_evidence_claims(results)  # tested 



# In[25]:


# adjust summarisation node 

def generate_risk_assessment(state: UnifiedResearchState):
    """Generate comprehensive AML risk assessment from evidence claims"""
    
    print("Executing generate_risk_assessment...")
    
    # Group claims by type for analysis
    claims_by_type = {
        "fine": [],
        "investigation": [],
        "allegation": [],
        "clearance": [],
        "remediation": []
    }

    print("State keys" , state.keys())
    
    print("Iterate through claim cases: ")
    for claim in state["evidence_claims"]:
        claims_by_type[claim.claim_type].append({
            "text": claim.claim_text,
            "sources": len(claim.supporting_urls),
            "date_publish":claim.date_publish
        })
    
    entity_name = state["entity_name"]
    
    system_message = SystemMessage(content=f"""
    You are a senior AML compliance officer conducting risk assessment for {entity_name}.
    
    Analyze the evidence to determine banking partnership viability.
    
    KEY ASSESSMENT CRITERIA:
    
    1. VIOLATION SEVERITY
    - Calculate total fines from the evidence (don't hardcode)
    - Identify the scale of violations (amounts involved vs fines paid)
    - Determine if violations were systemic or isolated
    - Check if company admitted guilt or just settled
    
    2. CURRENT RISK STATUS
    - Count active investigations (ongoing = unresolved risk)
    - Identify investigating authorities (DOJ, EU, FATF = high concern)
    - Assess geographic spread (multiple jurisdictions = higher risk)
    - Evaluate timeline (recent violations = weak current controls)
    
    3. CONTROL ENVIRONMENT
    - Evaluate remediation efforts (genuine improvement vs PR)
    - Check for leadership accountability (executives resigned?)
    - Assess system improvements (new AML systems, training?)
    - Look for independent validation (monitors, audits)
    
    4. PATTERN RECOGNITION
    - Multiple fines for similar issues = poor compliance culture
    - Repeated violations across years = systemic failure
    - Mix of old violations + strong remediation = possible reform
    - Only allegations with no convictions = lower concern
    
    DECISION FRAMEWORK:
    
    AVOID PARTNERSHIP if:
    - Criminal investigations ongoing
    - Multiple violations without meaningful remediation
    - Pattern of violations continuing to present
    - Total fines exceed €1 billion with no improvement
    
    ENHANCED DUE DILIGENCE if:
    - Significant past violations but settled
    - Active remediation program with evidence
    - Civil investigations ongoing (not criminal)
    - Mix of violations and clearances
    
    STANDARD MONITORING if:
    - Old issues (3+ years) fully resolved
    - Strong remediation verified
    - Recent clearances of allegations
    - No new violations in 2+ years
    
    REQUIRED OUTPUT:
    1. EXECUTIVE SUMMARY: Core issues and total financial impact (calculate from evidence)
    2. VIOLATION ANALYSIS: What they did wrong and when
    3. CURRENT STATUS: What's resolved vs ongoing
    4. TRAJECTORY: Getting better or worse? (compare dates)
    5. RISK RATING: HIGH/MEDIUM/LOW with specific justification
    6. PARTNERSHIP RECOMMENDATION: Your decision with conditions
    
    Be specific about amounts, dates, and authorities involved.
    """)
    
    # Prepare evidence for analysis
    evidence_prompt = f"""
    Evidence for {entity_name}:
    
    CONFIRMED FINES:
    {json.dumps(claims_by_type['fine'], indent=2)}
    
    ACTIVE INVESTIGATIONS:
    {json.dumps(claims_by_type['investigation'], indent=2)}
    
    UNPROVEN ALLEGATIONS:
    {json.dumps(claims_by_type['allegation'], indent=2)}
    
    DISMISSED CASES/CLEARANCES:
    {json.dumps(claims_by_type['clearance'], indent=2)}
    
    REMEDIATION EFFORTS:
    {json.dumps(claims_by_type['remediation'], indent=2)}
    
    Based on this evidence:
    1. Calculate the total fines paid
    2. Identify the most serious issue (hint: look for large transaction amounts)
    3. Determine if this is historical or ongoing risk
    4. Make your partnership recommendation with clear reasoning
    
    """
    print("Request LLM Response ")
    messages = [system_message, HumanMessage(content=evidence_prompt)]
    analysis_response = llm_summarization.invoke(messages)
    
    print("Risk assessment completed")
    return {"messages": [analysis_response]}


# In[26]:


## add node to the workflow 


# assemble agent
builder = StateGraph(UnifiedResearchState)

# content extraction
builder.add_node( "llm", llm_node)
builder.add_node( "tool_node_chunk_selection_exec", tool_node_chunk_selection_exec)
builder.add_node( "extract_content", extract_content)
builder.add_node( "generate_risk_assessment", generate_risk_assessment)
builder.add_node( "filter_relevant_content",filter_relevant_content)
  # the node DOES wait for all incoming edges to complete before

# hyde generation
builder.add_node( "create_journalists", create_journalists )
builder.add_node( "generate_hyde_document", generate_hyde_document )

# estimate score 
builder.add_node( "assign_score_vs_hyde" ,  assign_score_vs_hyde )

# evidence assessment 
builder.add_node( "extract_evidence_claims", extract_evidence_claims)

# logic 

# content extraction
builder.add_edge(START, "llm")
builder.add_edge( "llm", "tool_node_chunk_selection_exec")
builder.add_edge( "tool_node_chunk_selection_exec", "extract_content")
builder.add_edge( "extract_content", "assign_score_vs_hyde")  # Then end

# article writer
builder.add_edge( "llm" , "create_journalists")
builder.add_edge( "create_journalists" , "generate_hyde_document")
builder.add_edge( "generate_hyde_document" , "assign_score_vs_hyde" )
builder.add_edge( "assign_score_vs_hyde" , "filter_relevant_content" )
builder.add_edge( "filter_relevant_content" , "extract_evidence_claims" )
builder.add_edge( "extract_evidence_claims" , "generate_risk_assessment" )
builder.add_edge("generate_risk_assessment", END)  # Then end

# assemble agent
graph = builder.compile( )

mermaid_code = graph.get_graph(xray=True).draw_mermaid()
render_mermaid_graph(mermaid_code)



# In[ ]:


# test 
# Moldretail
# Lukoil
# Finpar Invest
# trans oil
# danube logistics

# try different search algoriths
# LLM as local 
# juridical cases in pdf , justice md

initial_message = [ HumanMessage(content = "Danube Logistics" )  ]
results = graph.invoke(   
               {
                    "messages" : initial_message,
                    "max_journalists": 10,
                    "journalists": [],  # Will be populated by create_journalists node
                    "hyde_list": []     # Will be populated by generate_hyde_document node
                    
                } )
results 


# In[ ]:


summary_res = results.get("messages", None)
print( summary_res[-1].content )


# In[ ]:


results


# 
