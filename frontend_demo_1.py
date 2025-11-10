#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st

# Page config
st.set_page_config(
    page_title="Company Investigation Search Engine",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        font-size: 1.1rem;
    }
    .output-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🔍 Company Investigation Search Engine</div>', unsafe_allow_html=True)

# Input Section
st.markdown('<div class="section-header">Search Parameters</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    company_name = st.text_input(
        "Company Name",
        placeholder="Enter company name...",
        help="Enter the name of the company you want to investigate"
    )

with col2:
    investigation_subject = st.text_input(
        "Subject of Investigation",
        placeholder="e.g., Financial fraud, Money laundering, Sanctions...",
        help="Specify what you're investigating"
    )

# Language Selection
st.markdown('<div class="section-header">Search Languages</div>', unsafe_allow_html=True)

languages = st.multiselect(
    "Select one or more languages for search",
    options=["English", "Russian", "German", "French", "Romanian"],
    default=["English"],
    help="Choose languages to search in. Multiple selections will broaden your search."
)

# Search Button
st.markdown("<br>", unsafe_allow_html=True)
search_button = st.button("🔎 Start Search", type="primary")

# Output Section (only shown after search button is clicked)
if search_button:
    if not company_name:
        st.error("⚠️ Please enter a company name to search")
    elif not investigation_subject:
        st.error("⚠️ Please enter a subject of investigation")
    elif not languages:
        st.error("⚠️ Please select at least one language")
    else:
        st.success(f"✅ Searching for '{company_name}' - Investigation: '{investigation_subject}' in {', '.join(languages)}")
        
        # Divider
        st.markdown("---")
        
        # Create two columns for outputs
        output_col1, output_col2 = st.columns([1, 1])
        
        # Collected Links Section
        with output_col1:
            st.markdown('<div class="section-header">📎 Collected Links</div>', unsafe_allow_html=True)
            
            with st.container():
                st.markdown('<div class="output-box">', unsafe_allow_html=True)
                
                # Demo placeholder links
                st.markdown("""
                **Search Results:**
                
                1. [Financial Times - Company Profile](#)
                   - Source: English
                   - Date: 2024-01-15
                
                2. [Reuters Investigation Report](#)
                   - Source: English
                   - Date: 2024-02-20
                
                3. [Handelsblatt - Financial News](#)
                   - Source: German
                   - Date: 2024-01-28
                
                4. [Le Monde - Business Section](#)
                   - Source: French
                   - Date: 2024-03-10
                
                5. [Official Company Registry](#)
                   - Source: Romanian
                   - Date: 2024-02-05
                
                ---
                **Total Links Found:** 5
                """)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Summary Section
        with output_col2:
            st.markdown('<div class="section-header">📋 Summary</div>', unsafe_allow_html=True)
            
            with st.container():
                st.markdown('<div class="output-box">', unsafe_allow_html=True)
                
                # Demo placeholder summary
                st.markdown(f"""
                **Investigation Summary for: {company_name}**
                
                **Subject:** {investigation_subject}
                
                **Key Findings:**
                - Multiple news sources identified across {len(languages)} language(s)
                - Investigation spans financial records and news articles
                - Cross-border activities detected in European markets
                - Regulatory filings found in multiple jurisdictions
                
                **Risk Assessment:**
                - Medium to High risk indicators identified
                - Requires further detailed investigation
                - Multiple data sources corroborate findings
                
                **Recommended Actions:**
                1. Review detailed financial records
                2. Conduct deeper due diligence
                3. Consult with legal experts
                4. Monitor ongoing developments
                
                ---
                **Languages Searched:** {', '.join(languages)}
                
                **Search Completed:** Successfully
                """)
                
                st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">'
    '🔍 Demo Search Engine Interface | For demonstration purposes only'
    '</div>',
    unsafe_allow_html=True
)

