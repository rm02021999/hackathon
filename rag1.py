import streamlit as st
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import tempfile
import os
import httpx
import tiktoken
from dotenv import load_dotenv
from datetime import datetime
import json

# Load variables from .env file
load_dotenv()

# Please modify the "tiktoken_cache_dir" to the directory wherever you are placing your "tiktoken_cache" folder
tiktoken_cache_dir = "tiktoken_cache"

os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
assert os.path.exists(os.path.join(tiktoken_cache_dir, "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"))

client = httpx.Client(verify=False)


# LLM setup (IMPORTANT)
# Please update the base_url, model, api_key as specified below.
llm = ChatOpenAI(
    base_url = os.getenv("api_endpoint"),
    api_key = os.getenv("api_key"),
    model="azure/genailab-maas-gpt-35-turbo",
    http_client=client
)


# EMBEDDING setup (IMPORTANT)
# Please update the base_url, model, api_key as specified below.
embedding_model = OpenAIEmbeddings(
    base_url = os.getenv("api_endpoint"),
    api_key = os.getenv("api_key"),
    model="azure/genailab-maas-text-embedding-3-large",
    http_client=client
)

st.set_page_config(page_title="Meeting Minutes Generator", page_icon="📝", layout="wide")
st.title("AI-Powered Meeting Minutes Generator (MOM)")
st.markdown("Upload a meeting transcript (PDF or TXT) to automatically generate comprehensive meeting minutes")

# Sidebar for meeting metadata
with st.sidebar:
    st.header("Meeting Information")
    meeting_title = st.text_input("Meeting Title", "Team Sync Meeting")
    meeting_date = st.date_input("Meeting Date", datetime.now())
    meeting_time = st.text_input("Meeting Time", "10:00 AM - 11:00 AM")
    attendees = st.text_area("Attendees (comma-separated)", "John Doe, Jane Smith, Bob Johnson")
    facilitator = st.text_input("Facilitator/Chair", "John Doe")
    
    st.divider()
    st.header("MOM Options")
    include_verbatim = st.checkbox("Include Key Quotes", value=True)
    include_sentiment = st.checkbox("Include Sentiment Analysis", value=False)
    export_format = st.selectbox("Export Format", ["Markdown", "Plain Text", "JSON"])

# File upload section
col1, col2 = st.columns([2, 1])
with col1:
    upload_file = st.file_uploader("Upload Meeting Transcript", type=["pdf", "txt"])
with col2:
    st.info("**Tip**: Upload a clear transcript for best results")

if upload_file:
    # Process the uploaded file
    if upload_file.type == "application/pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(upload_file.read())
            temp_file_path = temp_file.name
        # Step 1: Extract text
        raw_text = extract_text(temp_file_path)
    else:  # txt file
        raw_text = upload_file.read().decode("utf-8")
    
    # Display preview of transcript
    with st.expander("View Transcript Preview"):
        st.text_area("Transcript", raw_text[:1000] + "..." if len(raw_text) > 1000 else raw_text, height=200)

    # Step 2: Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(raw_text)

    # Step 3: Embed and store in Chroma
    with st.spinner("Indexing meeting transcript..."):
        vectordb = Chroma.from_texts(chunks, embedding_model, persist_directory="./chroma_index")
        vectordb.persist()

    # Step 4: RAG QA Chain
    retriever = vectordb.as_retriever()
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    
    # Step 5: Generate comprehensive MOM
    st.divider()
    st.header("Generated Meeting Minutes")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📌 Summary", 
        "🎯 Key Decisions", 
        "✅ Action Items", 
        "📊 Discussion Points",
        "❓ Open Issues",
        "📄 Full MOM"
    ])
    
    # Generate different sections of MOM
    with tab1:
        st.subheader("Executive Summary")
        summary_prompt = """Based on the meeting transcript, provide a concise executive summary that includes:
        1. Meeting purpose and objectives
        2. Key outcomes
        3. Overall sentiment and tone
        Keep it brief (3-5 sentences)."""
        
        with st.spinner("Generating summary..."):
            summary_result = rag_chain.invoke(summary_prompt)
        st.write(summary_result['result'])
    
    with tab2:
        st.subheader("Key Decisions Made")
        decisions_prompt = """Extract all key decisions made during the meeting. For each decision:
        1. State the decision clearly
        2. Mention who made or agreed to the decision
        3. Note any conditions or timeline
        Format as a numbered list."""
        
        with st.spinner("Extracting decisions..."):
            decisions_result = rag_chain.invoke(decisions_prompt)
        st.write(decisions_result['result'])
    
    with tab3:
        st.subheader("Action Items")
        actions_prompt = """Extract all action items from the meeting. For each action item, provide:
        1. The specific task or action
        2. Person responsible (owner)
        3. Deadline or timeline (if mentioned)
        4. Priority level (if indicated)
        Format as a clear numbered list with sub-points."""
        
        with st.spinner("Identifying action items..."):
            actions_result = rag_chain.invoke(actions_prompt)
        st.write(actions_result['result'])
        
        # Create downloadable action items table
        st.markdown("---")
        st.markdown("**Action Items Tracker** (Copy to your project management tool)")
        st.code("""
| # | Action Item | Owner | Deadline | Status |
|---|-------------|-------|----------|--------|
| 1 |             |       |          | Pending|
| 2 |             |       |          | Pending|
        """)
    
    with tab4:
        st.subheader("Discussion Points")
        discussion_prompt = """Summarize the main discussion points from the meeting. Group them by topic or theme. 
        For each discussion point include:
        1. The topic discussed
        2. Key perspectives shared
        3. Any consensus or disagreements
        4. Relevant context"""
        
        with st.spinner("Analyzing discussions..."):
            discussion_result = rag_chain.invoke(discussion_prompt)
        st.write(discussion_result['result'])
        
        if include_verbatim:
            st.markdown("---")
            st.markdown("**Key Quotes**")
            quotes_prompt = "Extract 3-5 significant quotes from the meeting that capture important points or decisions."
            with st.spinner("Finding key quotes..."):
                quotes_result = rag_chain.invoke(quotes_prompt)
            st.info(quotes_result['result'])
    
    with tab5:
        st.subheader("Open Issues & Follow-ups")
        issues_prompt = """Identify any open issues, unresolved questions, or items requiring follow-up. 
        For each item specify:
        1. The issue or question
        2. Why it remains open
        3. Next steps or who will follow up"""
        
        with st.spinner("Identifying open issues..."):
            issues_result = rag_chain.invoke(issues_prompt)
        st.write(issues_result['result'])
    
    with tab6:
        st.subheader("Complete Meeting Minutes")
        
        # Compile full MOM
        full_mom = f"""
# MEETING MINUTES

**Meeting Title:** {meeting_title}
**Date:** {meeting_date.strftime('%B %d, %Y')}
**Time:** {meeting_time}
**Facilitator:** {facilitator}
**Attendees:** {attendees}

---

## 1. EXECUTIVE SUMMARY
{summary_result['result']}

---

## 2. KEY DECISIONS
{decisions_result['result']}

---

## 3. ACTION ITEMS
{actions_result['result']}

---

## 4. DISCUSSION POINTS
{discussion_result['result']}

---

## 5. OPEN ISSUES & FOLLOW-UPS
{issues_result['result']}

---

**Minutes Prepared By:** AI-Powered MOM Generator
**Date Generated:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
        """
        
        if include_sentiment:
            sentiment_prompt = "Analyze the overall sentiment and tone of the meeting. Was it collaborative, tense, productive, etc.?"
            with st.spinner("Analyzing sentiment..."):
                sentiment_result = rag_chain.invoke(sentiment_prompt)
            full_mom += f"\n\n---\n\n## 6. MEETING SENTIMENT ANALYSIS\n{sentiment_result['result']}"
        
        st.markdown(full_mom)
        
        # Export options
        st.divider()
        st.subheader("📥 Export Meeting Minutes")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label="⬇️ Download as Markdown",
                data=full_mom,
                file_name=f"MOM_{meeting_title.replace(' ', '_')}_{meeting_date}.md",
                mime="text/markdown"
            )
        
        with col2:
            # Plain text version (remove markdown formatting)
            plain_text = full_mom.replace('#', '').replace('**', '').replace('*', '')
            st.download_button(
                label="⬇️ Download as Text",
                data=plain_text,
                file_name=f"MOM_{meeting_title.replace(' ', '_')}_{meeting_date}.txt",
                mime="text/plain"
            )
        
        with col3:
            # JSON version
            mom_json = {
                "meeting_info": {
                    "title": meeting_title,
                    "date": str(meeting_date),
                    "time": meeting_time,
                    "facilitator": facilitator,
                    "attendees": attendees.split(',')
                },
                "summary": summary_result['result'],
                "decisions": decisions_result['result'],
                "action_items": actions_result['result'],
                "discussions": discussion_result['result'],
                "open_issues": issues_result['result'],
                "generated_at": datetime.now().isoformat()
            }
            st.download_button(
                label="⬇️ Download as JSON",
                data=json.dumps(mom_json, indent=2),
                file_name=f"MOM_{meeting_title.replace(' ', '_')}_{meeting_date}.json",
                mime="application/json"
            )

    
    st.divider()
    st.header("🔍 Interactive Q&A")
    st.markdown("Ask specific questions about the meeting")
    
    user_question = st.text_input("Ask a question about the meeting:", 
                                  placeholder="e.g., What was discussed about the project timeline?")
    
    if user_question:
        with st.spinner("Finding answer..."):
            qa_result = rag_chain.invoke(user_question)
        st.success("**Answer:**")
        st.write(qa_result['result'])
        
        if qa_result.get('source_documents'):
            with st.expander("📚 View Source Context"):
                for i, doc in enumerate(qa_result['source_documents'][:3]):
                    st.markdown(f"**Source {i+1}:**")
                    st.text(doc.page_content[:300] + "...")

else:
    # Welcome screen with instructions
    st.info("""
    ### 👋 Welcome to the Meeting Minutes Generator!
    
    **How to use:**
    1. Fill in meeting information in the sidebar (left)
    2. Upload your meeting transcript (PDF or TXT format)
    3. Review and customize the generated minutes
    4. Download in your preferred format
    
    **Best practices for transcripts:**
    - Include speaker names when possible
    - Ensure clear formatting
    - Include timestamps if available
    - Remove any sensitive information before uploading
    """)
    
    # Example transcript section
    with st.expander("📝 See Example Transcript Format"):
        st.code("""
Meeting Transcript Example:

John (10:00 AM): Good morning everyone. Let's start with the project status update.

Sarah (10:02 AM): The development phase is 80% complete. We're on track for the Q4 launch.

Mike (10:05 AM): I have concerns about the testing timeline. We need at least 3 more weeks.

John (10:07 AM): Agreed. Let's adjust the timeline. Sarah, can you update the project plan?

Sarah (10:08 AM): Yes, I'll have it ready by Friday.

[ACTION ITEM]: Sarah to update project plan by Friday
[DECISION]: Extend testing phase by 3 weeks
        """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    <p>AI-Powered Meeting Minutes Generator | Powered by RAG & LangChain</p>
    <p>💡 Tip: For best results, ensure your transcript is clear and well-formatted</p>
</div>
""", unsafe_allow_html=True)