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

# Load environment variables
load_dotenv()

# TikToken cache setup
tiktoken_cache_dir = "tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
assert os.path.exists(os.path.join(tiktoken_cache_dir, "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"))

client = httpx.Client(verify=False)

# -----------------------------
# 1️⃣ LLM and Embeddings Setup
# -----------------------------
llm = ChatOpenAI(
    base_url=os.getenv("api_endpoint"),
    api_key=os.getenv("api_key"),
    model="azure/genailab-maas-gpt-35-turbo",
    http_client=client
)

embedding_model = OpenAIEmbeddings(
    base_url=os.getenv("api_endpoint"),
    api_key=os.getenv("api_key"),
    model="azure/genailab-maas-text-embedding-3-large",
    http_client=client
)

# Streamlit UI setup
st.set_page_config(page_title="Meeting Minutes Generator", page_icon="📝", layout="wide")
st.title("📝 AI-Powered Meeting Minutes Generator (MOM)")
st.markdown("Upload a meeting transcript (PDF or TXT) to automatically generate comprehensive meeting minutes")

# -----------------------------
# 2️⃣ Sidebar for Meeting Info
# -----------------------------
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

# -----------------------------
# 3️⃣ File Upload Section
# -----------------------------
col1, col2 = st.columns([2, 1])
with col1:
    upload_file = st.file_uploader("Upload Meeting Transcript", type=["pdf", "txt"])
with col2:
    st.info("💡 **Tip**: Upload a clear transcript for best results")

if upload_file:
    # Step 1: Extract text
    if upload_file.type == "application/pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(upload_file.read())
            temp_file_path = temp_file.name
        raw_text = extract_text(temp_file_path)
    else:
        raw_text = upload_file.read().decode("utf-8")
    
    with st.expander("📄 View Transcript Preview"):
        st.text_area("Transcript", raw_text[:1000] + "..." if len(raw_text) > 1000 else raw_text, height=200)

    # Step 2: Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(raw_text)

    # Step 3: Create vector database
    with st.spinner("🔄 Indexing meeting transcript..."):
        vectordb = Chroma.from_texts(chunks, embedding_model, persist_directory="./chroma_index")
        vectordb.persist()

    # -----------------------------
    # 4️⃣ Guardrailed Retrieval QA
    # -----------------------------
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    # Guardrail + Anti-Hallucination Function
    def safe_rag_answer(question, rag_chain, retriever):
        docs = retriever.get_relevant_documents(question)
        
        # Guardrail: if no docs found
        if not docs or all(len(d.page_content.strip()) == 0 for d in docs):
            return {
                "result": "⚠️ Sorry, I couldn't find relevant information in the document to answer your question.",
                "source_documents": []
            }

        grounding_prompt = f"""
        You are an AI assistant specialized in extracting information ONLY from the provided meeting transcript.
        Use ONLY the retrieved text to answer the question below.
        If the answer cannot be found or inferred directly from the transcript, reply exactly with:
        "⚠️ The information is not available in the provided meeting transcript."

        Question: {question}

        Transcript context:
        {''.join([doc.page_content for doc in docs])}

        Answer concisely and factually using only the given text.
        """

        result = rag_chain.invoke(grounding_prompt)
        result["source_documents"] = docs
        return result

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # -----------------------------
    # 5️⃣ Generate MOM Sections
    # -----------------------------
    st.divider()
    st.header("📋 Generated Meeting Minutes")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📌 Summary", "🎯 Key Decisions", "✅ Action Items",
        "📊 Discussion Points", "❓ Open Issues", "📄 Full MOM"
    ])

    with tab1:
        st.subheader("Executive Summary")
        prompt = """Based on the meeting transcript, provide a concise executive summary that includes:
        1. Meeting purpose and objectives
        2. Key outcomes
        3. Overall sentiment and tone
        Keep it brief (3-5 sentences)."""
        with st.spinner("Generating summary..."):
            summary_result = safe_rag_answer(prompt, rag_chain, retriever)
        st.write(summary_result["result"])

    with tab2:
        st.subheader("Key Decisions Made")
        prompt = """Extract all key decisions made during the meeting."""
        with st.spinner("Extracting decisions..."):
            decisions_result = safe_rag_answer(prompt, rag_chain, retriever)
        st.write(decisions_result["result"])

    with tab3:
        st.subheader("Action Items")
        prompt = """Extract all action items from the meeting transcript."""
        with st.spinner("Identifying action items..."):
            actions_result = safe_rag_answer(prompt, rag_chain, retriever)
        st.write(actions_result["result"])

    with tab4:
        st.subheader("Discussion Points")
        prompt = """Summarize the main discussion points grouped by theme."""
        with st.spinner("Analyzing discussions..."):
            discussion_result = safe_rag_answer(prompt, rag_chain, retriever)
        st.write(discussion_result["result"])

    with tab5:
        st.subheader("Open Issues & Follow-ups")
        prompt = """Identify unresolved issues or pending follow-ups from the meeting."""
        with st.spinner("Identifying open issues..."):
            issues_result = safe_rag_answer(prompt, rag_chain, retriever)
        st.write(issues_result["result"])

    with tab6:
        st.subheader("Complete Meeting Minutes")

        full_mom = f"""
# MEETING MINUTES

**Meeting Title:** {meeting_title}
**Date:** {meeting_date.strftime('%B %d, %Y')}
**Time:** {meeting_time}
**Facilitator:** {facilitator}
**Attendees:** {attendees}

---

## EXECUTIVE SUMMARY
{summary_result['result']}

---

## KEY DECISIONS
{decisions_result['result']}

---

## ACTION ITEMS
{actions_result['result']}

---

## DISCUSSION POINTS
{discussion_result['result']}

---

## OPEN ISSUES & FOLLOW-UPS
{issues_result['result']}

---

**Minutes Prepared By:** AI-Powered MOM Generator  
**Date Generated:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
"""
        st.markdown(full_mom)

    # -----------------------------
    # 6️⃣ Interactive Guardrailed Q&A
    # -----------------------------
    st.divider()
    st.header("🔍 Interactive Q&A")
    st.markdown("Ask specific questions about the meeting")

    user_question = st.text_input(
        "Ask a question about the meeting:",
        placeholder="e.g., What was discussed about the project timeline?"
    )

    if user_question:
        with st.spinner("Finding answer from document..."):
            qa_result = safe_rag_answer(user_question, rag_chain, retriever)

        st.success("**Answer:**")
        st.write(qa_result["result"])

        if qa_result.get("source_documents"):
            with st.expander("📚 View Source Context"):
                for i, doc in enumerate(qa_result["source_documents"][:3]):
                    st.markdown(f"**Source {i+1}:**")
                    st.text(doc.page_content[:300] + "...")

else:
    st.info("""
    ### 👋 Welcome to the Meeting Minutes Generator!
    1. Fill in meeting info in the sidebar  
    2. Upload a PDF or TXT transcript  
    3. Review generated minutes  
    4. Download results  
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    <p>AI-Powered Meeting Minutes Generator | Powered by RAG & LangChain</p>
    <p>💡 For best results, ensure your transcript is clear and well-formatted</p>
</div>
""", unsafe_allow_html=True)
