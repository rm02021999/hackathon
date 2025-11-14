import streamlit as st
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import tempfile
import os
import httpx
from dotenv import load_dotenv

# Load ENV variables
load_dotenv()

tiktoken_cache_dir = "tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

client = httpx.Client(verify=False)

# ------------------------------
# LLM CONFIG
# ------------------------------
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

# ------------------------------
# STREAMLIT CONFIG
# ------------------------------
st.set_page_config(
    page_title="AI Agent for IoT Network Event Explanation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# SIDEBAR NAVIGATION
# ------------------------------
with st.sidebar:
    st.title("⚙️ Navigation")

    selected_page = st.radio(
        "Navigate to",
        [
            "IoT Event Interpreter",
            "Dashboard",
            "Event Explanation",
            "Device Status",
            "Logs"
        ]
    )

    st.markdown("---")
    st.subheader("🆕 New Feature Toggles")
    feature_1 = st.checkbox("Advanced Event Insights")
    feature_2 = st.checkbox("Root Cause Prediction")
    feature_3 = st.checkbox("Auto-Generated Fix Suggestions")
    feature_4 = st.checkbox("Impact Heatmap Visualization")
    feature_5 = st.checkbox("Event Simulation")

# ------------------------------
# PAGE 1 — IoT RAG MAIN ENGINE
# ------------------------------
if selected_page == "IoT Event Interpreter":

    st.title("🧠 AI Agent for IoT Network Event Interpretation (AIOps Enhanced)")

    upload_file = st.file_uploader(
        "Upload IoT network logs / device documentation (PDF)",
        type="pdf"
    )

    # Synthetic dataset
    synthetic_iot_data = """
    Device_ID: TempSensor_108
    Event: High Packet Loss
    Timestamp: 2025-02-10 14:21:44
    RSSI: -88dBm
    SNR: 4.2 dB
    Errors: 231/min

    Event: Sudden Battery Drop (12% → 2%)
    Impact: Intermittent connectivity

    Device_ID: SmartMeter_42
    Event: Authentication Failure
    Details:
     - Invalid signature
     - 14 retries in 5 minutes
    """

    raw_text = ""

    # STEP 1 — Text Extraction
    if upload_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(upload_file.read())
            temp_file_path = temp_file.name

        raw_text = extract_text(temp_file_path)

    else:
        st.warning("No PDF uploaded. Using synthetic IoT event dataset.")
        raw_text = synthetic_iot_data

    # STEP 2 — Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(raw_text)

    # STEP 3 — Embedding
    with st.spinner("Indexing IoT knowledge base..."):
        vectordb = Chroma.from_texts(
            chunks,
            embedding_model,
            persist_directory="./iot_chroma_index"
        )
        vectordb.persist()

    retriever = vectordb.as_retriever()

    # STEP 4 — RAG CHAIN
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # MAIN QUERY
    st.subheader("Ask your IoT Event Question")
    query = st.text_input("Example: “Explain high packet loss in TempSensor_108”")

    if query:
        with st.spinner("Generating explanation..."):
            base_prompt = f"""
You are an advanced AIOps IoT Analysis Agent.

Using the provided IoT logs + general IoT knowledge:
1. Provide a simple explanation.
2. Identify root causes.
3. Estimate probability (%) for each cause.
4. Give risk level (Low/Med/High).
5. Provide recommended actions (playbook steps).
6. Compute a health score (0–100).
7. Create a timeline summary (if timestamps exist).
8. Classify event category:
   - RF Issue
   - Battery Issue
   - Firmware Issue
   - Security Issue
   - Network Congestion
9. Add “What the operator should monitor next”.

User question: {query}
"""
            result = rag_chain.invoke(base_prompt)

        st.subheader("📘 IoT Event Explanation")
        st.write(result["result"])

        st.markdown("### 🔍 Retrieved Context")
        for doc in result["source_documents"]:
            st.caption(doc.page_content[:600] + "...")

        # Scenario analyzer
        st.markdown("---")
        st.markdown("### 🧪 'What-If' Scenario Analyzer")
        scenario = st.text_input(
            "Try: 'What if RSSI improves to -70dBm?' or 'What if battery replaced?'"
        )
        if scenario:
            with st.spinner("Analyzing scenario..."):
                scenario_prompt = f"""
Interpret this hypothetical scenario based on IoT logs:
{scenario}

Explain how this will affect:
- Packet loss
- Reliability
- Device health score
- Network stability
"""
                scenario_result = llm.invoke(scenario_prompt)
            st.write(scenario_result.content)

# ------------------------------
# PAGE 2 — Dashboard
# ------------------------------
elif selected_page == "Dashboard":
    st.title("📊 Dashboard Overview")
    st.write("High-level summary of IoT devices, network events, and alerts.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total IoT Devices", "128", "+7 today")
    with col2:
        st.metric("Alerts Triggered", "14", "🔽 -3 vs yesterday")
    with col3:
        st.metric("Network Health", "92%", "Stable")

    st.subheader("Recent Events")
    sample_events = [
        {"Device": "TempSensor-4", "Event": "High Latency", "Severity": "Medium"},
        {"Device": "Cam-22", "Event": "Packet Loss", "Severity": "High"},
        {"Device": "DoorLock-3", "Event": "Authentication Timeout", "Severity": "Low"},
        {"Device": "MotionNode-7", "Event": "Intermittent Disconnect", "Severity": "Medium"},
    ]
    st.table(sample_events)

# ------------------------------
# PAGE 3 — Manual Explanation UI
# ------------------------------
elif selected_page == "Event Explanation":
    st.title("📝 Natural-Language Event Explanation")

    event_input = st.text_area(
        "Enter network event summary",
        placeholder="Example: Device TempSensor-4 showing high latency and intermittent data loss..."
    )

    if st.button("Generate Explanation"):
        if event_input.strip() == "":
            st.warning("Please enter an event summary.")
        else:
            st.success("Generated Explanation:")
            st.write(
                f"**Cause (Probable):** The device is experiencing unstable wireless connectivity.\n\n"
                f"**Impact:** Data packets may be delayed or dropped.\n\n"
                f"**Recommended Action:** Check signal strength & router interference."
            )

# ------------------------------
# PAGE 4 — Device Status
# ------------------------------
elif selected_page == "Device Status":
    st.title("📡 Live IoT Device Status")

    device = st.selectbox(
        "Select Device",
        ["TempSensor-4", "Cam-22", "MotionNode-7", "AirQualityNode-3"]
    )

    st.metric("Status", "Online")
    st.metric("Signal Strength", "-67 dBm")
    st.metric("Battery", "84%")

# ------------------------------
# PAGE 5 — Logs
# ------------------------------
elif selected_page == "Logs":
    st.title("📁 System & Network Logs")

    uploaded_file = st.file_uploader("Upload Log File")

    if uploaded_file:
        st.success("File uploaded!")
        content = uploaded_file.read().decode("utf-8")[:5000]
        st.text_area("Preview", content, height=300)

# ------------------------------
# RENDER NEW FEATURES
# ------------------------------
st.markdown("---")
st.header("⚡ Enabled Features")

if feature_1:
    st.success("✔ Advanced Event Insights Enabled")
if feature_2:
    st.success("✔ Root Cause Prediction Enabled")
if feature_3:
    st.success("✔ Auto-Generated Fix Suggestions Enabled")
if feature_4:
    st.success("✔ Impact Heatmap Visualization Enabled (placeholder)")
if feature_5:
    st.success("✔ Event Simulation Feature Enabled")
