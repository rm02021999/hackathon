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
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------
# INIT
# ------------------------------------------------------
load_dotenv()
st.set_page_config(
    page_title="IoT Event AI Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

client = httpx.Client(verify=False)

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

# ------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Menu")

    selected_page = st.radio(
        "Navigate",
        [
            "IoT Event Interpreter",
            "Dashboard",
            "Device Simulation",
            "Event Explanation",
            "Device Status",
            "Logs"
        ]
    )

    st.markdown("---")
    st.subheader("🆕 Features")
    feature_charts = st.checkbox("Enable Analytics Charts", True)
    feature_sim = st.checkbox("Enable Device Simulation Engine", True)

# ------------------------------------------------------
# UTILITY — DEVICE SIMULATION ENGINE
# ------------------------------------------------------
def simulate_device_metrics():
    """Generate realistic IoT sensor network metrics."""
    return {
        "RSSI": random.randint(-95, -40),
        "SNR": round(random.uniform(2.0, 12.0), 2),
        "Battery": random.randint(5, 100),
        "PacketLoss": random.randint(0, 40),
        "Temp": random.randint(10, 70)
    }

# ------------------------------------------------------
# PAGE 1 — IoT Event Interpreter (RAG Engine)
# ------------------------------------------------------
if selected_page == "IoT Event Interpreter":

    st.title("🧠 AI Agent for IoT Network Event Interpretation")

    upload_file = st.file_uploader("Upload IoT Logs / PDF", type="pdf")

    synthetic_text = """
    Device_ID: TempSensor_108
    Event: High Packet Loss
    RSSI: -88 dBm
    Battery: 12%

    Device_ID: SmartMeter_42
    Event: Authentication Failure
    14 retries in 5 minutes.
    """

    if upload_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(upload_file.read())
            tmp_path = temp_file.name
        raw_text = extract_text(tmp_path)
    else:
        st.warning("Using synthetic IoT dataset")
        raw_text = synthetic_text

    # Text Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(raw_text)

    with st.spinner("Indexing knowledge base..."):
        vectordb = Chroma.from_texts(
            chunks,
            embedding_model,
            persist_directory="./iot_vector_db"
        )
        vectordb.persist()

    retriever = vectordb.as_retriever()

    rag = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    st.subheader("Ask IoT Question")
    query = st.text_input("Example: What causes packet loss in TempSensor_108?")

    if query:
        with st.spinner("Thinking..."):
            prompt = f"""
Explain IoT event based on logs:

1. Root cause analysis  
2. Probability (%)  
3. Risk level  
4. Recommended actions  
5. Timeline summary  
6. Health score  
7. What to monitor next  

User query: {query}
"""
            result = rag.invoke(prompt)

        st.write("### 📘 Explanation")
        st.write(result["result"])

        st.write("### 📎 Context Used")
        for doc in result["source_documents"]:
            st.caption(doc.page_content[:500] + "...")

# ------------------------------------------------------
# PAGE 2 — DASHBOARD WITH CHARTS
# ------------------------------------------------------
elif selected_page == "Dashboard":

    st.title("📊 IoT Analytics Dashboard")

    # Example data for charts
    devices = ["Sensor A", "Sensor B", "Sensor C", "Sensor D"]
    rssi_values = np.random.randint(-90, -40, size=4)
    packet_loss = np.random.randint(0, 35, size=4)

    # RSSI Line Chart
    if feature_charts:
        st.subheader("📡 RSSI Trend")

        rssi_series = pd.DataFrame({
            "Time": range(10),
            "RSSI": [random.randint(-90, -50) for _ in range(10)]
        })

        fig, ax = plt.subplots()
        ax.plot(rssi_series["Time"], rssi_series["RSSI"])
        ax.set_ylabel("RSSI (dBm)")
        ax.set_xlabel("Time")
        ax.set_title("RSSI Trend Over Time")
        st.pyplot(fig)

        # Packet Loss Bar Chart
        st.subheader("📦 Packet Loss Comparison")

        fig2, ax2 = plt.subplots()
        ax2.bar(devices, packet_loss)
        ax2.set_ylabel("Packet Loss (%)")
        st.pyplot(fig2)

        # Heatmap (Simulated)
        st.subheader("🔥 Network Health Heatmap")
        heatmap = np.random.rand(6, 6)

        fig3, ax3 = plt.subplots()
        cax = ax3.imshow(heatmap, cmap="coolwarm")
        fig3.colorbar(cax)
        st.pyplot(fig3)

# ------------------------------------------------------
# PAGE 3 — DEVICE SIMULATION ENGINE
# ------------------------------------------------------
elif selected_page == "Device Simulation":

    st.title("🧪 IoT Device Simulation Engine")

    if feature_sim:

        col1, col2 = st.columns(2)

        with col1:
            num_devices = st.slider("Select number of devices:", 1, 20, 5)

        if st.button("Run Simulation"):
            st.success("Simulation started")

            sim_data = []
            for i in range(num_devices):
                metrics = simulate_device_metrics()
                sim_data.append({
                    "Device": f"Device_{i+1}",
                    **metrics
                })

            df = pd.DataFrame(sim_data)
            st.write("### Simulated Device Metrics")
            st.dataframe(df)

            # Generate Alerts
            st.write("### 🚨 Alerts")
            for row in df.itertuples():
                if row.PacketLoss > 25:
                    st.error(f"{row.Device}: High Packet Loss ({row.PacketLoss}%)")
                if row.Battery < 15:
                    st.warning(f"{row.Device}: Low Battery ({row.Battery}%)")
                if row.RSSI < -85:
                    st.warning(f"{row.Device}: Weak Signal ({row.RSSI} dBm)")

    else:
        st.warning("Enable simulation engine from sidebar")

# ------------------------------------------------------
# PAGE 4 — Manual Explanation
# ------------------------------------------------------
elif selected_page == "Event Explanation":

    st.title("📝 Event Explanation")
    event = st.text_area("Enter event details")

    if st.button("Explain"):
        if event:
            out = llm.invoke(f"Explain this IoT event in simple terms: {event}")
            st.write(out.content)
        else:
            st.warning("Enter event first")

# ------------------------------------------------------
# PAGE 5 — Device Status
# ------------------------------------------------------
elif selected_page == "Device Status":

    st.title("📡 Device Status Viewer")
    device = st.selectbox("Choose Device", ["Sensor A", "Sensor B", "Sensor C"])

    metrics = simulate_device_metrics()
    st.metric("RSSI", f"{metrics['RSSI']} dBm")
    st.metric("Battery", f"{metrics['Battery']}%")
    st.metric("Packet Loss", f"{metrics['PacketLoss']}%")

# ------------------------------------------------------
# PAGE 6 — Logs Viewer
# ------------------------------------------------------
elif selected_page == "Logs":

    st.title("📁 Upload Logs")
    file = st.file_uploader("Upload log file")

    if file:
        st.text_area("Preview", file.read().decode("utf-8")[:5000])
