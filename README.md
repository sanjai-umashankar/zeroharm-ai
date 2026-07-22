# 🛡️ ZeroHarm AI

**AI-Powered Industrial Safety Intelligence for Zero-Harm Operations**

A multi-agent industrial safety intelligence platform that fuses real-time IoT/SCADA sensor data, digital work permits, CCTV feeds, shift logs, maintenance records, and regulatory knowledge into a predictive safety layer — detecting **compound risks** hours before they escalate.

Aligned with **OISD-STD-105**, **Factory Act 1948**, and **DGMS regulations**.

> 🔗 **Live Demo:** [add your Streamlit Cloud URL here]
> 🎥 **Demo Video:** [add your video link here]
> 📄 **Problem Statement:** AI-Powered Industrial Safety Intelligence for Zero-Harm Operations

---

## 📌 The Problem

Industrial safety systems today are **reactive** — a sensor alarms only after a threshold is already breached. But the most dangerous conditions are **compound**: a slowly rising toxic gas level combined with an active Hot Work Permit in the same zone is far more dangerous than either alone, and no current single-sensor system catches that in time.

## 💡 The Solution

ZeroHarm AI correlates sensor trends, active permits, worker locations, and zone classification in real time to surface compound hazards early — with every alert explained and traced back to the specific regulatory clause it violates.

---

## ✨ Key Features

| Module | Description |
|---|---|
| 📊 **Live Plant Overview** | Facility-wide risk score (0–100), active alerts, compound-risk indicator |
| 🗺️ **Geospatial Safety Heatmap** | Interactive plant-layout map with live risk zones, worker markers, permit overlays |
| ⚠️ **Compound Risk Detector** | Core innovation — correlates gas/temp trends with permit type + zone classification, with full "Why?" explainability |
| 📋 **Digital Permit Intelligence** | Continuously re-validates active work permits against live zone conditions |
| 💬 **Incident Pattern RAG Chat** | Natural-language Q&A grounded in OISD/DGMS regulations and historical incidents |
| 🎥 **CCTV Analytics Feed** | Simulated PPE compliance and unauthorized zone entry detection |
| 🚨 **Emergency Orchestrator** | Human-confirmed alert, evacuation-protocol simulation, and auto-drafted incident reports |
| ✅ **Compliance Audit** | Live deviation log mapped directly to OISD/Factory Act/DGMS clauses |

---

## 🏗️ System Architecture

ZeroHarm AI runs as a set of cooperating agents rather than a single model, so every layer of reasoning stays explainable:

- **Sensor Agent** — evaluates live gas/temp/pressure readings against zone-specific thresholds
- **Permit Agent** — validates active permits (type, expiry, issuance conditions)
- **Compound Risk Agent** — fuses sensor trend + permit type + zone classification to catch correlated hazards
- **Scoring Agent** — produces an explainable 0–100 risk score with confidence, weighting compound risk higher than isolated single-sensor flags
- **Geospatial Agent** — renders zones, workers, and permits onto the live plant map
- **RAG Agent** — grounds chat answers in regulatory text and historical incident records

> **Design principle:** compound risk always takes precedence over an isolated single-sensor threshold, and every action requires human confirmation.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Dashboard | Streamlit |
| Geospatial Visualization | Folium / Leaflet |
| Data Visualization | Plotly |
| Computer Vision (roadmap) | YOLOv8, OpenCV |
| IoT / SCADA (roadmap) | MQTT, OPC-UA |
| Risk Engine | Rule-based multi-agent fusion (+ ML roadmap: XGBoost/LSTM) |
| Knowledge Layer / RAG | Keyword retrieval (MVP) → FAISS/pgvector + LLM (production) |
| Data | PostgreSQL + InfluxDB (roadmap) |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation
```bash
git clone https://github.com/yourusername/zeroharm-ai.git
cd zeroharm-ai
python -m venv venv
venv\Scripts\Activate.ps1      # Windows PowerShell
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### Run the app
```bash
streamlit run zeroharm_app.py
```

The app will open at `http://localhost:8501`.

---

## 📸 Screenshots

<img width="1600" height="950" alt="01-live-plant-overview" src="https://github.com/user-attachments/assets/3ea35acc-6893-4071-ace5-ef14c39cda6a" />
<img width="1600" height="900" alt="02-compound-risk-detector" src="https://github.com/user-attachments/assets/3b5a0e38-abc0-41ed-8817-d9646d7b0614" />
<img width="1600" height="950" alt="03-digital-permit-intelligence" src="https://github.com/user-attachments/assets/cac34813-6ce3-4d02-8ced-6dc4abd6e788" />
---

## 🧭 Demo Scenario

Zone **Z1 — Coke Oven Battery** has a scripted CO gas ramp-up built into the simulation. Combined with an auto-injected active Hot Work Permit in the same zone, it triggers a **COMPOUND risk** — escalating the score to CRITICAL. Click "Manual Refresh" a few times on the Live Overview page to watch it climb, then check the **Compound Risk Detector** page for the full explanation.

---

## 🗺️ Roadmap

1. **MVP** — multi-agent risk engine, dashboard, heatmap, and permit intelligence on simulated data ✅
2. **Field Integration** — live SCADA/IoT feeds via MQTT/OPC-UA + real permit system API
3. **Computer Vision** — YOLOv8-based PPE and confined-space intrusion detection on CCTV
4. **ML-Augmented Scoring** — XGBoost/LSTM anomaly model layered on the explainable rule engine
5. **Enterprise RAG** — vector DB (FAISS/pgvector) + LLM grounded on the full OISD/DGMS/Factory Act corpus

---

## 📜 Regulatory Alignment

| Standard | How ZeroHarm AI Aligns |
|---|---|
| OISD-STD-105 (Work Permit System) | Digital Permit Intelligence validates permits against live conditions |
| OISD-STD-116 (LPG Storage) | Blocks Hot Work Permits in explosive-atmosphere zones by design |
| Factory Act 1948, Sec. 35 & 36 | Confined-space/hazardous-process rules embedded in the Compound Risk Agent |
| DGMS Confined Space Circular | Continuous atmospheric trend monitoring, not just point-in-time checks |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

Built for the problem statement: *"AI-Powered Industrial Safety Intelligence for Zero-Harm Operations"*
