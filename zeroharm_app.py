# ZeroHarm AI — Industrial Safety Intelligence Platform
# SINGLE-FILE MVP PROTOTYPE (all modules merged for convenience)
#
# Run:
#     pip install streamlit folium streamlit-folium plotly pandas numpy
#     streamlit run app.py
#
# Sections in this file:
#   1. DATA ENGINE       - simulated sensors, permits, workers, plant zones
#   2. RISK ENGINE        - multi-agent risk detection (single + compound risk fusion, explainability)
#   3. KNOWLEDGE BASE      - RAG-lite regulatory + incident retrieval
#   4. STREAMLIT APP       - dashboard UI (8 pages, professional theme)


# ==============================================================================
# SECTION 1: DATA ENGINE
# ==============================================================================

# ZeroHarm AI - Simulated Data Engine
# Generates realistic, plausible mock data for a steel plant / refinery:
# sensors (gas, temp, pressure), digital work permits, worker positions,
# and plant zone geometry. In production, this module is replaced by
# SCADA/IoT ingestion + permit-system API + RTLS feeds.

import numpy as np
import pandas as pd
import random
import time
from datetime import datetime, timedelta

random.seed()
np.random.seed()

# ---------------------------------------------------------------------------
# PLANT ZONE DEFINITIONS (mock steel plant layout, lat/lon around a fixed site)
# ---------------------------------------------------------------------------
PLANT_CENTER = (22.8046, 86.2029)  # Jamshedpur-ish reference point (steel belt)

ZONES = [
    {
        "id": "Z1", "name": "Coke Oven Battery", "type": "confined_space_adjacent",
        "polygon": [
            (22.8060, 86.2010), (22.8066, 86.2010),
            (22.8066, 86.2020), (22.8060, 86.2020)
        ],
        "gas_baseline": 18, "gas_type": "CO (Carbon Monoxide)", "gas_threshold": 50,
        "temp_baseline": 55, "temp_threshold": 85,
        "hazard_class": "High - Toxic Gas / Hot Work Zone"
    },
    {
        "id": "Z2", "name": "Blast Furnace Cast House", "type": "high_temp",
        "polygon": [
            (22.8040, 86.2010), (22.8046, 86.2010),
            (22.8046, 86.2020), (22.8040, 86.2020)
        ],
        "gas_baseline": 8, "gas_type": "H2S (Hydrogen Sulfide)", "gas_threshold": 10,
        "temp_baseline": 65, "temp_threshold": 90,
        "hazard_class": "High - Molten Metal / Burns"
    },
    {
        "id": "Z3", "name": "LPG Storage & Piping", "type": "confined_space",
        "polygon": [
            (22.8040, 86.2025), (22.8046, 86.2025),
            (22.8046, 86.2035), (22.8040, 86.2035)
        ],
        "gas_baseline": 5, "gas_type": "LPG (Combustible)", "gas_threshold": 20,
        "temp_baseline": 32, "temp_threshold": 60,
        "hazard_class": "Critical - Explosive Atmosphere"
    },
    {
        "id": "Z4", "name": "Confined Space - Tank Cleaning Bay", "type": "confined_space",
        "polygon": [
            (22.8060, 86.2025), (22.8066, 86.2025),
            (22.8066, 86.2035), (22.8060, 86.2035)
        ],
        "gas_baseline": 6, "gas_type": "O2 Deficiency / H2S", "gas_threshold": 15,
        "temp_baseline": 30, "temp_threshold": 55,
        "hazard_class": "Critical - Confined Space Entry"
    },
    {
        "id": "Z5", "name": "Rolling Mill Floor", "type": "mechanical",
        "polygon": [
            (22.8050, 86.2005), (22.8058, 86.2005),
            (22.8058, 86.2012), (22.8050, 86.2012)
        ],
        "gas_baseline": 2, "gas_type": "N/A", "gas_threshold": 999,
        "temp_baseline": 38, "temp_threshold": 65,
        "hazard_class": "Medium - Moving Machinery"
    },
    {
        "id": "Z6", "name": "Administrative / Control Room", "type": "office",
        "polygon": [
            (22.8050, 86.2036), (22.8058, 86.2036),
            (22.8058, 86.2042), (22.8050, 86.2042)
        ],
        "gas_baseline": 0, "gas_type": "N/A", "gas_threshold": 999,
        "temp_baseline": 26, "temp_threshold": 40,
        "hazard_class": "Low - General Area"
    },
]

WORKER_NAMES = [
    "R. Kumar", "S. Verma", "A. Singh", "P. Das", "M. Yadav", "T. Roy",
    "V. Nair", "K. Reddy", "J. Mahato", "N. Sharma", "D. Ghosh", "L. Behera"
]

ROLES = ["Fitter", "Welder", "Rigger", "Confined Space Entrant", "Shift Supervisor",
         "Gas Tester", "Fire Watch", "Operator"]

PERMIT_TYPES = ["Hot Work Permit", "Confined Space Entry Permit", "Electrical LOTO Permit",
                "Working at Height Permit", "Excavation Permit", "General Work Permit"]

REGULATIONS = {
    "Hot Work Permit": "OISD-STD-105 Sec 6.2",
    "Confined Space Entry Permit": "OISD-STD-105 Sec 7 / DGMS Confined Space Circular",
    "Electrical LOTO Permit": "Factory Act 1948 Sec 36 / OISD-STD-105 Sec 8",
    "Working at Height Permit": "Factory Act 1948 Sec 35",
    "Excavation Permit": "OISD-STD-105 Sec 9",
    "General Work Permit": "OISD-STD-105 Sec 5",
}


def _jitter_point(polygon):
    """Return a random point roughly inside a zone's bounding box."""
    lats = [p[0] for p in polygon]
    lons = [p[1] for p in polygon]
    return (random.uniform(min(lats), max(lats)), random.uniform(min(lons), max(lons)))


def generate_sensor_reading(zone, tick, spike_zone_id=None):
    """
    Generate one time-series sensor reading for a zone.
    Occasionally injects a plausible spike (simulating a real leak / heat event)
    for demo purposes, controlled by spike_zone_id.
    """
    gas_noise = np.random.normal(0, zone["gas_baseline"] * 0.15 + 1)
    temp_noise = np.random.normal(0, 2)
    pressure = round(np.random.normal(1.0, 0.03), 3)  # bar, relative

    gas_val = max(0, zone["gas_baseline"] + gas_noise)
    temp_val = max(15, zone["temp_baseline"] + temp_noise)

    if spike_zone_id == zone["id"]:
        # simulate a rising leak/heat trend
        ramp = min(1.0, (tick % 20) / 12)
        gas_val += ramp * (zone["gas_threshold"] * 1.6)
        temp_val += ramp * (zone["temp_threshold"] * 0.35)

    return {
        "zone_id": zone["id"],
        "zone_name": zone["name"],
        "timestamp": datetime.now(),
        "gas_type": zone["gas_type"],
        "gas_ppm": round(gas_val, 1),
        "gas_threshold": zone["gas_threshold"],
        "temp_c": round(temp_val, 1),
        "temp_threshold": zone["temp_threshold"],
        "pressure_bar": pressure,
        "hazard_class": zone["hazard_class"],
    }


def generate_permits(tick, force_conflict_zone=None):
    """Generate a plausible set of active digital work permits."""
    permits = []
    n_permits = random.randint(3, 6)
    used_zones = set()
    for i in range(n_permits):
        zone = random.choice(ZONES)
        ptype = random.choice(PERMIT_TYPES)
        issued = datetime.now() - timedelta(minutes=random.randint(5, 240))
        valid_hours = random.choice([4, 8, 12])
        permits.append({
            "permit_id": f"WP-{2026000 + tick * 10 + i}",
            "zone_id": zone["id"],
            "zone_name": zone["name"],
            "type": ptype,
            "regulation_ref": REGULATIONS[ptype],
            "issued_by": "Shift Supervisor - " + random.choice(WORKER_NAMES),
            "holder": random.choice(WORKER_NAMES),
            "issued_at": issued,
            "valid_until": issued + timedelta(hours=valid_hours),
            "status": "Active",
        })

    if force_conflict_zone:
        # ensure a Hot Work Permit is active in the spike zone -> compound risk demo
        zone = next(z for z in ZONES if z["id"] == force_conflict_zone)
        permits.append({
            "permit_id": f"WP-{2026999}",
            "zone_id": zone["id"],
            "zone_name": zone["name"],
            "type": "Hot Work Permit",
            "regulation_ref": REGULATIONS["Hot Work Permit"],
            "issued_by": "Shift Supervisor - " + random.choice(WORKER_NAMES),
            "holder": random.choice(WORKER_NAMES),
            "issued_at": datetime.now() - timedelta(minutes=20),
            "valid_until": datetime.now() + timedelta(hours=4),
            "status": "Active",
        })
    return permits


def generate_workers(tick):
    """Generate plausible worker positions across zones."""
    workers = []
    n_workers = random.randint(8, 14)
    for i in range(n_workers):
        zone = random.choice(ZONES)
        lat, lon = _jitter_point(zone["polygon"])
        ppe_ok = random.random() > 0.12  # ~88% compliance baseline
        workers.append({
            "worker_id": f"EMP-{1000+i}",
            "name": random.choice(WORKER_NAMES),
            "role": random.choice(ROLES),
            "zone_id": zone["id"],
            "zone_name": zone["name"],
            "lat": lat,
            "lon": lon,
            "ppe_compliant": ppe_ok,
            "ppe_flag": None if ppe_ok else random.choice(
                ["No Helmet Detected", "No Gas Monitor Worn", "Missing Fire-Retardant Suit"]
            ),
        })
    return workers


class SimulationState:
    """Holds rolling state for the demo simulation across reruns."""
    def __init__(self):
        self.tick = 0
        self.spike_zone_id = "Z1"  # default demo scenario: coke oven gas + hot work
        self.history = {z["id"]: [] for z in ZONES}

    def step(self):
        self.tick += 1
        sensors = [generate_sensor_reading(z, self.tick, self.spike_zone_id) for z in ZONES]
        for s in sensors:
            self.history[s["zone_id"]].append(s)
            if len(self.history[s["zone_id"]]) > 60:
                self.history[s["zone_id"]].pop(0)
        permits = generate_permits(self.tick, force_conflict_zone=self.spike_zone_id)
        workers = generate_workers(self.tick)
        return sensors, permits, workers

    def history_df(self, zone_id):
        return pd.DataFrame(self.history.get(zone_id, []))

# ==============================================================================
# SECTION 2: RISK ENGINE
# ==============================================================================

# ZeroHarm AI - Multi-Agent Risk Detection Engine
# ------------------------------------------------
# Simulates cooperating agents:
#   1. SensorAgent        - single-source threshold breach detection
#   2. PermitAgent        - validates permits against live conditions
#   3. CompoundRiskAgent  - fuses sensor + permit + zone-type signals to find
#                            correlated risks that no single sensor would flag
#   4. ScoringAgent       - produces a 0-100 explainable risk score
#
# Design principle: prefer LOW FALSE NEGATIVES for life-critical detection.
# Compound risks always take precedence over single-sensor thresholds.
# Every score returns a "Why?" explanation with contributing factors + a
# confidence estimate. Actions are human-in-the-loop (recommend, not auto-execute)
# except where explicitly simulated in the Emergency Orchestrator.

from datetime import datetime


def sensor_agent(sensor):
    """Flags single-source threshold breaches."""
    flags = []
    gas_ratio = sensor["gas_ppm"] / sensor["gas_threshold"] if sensor["gas_threshold"] < 999 else 0
    temp_ratio = sensor["temp_c"] / sensor["temp_threshold"]

    if gas_ratio >= 1.0:
        flags.append({
            "factor": f"{sensor['gas_type']} at {sensor['gas_ppm']} ppm exceeds threshold "
                      f"({sensor['gas_threshold']} ppm)",
            "weight": min(40, 20 + gas_ratio * 10),
        })
    elif gas_ratio >= 0.7:
        flags.append({
            "factor": f"{sensor['gas_type']} trending high: {sensor['gas_ppm']} ppm "
                      f"({int(gas_ratio*100)}% of threshold)",
            "weight": 12,
        })

    if temp_ratio >= 1.0:
        flags.append({
            "factor": f"Temperature {sensor['temp_c']}°C exceeds safe threshold "
                      f"({sensor['temp_threshold']}°C)",
            "weight": min(30, 15 + temp_ratio * 8),
        })
    elif temp_ratio >= 0.85:
        flags.append({
            "factor": f"Temperature trending high: {sensor['temp_c']}°C",
            "weight": 8,
        })

    return flags, gas_ratio, temp_ratio


def permit_agent(zone_id, permits):
    """Validates permits active in a zone; returns permit-related risk flags."""
    zone_permits = [p for p in permits if p["zone_id"] == zone_id]
    flags = []
    hot_work_active = any(p["type"] == "Hot Work Permit" for p in zone_permits)
    confined_space_active = any(p["type"] == "Confined Space Entry Permit" for p in zone_permits)

    for p in zone_permits:
        remaining = (p["valid_until"] - datetime.now()).total_seconds() / 60
        if remaining < 0:
            flags.append({
                "factor": f"Permit {p['permit_id']} ({p['type']}) has EXPIRED but appears active",
                "weight": 25,
            })

    return flags, hot_work_active, confined_space_active, zone_permits


def compound_risk_agent(sensor, gas_ratio, temp_ratio, hot_work_active, confined_space_active, zone_meta):
    """
    Core differentiator: detects CORRELATED risks across sensor + permit +
    zone-type signals that individually would sit below alarm threshold,
    but combined represent a critical, escalating hazard.
    Compound risks are weighted higher and take precedence in the final score.
    """
    compound_flags = []

    # Rule 1: Elevated/rising gas + active Hot Work permit (ignition source)
    if gas_ratio >= 0.55 and hot_work_active:
        compound_flags.append({
            "factor": f"COMPOUND: Rising {sensor['gas_type']} ({int(gas_ratio*100)}% of threshold) "
                      f"coincides with an ACTIVE Hot Work Permit in the same zone — ignition risk",
            "weight": 45,
            "regulation": "OISD-STD-105 Sec 6.2 (Hot Work in presence of flammable/toxic atmosphere prohibited)",
        })

    # Rule 2: Confined space entry permit + gas/O2 anomaly
    if confined_space_active and gas_ratio >= 0.5:
        compound_flags.append({
            "factor": f"COMPOUND: Confined Space Entry Permit active while "
                      f"{sensor['gas_type']} levels are elevated — entrant asphyxiation/toxicity risk",
            "weight": 45,
            "regulation": "OISD-STD-105 Sec 7 (Confined Space Entry Protocol)",
        })

    # Rule 3: High temp + high gas simultaneously (runaway condition)
    if gas_ratio >= 0.7 and temp_ratio >= 0.7:
        compound_flags.append({
            "factor": "COMPOUND: Simultaneous gas AND temperature escalation — possible runaway "
                      "thermal/chemical event forming",
            "weight": 35,
            "regulation": "OISD-STD-105 Gas Handling Guidelines",
        })

    # Rule 4: Explosive-atmosphere zone type + any hot work regardless of live gas value (procedural)
    if zone_meta["type"] == "confined_space" and hot_work_active and "Explosive" in zone_meta["hazard_class"]:
        compound_flags.append({
            "factor": "COMPOUND: Hot Work Permit issued inside a designated explosive-atmosphere "
                      "storage zone — procedural violation regardless of instantaneous reading",
            "weight": 50,
            "regulation": "OISD-STD-105 Sec 6.2 / OISD-STD-116 (LPG Storage)",
        })

    return compound_flags


def scoring_agent(single_flags, compound_flags, permit_flags):
    """
    Fuses all agent outputs into a single explainable 0-100 risk score.
    Compound risks are weighted to dominate the score (precedence rule).
    """
    all_flags = single_flags + permit_flags + compound_flags
    base = sum(f["weight"] for f in all_flags)
    # compound risks get an additional escalation multiplier
    compound_bonus = 1.25 if compound_flags else 1.0
    score = min(100, round(base * compound_bonus))

    if score >= 75:
        level, color = "CRITICAL", "rose"
    elif score >= 45:
        level, color = "WARNING", "amber"
    elif score >= 20:
        level, color = "ELEVATED", "amber"
    else:
        level, color = "SAFE", "emerald"

    confidence = min(0.97, 0.55 + 0.08 * len(all_flags) + (0.15 if compound_flags else 0))

    return {
        "score": score,
        "level": level,
        "color": color,
        "confidence": round(confidence, 2),
        "contributing_factors": all_flags,
        "is_compound": len(compound_flags) > 0,
    }


def assess_zone(sensor, permits, zone_meta):
    """Top-level orchestration: runs all agents for a single zone."""
    single_flags, gas_ratio, temp_ratio = sensor_agent(sensor)
    perm_flags, hot_work_active, confined_active, zone_permits = permit_agent(zone_meta["id"], permits)
    compound_flags = compound_risk_agent(
        sensor, gas_ratio, temp_ratio, hot_work_active, confined_active, zone_meta
    )
    result = scoring_agent(single_flags, compound_flags, perm_flags)
    result.update({
        "zone_id": zone_meta["id"],
        "zone_name": zone_meta["name"],
        "gas_ppm": sensor["gas_ppm"],
        "gas_type": sensor["gas_type"],
        "temp_c": sensor["temp_c"],
        "active_permits": zone_permits,
        "hot_work_active": hot_work_active,
        "confined_space_active": confined_active,
        "timestamp": sensor["timestamp"],
    })
    return result


def validate_permit(permit, zone_assessment):
    """
    Digital Permit Intelligence: flags a permit as Compliant / Flagged based on
    real-time zone conditions (not just paperwork correctness).
    """
    reasons = []
    if zone_assessment["is_compound"]:
        for cf in zone_assessment["contributing_factors"]:
            if cf.get("regulation") and permit["type"] in cf["factor"]:
                reasons.append(cf["factor"])
        if not reasons and permit["type"] in ("Hot Work Permit", "Confined Space Entry Permit"):
            reasons.append("Zone shows compound risk conditions incompatible with this permit type")

    remaining_min = (permit["valid_until"] - datetime.now()).total_seconds() / 60
    if remaining_min < 0:
        reasons.append("Permit validity window has expired")

    status = "Flagged" if reasons else "Compliant"
    return status, reasons

# ==============================================================================
# SECTION 3: KNOWLEDGE BASE (RAG-lite)
# ==============================================================================

# ZeroHarm AI - Incident Pattern & Regulatory Knowledge Base (RAG-lite)
# ----------------------------------------------------------------------
# MVP retrieval layer: keyword/TF-lite scoring over a curated set of
# regulatory excerpts + synthetic historical incident summaries.
# In production this would be swapped for a real vector store (e.g. FAISS/
# pgvector) embedding OISD/DGMS/Factory-Act PDFs + the incident log DB,
# with an LLM generation step grounded strictly on retrieved passages.

import re

DOCUMENTS = [
    {
        "id": "OISD-105-6.2",
        "source": "OISD-STD-105, Section 6.2 - Hot Work Permit",
        "text": ("Hot work shall not be permitted in areas where flammable gas concentration "
                 "exceeds 10% of the Lower Explosive Limit (LEL). Gas testing must be conducted "
                 "immediately prior to permit issuance and repeated hourly during the work. "
                 "Hot work permits must be suspended immediately if gas levels rise above threshold "
                 "at any point during the activity."),
    },
    {
        "id": "OISD-105-7",
        "source": "OISD-STD-105, Section 7 - Confined Space Entry",
        "text": ("Entry into confined spaces requires atmospheric testing for oxygen (19.5%-23.5%), "
                 "flammable gases, and toxic gases (H2S, CO) before and continuously during entry. "
                 "A dedicated fire watch / attendant must remain at the entry point at all times. "
                 "Entry permits are invalid if any single parameter drifts outside safe limits."),
    },
    {
        "id": "OISD-116-LPG",
        "source": "OISD-STD-116 - LPG Storage Facilities",
        "text": ("No hot work or spark-generating activity is permitted within the designated LPG "
                 "storage and piping exclusion zone under any circumstance, irrespective of "
                 "instantaneous gas readings, due to the high consequence severity of vapor cloud "
                 "explosions in such environments."),
    },
    {
        "id": "FACTORY-ACT-36",
        "source": "Factory Act 1948, Section 36 - Hazardous Processes",
        "text": ("Precautions must be taken against dangerous fumes, gases, and lack of oxygen in "
                 "confined spaces. Employers must ensure a permit-to-work system, gas testing, and "
                 "emergency rescue arrangements are in place before entry is authorized."),
    },
    {
        "id": "DGMS-CS-CIRC",
        "source": "DGMS Confined Space Entry Circular",
        "text": ("All confined space entries in mining and allied industries require a competent "
                 "person's certificate of gas-free status, continuous monitoring during occupancy, "
                 "and immediate evacuation protocol if gas concentration trends upward by more than "
                 "20% within any 15-minute window."),
    },
    {
        "id": "INC-2023-014",
        "source": "Historical Incident INC-2023-014 (Coke Oven Battery, similar plant)",
        "text": ("CO gas concentration rose gradually over 40 minutes while a hot work permit remained "
                 "active nearby. No compound alarm existed at the time; single-sensor threshold alarm "
                 "triggered only after gas exceeded 50 ppm, by which point ignition had already occurred. "
                 "Root cause: absence of correlated permit+sensor monitoring. Recommendation: implement "
                 "compound risk detection fusing permit status with live gas trend."),
    },
    {
        "id": "INC-2022-031",
        "source": "Historical Incident INC-2022-031 (Confined Space Tank Cleaning)",
        "text": ("Entrant experienced oxygen deficiency after ventilation fan failure went undetected "
                 "for 18 minutes. Fire watch was present but had no real-time sensor readout. "
                 "Recommendation: mandatory live dashboard feed to fire watch / attendant, not just "
                 "periodic manual gas testing."),
    },
    {
        "id": "INC-2021-007",
        "source": "Historical Incident INC-2021-007 (LPG Storage Area)",
        "text": ("A welding job was mistakenly authorized within the LPG exclusion zone due to a "
                 "manual paperwork error; gas levels were within normal range at approval time. "
                 "Recommendation: zone-type based procedural blocking of hot work permits in "
                 "explosive-atmosphere zones regardless of live readings."),
    },
]


def _tokenize(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def retrieve(query, top_k=3):
    """Simple overlap-scoring retrieval (MVP stand-in for embedding similarity)."""
    q_tokens = _tokenize(query)
    scored = []
    for doc in DOCUMENTS:
        d_tokens = _tokenize(doc["text"] + " " + doc["source"])
        overlap = len(q_tokens & d_tokens)
        if overlap > 0:
            scored.append((overlap, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]


def answer_query(query):
    """
    Produces a grounded answer + citations from retrieved docs.
    This is a template-based generator for the MVP; swap for an LLM call
    grounded strictly on the retrieved passages in production.
    """
    hits = retrieve(query, top_k=3)
    if not hits:
        return {
            "answer": "No directly matching regulatory or incident records found for this query. "
                      "Try terms like 'gas', 'hot work', 'confined space', 'LPG', or a permit type.",
            "sources": [],
        }

    lines = []
    for doc in hits:
        lines.append(f"**{doc['source']}**: {doc['text']}")

    summary = (
        f"Based on {len(hits)} matching record(s), here is the relevant guidance/history:\n\n"
        + "\n\n".join(lines)
    )
    return {"answer": summary, "sources": [d["source"] for d in hits]}

# ==============================================================================
# SECTION 4: STREAMLIT APP
# ==============================================================================

# ZeroHarm AI — Industrial Safety Intelligence Platform
# MVP prototype (Streamlit). Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import folium
from streamlit_folium import st_folium
from datetime import datetime
import time


# ---------------------------------------------------------------------------
# PAGE CONFIG + DARK INDUSTRIAL THEME
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ZeroHarm AI — Industrial Safety Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg-primary: #0f172a;
    --bg-card: #1a2438;
    --bg-card-hover: #212d47;
    --sidebar-bg: #1e293b;
    --sidebar-active: #2a3a58;
    --border: #2f3d56;
    --accent: #3b82f6;
    --emerald: #22c55e;
    --amber: #f59e0b;
    --rose: #f43f5e;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
}

html, body, .stApp { font-family: 'Inter', -apple-system, sans-serif; }
.stApp { background-color: var(--bg-primary); color: var(--text-primary); }
h1, h2, h3, h4 { color: var(--text-primary) !important; font-weight: 700; letter-spacing: -0.01em; }
p, span, label, div { color: var(--text-primary); }

/* Top toolbar / header bar cleanup */
header[data-testid="stHeader"] { background-color: var(--bg-primary); }
header[data-testid="stHeader"] * { color: var(--text-primary) !important; }
header[data-testid="stHeader"] svg { fill: var(--text-primary) !important; }
header[data-testid="stHeader"] button:hover { background: var(--bg-card-hover) !important; }
div[data-testid="stToolbarActions"] button { color: var(--text-primary) !important; }

/* All dropdown popovers: kebab "..." menu, selectbox (e.g. role picker), multiselect, etc. */
div[data-baseweb="popover"] { filter: none !important; }
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] ul[role="listbox"],
div[data-baseweb="popover"] ul[data-testid="stMainMenuList"],
div[data-baseweb="popover"] ul[data-testid="stSelectboxVirtualDropdown"],
div[data-baseweb="popover"] div[data-baseweb="menu"],
div[data-baseweb="menu"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.45) !important;
}
div[data-baseweb="popover"] li,
div[data-baseweb="popover"] li *,
div[data-baseweb="popover"] [role="option"],
div[data-baseweb="popover"] [role="option"] *,
div[data-baseweb="menu"] li,
div[data-baseweb="menu"] li * {
    color: var(--text-primary) !important;
    background-color: transparent !important;
}
div[data-baseweb="popover"] li:hover,
div[data-baseweb="popover"] [role="option"]:hover,
div[data-baseweb="popover"] [aria-selected="true"],
div[data-baseweb="menu"] li:hover {
    background-color: var(--bg-card-hover) !important;
}
div[data-baseweb="popover"] hr { border-color: var(--border) !important; }

/* Sidebar collapse / re-open control — make it a clearly visible pill button */
button[data-testid="stSidebarCollapsedControl"],
div[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 6px 10px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.35);
    opacity: 1 !important;
    visibility: visible !important;
    z-index: 999999 !important;
}
button[data-testid="stSidebarCollapsedControl"]:hover,
[data-testid="collapsedControl"]:hover {
    background: var(--bg-card-hover) !important;
    border-color: var(--accent) !important;
}
button[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="collapsedControl"] svg {
    fill: var(--text-primary) !important;
}
button[data-testid="baseButton-headerNoPadding"] svg { fill: var(--text-primary) !important; }

/* ---- Sidebar: brighter, higher-contrast, more structured ---- */
section[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small {
    color: var(--text-secondary) !important;
}
section[data-testid="stSidebar"] hr { border-color: var(--border); margin: 14px 0; }

.sidebar-brand {
    display: flex; align-items: center; gap: 10px; padding: 4px 0 2px 0;
}
.sidebar-brand .icon {
    width: 34px; height: 34px; border-radius: 9px;
    background: linear-gradient(135deg, var(--accent), #60a5fa);
    display: flex; align-items: center; justify-content: center; font-size: 1.1rem;
}
.sidebar-brand .name { font-size: 1.15rem; font-weight: 800; line-height: 1.1; }
.sidebar-brand .sub { font-size: 0.72rem; color: var(--text-secondary); }
.sidebar-section-label {
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--text-secondary); margin: 4px 0 6px 0;
}

/* Sidebar radio nav styled as clean vertical menu items */
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 2px; }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: transparent; border-radius: 8px; padding: 8px 10px !important;
    margin-bottom: 2px; transition: background 0.15s ease; border: 1px solid transparent;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: var(--bg-card-hover);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {
    background: var(--sidebar-active); border: 1px solid var(--border);
}

/* ---- Buttons EVERYWHERE (sidebar AND main content) ---- */
.stButton button, button[kind="secondary"], button[kind="primary"],
div[data-testid="stBaseButton-secondary"], div[data-testid="stBaseButton-primary"] {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    color: var(--text-primary) !important; border-radius: 8px !important; font-weight: 600;
}
.stButton button:hover, button[kind="secondary"]:hover, button[kind="primary"]:hover {
    background: var(--bg-card-hover) !important; border-color: var(--accent) !important;
}
.stButton button p, .stButton button span, .stButton button div { color: var(--text-primary) !important; }

/* ---- Selectboxes / multiselects EVERYWHERE (sidebar AND main content) ---- */
[data-baseweb="select"] > div, [data-baseweb="select"] div {
    background: var(--bg-card) !important; border-color: var(--border) !important;
    color: var(--text-primary) !important; border-radius: 8px !important;
}
[data-baseweb="select"] svg { fill: var(--text-primary) !important; }

/* ---- Dataframe toolbar (eye / download / search / fullscreen icons) ---- */
div[data-testid="stElementToolbar"], [data-testid*="Toolbar"] {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
div[data-testid="stElementToolbar"] button, [data-testid*="Toolbar"] button,
div[data-testid="stElementToolbar"] svg, [data-testid*="Toolbar"] svg {
    color: var(--text-primary) !important; fill: var(--text-primary) !important;
}
div[data-testid="stElementToolbar"] button:hover, [data-testid*="Toolbar"] button:hover {
    background: var(--bg-card-hover) !important;
}

/* ---- Dataframe column header menu (Sort / Autosize / Pin / Hide) ---- */
div[data-testid="stDataFrameColumnMenu"], [data-testid*="ColumnMenu"],
ul[data-testid="stDataFrameColumnMenu"] {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: 10px !important; box-shadow: 0 8px 24px rgba(0,0,0,0.45) !important;
}
div[data-testid="stDataFrameColumnMenu"] *, [data-testid*="ColumnMenu"] *,
ul[data-testid="stDataFrameColumnMenu"] * {
    color: var(--text-primary) !important;
}
div[data-testid="stDataFrameColumnMenu"] div:hover, [data-testid*="ColumnMenu"] div:hover,
div[data-testid="stDataFrameColumnMenu"] li:hover {
    background: var(--bg-card-hover) !important;
}

/* ---- Checkbox (e.g. Auto-refresh toggle) ---- */
[data-baseweb="checkbox"] span { color: var(--text-primary) !important; }
[data-baseweb="checkbox"] > div:first-child { border-color: var(--border) !important; }

/* ---- Cards, badges, metrics ---- */
.metric-card {
    background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px;
    padding: 18px 20px; margin-bottom: 10px; transition: border-color 0.15s ease;
}
.metric-card:hover { border-color: var(--accent); background: var(--bg-card-hover); }
.metric-label { color: var(--text-secondary); font-size: 0.75rem; text-transform: uppercase;
    letter-spacing: 0.06em; margin-bottom: 6px; font-weight: 600; }
.metric-value { font-size: 2rem; font-weight: 800; }

.badge { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.7rem;
    font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em; }
.badge-safe { background: rgba(34,197,94,0.15); color: var(--emerald); border: 1px solid var(--emerald); }
.badge-warning { background: rgba(245,158,11,0.15); color: var(--amber); border: 1px solid var(--amber); }
.badge-critical { background: rgba(244,63,94,0.15); color: var(--rose); border: 1px solid var(--rose);
    animation: pulse 1.5s infinite; }
@keyframes pulse { 0%{opacity:1;} 50%{opacity:0.55;} 100%{opacity:1;} }

.risk-card { border-left: 4px solid var(--border); background: var(--bg-card); border-radius: 8px;
    padding: 14px 16px; margin-bottom: 10px; border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border); border-right: 1px solid var(--border); }
.risk-card.critical { border-left-color: var(--rose); }
.risk-card.warning { border-left-color: var(--amber); }
.risk-card.safe { border-left-color: var(--emerald); }

/* Page header block */
.page-header { padding-bottom: 6px; margin-bottom: 8px; border-bottom: 1px solid var(--border); }

hr { border-color: var(--border); }
.footer-note { color: var(--text-secondary); font-size: 0.75rem; text-align: center; margin-top: 30px; }

/* Streamlit containers used as cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--border) !important; border-radius: 10px !important;
}

/* Tighten default top padding */
.block-container { padding-top: 2rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def badge(level):
    cls = {"CRITICAL": "badge-critical", "WARNING": "badge-warning", "ELEVATED": "badge-warning",
           "SAFE": "badge-safe"}.get(level, "badge-safe")
    return f'<span class="badge {cls}">{level}</span>'


def color_for(level):
    return {"CRITICAL": "#f43f5e", "WARNING": "#f59e0b", "ELEVATED": "#f59e0b", "SAFE": "#10b981"}.get(level, "#10b981")


# ---------------------------------------------------------------------------
# SESSION STATE / MOCK AUTH
# ---------------------------------------------------------------------------
if "sim" not in st.session_state:
    st.session_state.sim = SimulationState()
if "role" not in st.session_state:
    st.session_state.role = "Safety Officer"
if "last_step" not in st.session_state:
    st.session_state.last_step = 0

sim = st.session_state.sim

# Advance simulation once per rerun (mimics 5-30s real-time updates)
sensors, permits, workers = sim.step()
assessments = {z["id"]: assess_zone(
    next(s for s in sensors if s["zone_id"] == z["id"]), permits, z
) for z in ZONES}

overall_score = round(np.mean([a["score"] for a in assessments.values()]))
overall_level = "CRITICAL" if overall_score >= 75 else "WARNING" if overall_score >= 45 else \
                 "ELEVATED" if overall_score >= 20 else "SAFE"

# ---------------------------------------------------------------------------
# SIDEBAR NAV + MOCK AUTH
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">'
        '<div class="icon">🛡️</div>'
        '<div><div class="name">ZeroHarm AI</div>'
        '<div class="sub">Industrial Safety Intelligence</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown('<div class="sidebar-section-label">Signed in as</div>', unsafe_allow_html=True)
    st.session_state.role = st.selectbox(
        "Signed in as", ["Safety Officer", "Operator", "Auditor"],
        index=["Safety Officer", "Operator", "Auditor"].index(st.session_state.role),
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown('<div class="sidebar-section-label">Navigate</div>', unsafe_allow_html=True)
    page = st.radio("Navigate", [
        "📊 Live Plant Overview",
        "🗺️ Geospatial Safety Heatmap",
        "⚠️ Compound Risk Detector",
        "📋 Digital Permit Intelligence",
        "💬 Incident Pattern RAG Chat",
        "🎥 CCTV Analytics Feed",
        "🚨 Emergency Orchestrator",
        "✅ Compliance Audit",
    ], label_visibility="collapsed")
    st.markdown("---")
    auto_refresh = st.checkbox("Auto-refresh (10s)", value=False)
    if st.button("🔄 Manual Refresh / Advance Tick", use_container_width=True):
        st.rerun()
    st.caption(f"Sim tick #{sim.tick} · {datetime.now().strftime('%H:%M:%S')}")
    st.markdown('<p class="footer-note">Demo scenario: Zone Z1 (Coke Oven) gas ramp<br>'
                'active for compound-risk illustration.</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PAGE: LIVE PLANT OVERVIEW
# ---------------------------------------------------------------------------
if page == "📊 Live Plant Overview":
    st.title("Live Plant Overview")
    st.caption(f"Role: {st.session_state.role} · Last updated {datetime.now().strftime('%H:%M:%S')}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Facility Risk Score</div>
            <div class="metric-value" style="color:{color_for(overall_level)}">{overall_score}/100</div>
            {badge(overall_level)}</div>""", unsafe_allow_html=True)
    with c2:
        active_critical = sum(1 for a in assessments.values() if a["level"] == "CRITICAL")
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Critical Zones</div>
            <div class="metric-value" style="color:#f43f5e">{active_critical}</div>
            <span style="color:#9ca3af;font-size:0.8rem">of {len(ZONES)} zones</span></div>""",
            unsafe_allow_html=True)
    with c3:
        compound_count = sum(1 for a in assessments.values() if a["is_compound"])
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Compound Risks Active</div>
            <div class="metric-value" style="color:#f59e0b">{compound_count}</div>
            <span style="color:#9ca3af;font-size:0.8rem">correlated hazards</span></div>""",
            unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Workers on Site</div>
            <div class="metric-value">{len(workers)}</div>
            <span style="color:#9ca3af;font-size:0.8rem">{sum(1 for w in workers if not w['ppe_compliant'])} PPE flags</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("### Zone Status Grid")
    cols = st.columns(3)
    for i, z in enumerate(ZONES):
        a = assessments[z["id"]]
        with cols[i % 3]:
            css_class = "critical" if a["level"] == "CRITICAL" else "warning" if a["level"] in \
                ("WARNING", "ELEVATED") else "safe"
            st.markdown(f"""
            <div class="risk-card {css_class}">
                <b>{a['zone_name']}</b> ({a['zone_id']}) {badge(a['level'])}<br>
                <span style="color:#9ca3af;font-size:0.85rem">Score: {a['score']}/100 ·
                Confidence: {int(a['confidence']*100)}%</span><br>
                <span style="color:#9ca3af;font-size:0.85rem">{a['gas_type']}: {a['gas_ppm']} ppm ·
                Temp: {a['temp_c']}°C</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("### Active Alerts")
    alert_zones = [a for a in assessments.values() if a["level"] in ("CRITICAL", "WARNING")]
    if not alert_zones:
        st.success("No active alerts. All zones within safe operating parameters.")
    else:
        for a in sorted(alert_zones, key=lambda x: -x["score"]):
            icon = "🔴" if a["level"] == "CRITICAL" else "🟠"
            st.warning(f"{icon} **{a['zone_name']}** — Risk {a['score']}/100 "
                       f"{'(COMPOUND RISK)' if a['is_compound'] else ''} — "
                       f"{a['contributing_factors'][0]['factor'] if a['contributing_factors'] else ''}")
        if auto_refresh:
            st.toast(f"⚠️ {len(alert_zones)} zone(s) require attention", icon="⚠️")

# ---------------------------------------------------------------------------
# PAGE: GEOSPATIAL HEATMAP
# ---------------------------------------------------------------------------
elif page == "🗺️ Geospatial Safety Heatmap":
    st.title("Geospatial Safety Heatmap")
    st.caption("Interactive plant layout — click a zone or marker for details")

    m = folium.Map(location=PLANT_CENTER, zoom_start=17, tiles="CartoDB dark_matter")

    for z in ZONES:
        a = assessments[z["id"]]
        color = color_for(a["level"])
        centroid_lat = np.mean([p[0] for p in z["polygon"]])
        centroid_lon = np.mean([p[1] for p in z["polygon"]])
        popup_html = f"""
        <div style="font-family:sans-serif;min-width:220px">
        <b>{z['name']}</b> ({z['id']})<br>
        Risk Score: <b>{a['score']}/100</b> ({a['level']})<br>
        {a['gas_type']}: {a['gas_ppm']} ppm (limit {z['gas_threshold']})<br>
        Temp: {a['temp_c']}°C (limit {z['temp_threshold']})<br>
        Active Permits: {len(a['active_permits'])}<br>
        {'⚠️ <b>COMPOUND RISK DETECTED</b>' if a['is_compound'] else ''}
        </div>"""
        folium.Polygon(
            locations=z["polygon"], color=color, weight=2, fill=True,
            fill_color=color, fill_opacity=0.35,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{z['name']} — {a['level']}",
        ).add_to(m)
        folium.map.Marker(
            [centroid_lat, centroid_lon],
            icon=folium.DivIcon(html=f"""<div style="font-size:11px;font-weight:700;color:white;
                text-shadow:1px 1px 2px #000">{z['id']}</div>""")
        ).add_to(m)

    for w in workers:
        icon_color = "green" if w["ppe_compliant"] else "red"
        folium.CircleMarker(
            location=[w["lat"], w["lon"]], radius=5,
            color=icon_color, fill=True, fill_color=icon_color, fill_opacity=0.9,
            popup=f"{w['name']} ({w['role']})<br>Zone: {w['zone_name']}<br>"
                  f"PPE: {'✅ Compliant' if w['ppe_compliant'] else '❌ ' + w['ppe_flag']}",
            tooltip=w["name"],
        ).add_to(m)

    st_folium(m, width=None, height=560, returned_objects=[])

    st.markdown("""
    **Legend:** 🟩 Safe zone &nbsp;&nbsp; 🟧 Warning/Elevated &nbsp;&nbsp; 🟥 Critical &nbsp;&nbsp; |
    &nbsp;&nbsp; 🟢 Worker PPE compliant &nbsp;&nbsp; 🔴 Worker PPE violation
    """)

    st.markdown("### Zone Detail Table")
    df = pd.DataFrame([{
        "Zone": a["zone_name"], "ID": a["zone_id"], "Score": a["score"], "Level": a["level"],
        "Gas": f"{a['gas_ppm']} ppm", "Temp": f"{a['temp_c']}°C",
        "Active Permits": len(a["active_permits"]), "Compound Risk": "Yes" if a["is_compound"] else "No",
    } for a in assessments.values()])
    st.dataframe(df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# PAGE: COMPOUND RISK DETECTOR
# ---------------------------------------------------------------------------
elif page == "⚠️ Compound Risk Detector":
    st.title("Compound Risk Detector")
    st.caption("Correlated multi-source risks — where individual sensors alone would stay silent")

    compound_assessments = [a for a in assessments.values() if a["is_compound"]]
    if not compound_assessments:
        st.success("No compound risks detected across any zone at this time.")
    else:
        for a in sorted(compound_assessments, key=lambda x: -x["score"]):
            with st.container(border=True):
                st.markdown(f"#### 🔴 {a['zone_name']} ({a['zone_id']}) {badge(a['level'])}",
                             unsafe_allow_html=True)
                st.markdown(f"**Risk Score:** {a['score']}/100 · **Confidence:** {int(a['confidence']*100)}%")
                st.markdown("**Why? Contributing factors:**")
                for f in a["contributing_factors"]:
                    reg = f" · _{f['regulation']}_" if "regulation" in f else ""
                    st.markdown(f"- {f['factor']} (weight: {f['weight']}){reg}")
                st.caption("Recommended action: Human-in-the-loop review required. "
                           "Suggest immediate suspension of active hot work / entry permit and re-test atmosphere.")

    st.markdown("### Risk Evolution Timeline (Zone Z1 — Coke Oven demo scenario)")
    hist = sim.history_df("Z1")
    if not hist.empty and len(hist) > 1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["timestamp"], y=hist["gas_ppm"], name="CO (ppm)",
                                  line=dict(color="#f43f5e", width=2)))
        fig.add_hline(y=hist["gas_threshold"].iloc[-1], line_dash="dash", line_color="#f59e0b",
                       annotation_text="Threshold")
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            height=350, margin=dict(l=10, r=10, t=30, b=10),
            title="Coke Oven Battery — CO Gas Trend vs Threshold"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Collecting time-series data — refresh a few times to see the trend build.")

# ---------------------------------------------------------------------------
# PAGE: DIGITAL PERMIT INTELLIGENCE
# ---------------------------------------------------------------------------
elif page == "📋 Digital Permit Intelligence":
    st.title("Digital Permit Intelligence")
    st.caption("AI-validated work permits against live zone conditions")

    rows = []
    for p in permits:
        zone_assessment = assessments.get(p["zone_id"])
        status, reasons = validate_permit(p, zone_assessment) if zone_assessment else ("Compliant", [])
        rows.append({**p, "ai_status": status, "reasons": reasons})

    flagged = [r for r in rows if r["ai_status"] == "Flagged"]
    compliant = [r for r in rows if r["ai_status"] == "Compliant"]

    c1, c2 = st.columns(2)
    c1.metric("Compliant Permits", len(compliant))
    c2.metric("Flagged Permits", len(flagged), delta=None)

    if flagged:
        st.markdown("### 🚩 Flagged Permits")
        for r in flagged:
            with st.container(border=True):
                st.markdown(f"**{r['permit_id']}** — {r['type']} — {r['zone_name']} "
                             f'<span class="badge badge-critical">FLAGGED</span>', unsafe_allow_html=True)
                st.caption(f"Holder: {r['holder']} · Issued by: {r['issued_by']} · "
                           f"Regulation: {r['regulation_ref']}")
                for reason in r["reasons"]:
                    st.markdown(f"- ⚠️ {reason}")

    st.markdown("### All Active Permits")
    df = pd.DataFrame([{
        "Permit ID": r["permit_id"], "Type": r["type"], "Zone": r["zone_name"],
        "Holder": r["holder"], "Issued": r["issued_at"].strftime("%H:%M"),
        "Valid Until": r["valid_until"].strftime("%H:%M"),
        "AI Status": r["ai_status"], "Regulation": r["regulation_ref"],
    } for r in rows])
    st.dataframe(df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# PAGE: RAG CHAT
# ---------------------------------------------------------------------------
elif page == "💬 Incident Pattern RAG Chat":
    st.title("Incident Pattern RAG Chat")
    st.caption("Ask about regulatory standards or historical incident patterns "
               "(retrieval-grounded — MVP keyword retrieval over OISD/DGMS docs + incident log)")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    example_qs = [
        "Show patterns from past gas incidents per OISD-STD-105",
        "What are the rules for hot work permits near flammable gas?",
        "Confined space entry requirements",
    ]
    cols = st.columns(len(example_qs))
    for i, q in enumerate(example_qs):
        if cols[i].button(q, key=f"ex_{i}"):
            st.session_state.pending_query = q

    query = st.chat_input("Ask about regulations or incident patterns...")
    final_query = query or st.session_state.pop("pending_query", None)

    if final_query:
        st.session_state.chat_history.append(("user", final_query))
        result = answer_query(final_query)
        st.session_state.chat_history.append(("assistant", result["answer"]))
        st.rerun()

# ---------------------------------------------------------------------------
# PAGE: CCTV ANALYTICS FEED
# ---------------------------------------------------------------------------
elif page == "🎥 CCTV Analytics Feed":
    st.title("CCTV Analytics Feed")
    st.caption("Simulated multi-camera analytics — PPE, confined space entry, unauthorized personnel")

    cam_zones = ZONES[:4]
    cols = st.columns(2)
    for i, z in enumerate(cam_zones):
        zone_workers = [w for w in workers if w["zone_id"] == z["id"]]
        violations = [w for w in zone_workers if not w["ppe_compliant"]]
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"**CAM-{i+1:02d} · {z['name']}**")
                st.markdown(f"""
                <div style="background:#000;height:160px;border-radius:8px;display:flex;
                align-items:center;justify-content:center;color:#4b5563;font-size:0.85rem;
                border:1px solid #2d3748;">📹 Live feed placeholder — {len(zone_workers)} person(s) tracked</div>
                """, unsafe_allow_html=True)
                if violations:
                    for v in violations:
                        st.error(f"🚨 {v['name']}: {v['ppe_flag']}")
                else:
                    st.success("✅ No anomalies detected")

    st.info("Production integration: replace placeholders with RTSP feed + YOLOv8 inference "
            "for PPE detection, restricted-zone intrusion, and fall detection.")

# ---------------------------------------------------------------------------
# PAGE: EMERGENCY ORCHESTRATOR
# ---------------------------------------------------------------------------
elif page == "🚨 Emergency Orchestrator":
    st.title("Emergency Orchestrator")
    st.caption("Human-in-the-loop response simulation — no action fires without explicit confirmation")

    critical = [a for a in assessments.values() if a["level"] == "CRITICAL"]
    if critical:
        st.error(f"⚠️ {len(critical)} zone(s) in CRITICAL state require review.")
        for a in critical:
            st.markdown(f"**{a['zone_name']}** — Score {a['score']}/100")
    else:
        st.info("No zones currently in CRITICAL state. Orchestrator on standby.")

    st.markdown("### Response Protocol")
    zone_choice = st.selectbox("Select zone for response action",
                                [z["name"] for z in ZONES])

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📢 Trigger Zone Alert", use_container_width=True):
            st.toast(f"Alert broadcast sent to all personnel in {zone_choice}", icon="📢")
            st.success(f"✅ Simulated: SMS + app push + siren triggered for {zone_choice}")
    with c2:
        if st.button("🚪 Simulate Evacuation Protocol", use_container_width=True):
            st.warning(f"🚪 Simulated evacuation routing activated for {zone_choice}. "
                       f"Nearest muster point notified. (Human confirmation required to execute in production)")
    with c3:
        if st.button("📝 Generate Preliminary Report", use_container_width=True):
            a = next(a for a in assessments.values() if a["zone_name"] == zone_choice)
            st.markdown("#### Preliminary Incident Report (auto-drafted — requires sign-off)")
            st.code(f"""ZONE: {a['zone_name']} ({a['zone_id']})
TIMESTAMP: {a['timestamp']}
RISK SCORE: {a['score']}/100 ({a['level']})
CONFIDENCE: {int(a['confidence']*100)}%
CONTRIBUTING FACTORS:
{chr(10).join('- ' + f['factor'] for f in a['contributing_factors']) if a['contributing_factors'] else '- None'}
ACTIVE PERMITS: {len(a['active_permits'])}
STATUS: Draft — pending Safety Officer review and sign-off
""", language="text")

    st.markdown("---")
    st.caption("⚙️ All emergency actions in this MVP are simulated and logged for audit; "
               "no real-world systems are triggered. Production deployment requires dual-confirmation "
               "for auto-shutdown/evacuation triggers per human-in-the-loop design principle.")

# ---------------------------------------------------------------------------
# PAGE: COMPLIANCE AUDIT
# ---------------------------------------------------------------------------
elif page == "✅ Compliance Audit":
    st.title("Compliance Audit")
    st.caption("Real-time deviation flags vs. OISD / Factory Act / DGMS standards")

    deviations = []
    for a in assessments.values():
        for f in a["contributing_factors"]:
            if "regulation" in f:
                deviations.append({
                    "Zone": a["zone_name"], "Deviation": f["factor"],
                    "Regulation": f["regulation"], "Severity": a["level"],
                })

    if not deviations:
        st.success("✅ No active regulatory deviations detected across the facility.")
    else:
        df = pd.DataFrame(deviations)
        st.dataframe(df, use_container_width=True, hide_index=True)

        fig = px.bar(df["Severity"].value_counts().reset_index(),
                      x="Severity", y="count", color="Severity",
                      color_discrete_map={"CRITICAL": "#f43f5e", "WARNING": "#f59e0b",
                                          "ELEVATED": "#f59e0b", "SAFE": "#10b981"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                           height=300, margin=dict(l=10, r=10, t=30, b=10),
                           title="Deviations by Severity")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Audit Log")
    st.caption("Immutable log of all AI-generated risk assessments and human actions (mock)")
    log_rows = [{
        "Time": a["timestamp"].strftime("%H:%M:%S"), "Zone": a["zone_name"],
        "Score": a["score"], "Level": a["level"], "Reviewed By": st.session_state.role,
    } for a in assessments.values()]
    st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
if auto_refresh:
    time.sleep(10)
    st.rerun()

st.markdown('<p class="footer-note">ZeroHarm AI — MVP Prototype · Simulated data · '
            'Aligned with OISD-STD-105, Factory Act 1948, DGMS guidance</p>', unsafe_allow_html=True)
