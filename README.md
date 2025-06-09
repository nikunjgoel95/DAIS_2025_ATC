# DAIS_2025_ATC

# 🚀 **SkyLink Navigator** — Multimodal Agent for Air-Traffic Coordination  
*“A single conversational interface that lets pilots request, receive, and confirm all clearances in seconds.”*

---

## 1. How the Pilot Interacts

| Step | Pilot Utterance (examples) | SkyLink Navigator Response | Behind-the-scenes Sub-Agents |
|------|---------------------------|----------------------------|------------------------------|
| 1️⃣ | “SkyLink, request IFR clearance to KSEA.” | Reads back clearance and squawk code. | **Schedule Tracker**, **Comms Analyser** |
| 2️⃣ | “Confirm weather on final approach.” | Summarises current METAR and wind-shear risk. | **Weather Agent** |
| 3️⃣ | “Advise on traffic ahead.” | Describes traffic’s call-sign, position, altitude; issues sequencing advice. | **GeoTracker**, **Comms Analyser** |
| 4️⃣ | *≋ No verbal request; aircraft drifts off localiser* | “Alert ⚠️ — You’re deviating 7° right of course.” | **GeoTracker** (autonomous trigger) |
| 5️⃣ | “Mayday, engine out!” | Automatically relays emergency to tower; suggests nearest runway & vectors. | **Comms Analyser**, **Schedule Tracker**, **GeoTracker**, **Weather** |

---

## 2. Sub-Agent Suite

| Sub-Agent | Key Role | Inputs | Outputs | Typical Alerts |
|-----------|----------|--------|---------|----------------|
| **Comms Analyser** | NLP model that inspects live ATC ⇄ pilot audio, flags unusual or emergency phrases. | 16-kHz audio streams, speech-to-text feed | Transcript + intent classification | *Mayday detected*, *pilot missed read-back*, *tower issued conflicting clearances* |
| **GeoTracker** | Real-time trajectory watchdog using ADS-B. | ADS-B positions @ 1 Hz | State (`in-air`, `approach`, `taxi`, etc.), deviation score | *Runway incursion*, *unstable approach*, *ground conflict* |
| **Schedule Tracker** | Gate / runway slot planner. | Airline schedules, live AODB, gate database | ETA/ETD, gate availability, slot confirmations | *Gate mismatch*, *runway change*, *pushback hold* |
| **Weather Agent** | Micro-weather decision aid. | METAR, TAF, surface radar, wind-shear sensor grid | Structured weather brief + risk score | *Crosswind above limits*, *microburst alert* |

### GeoTracker — Defined States

| State | Description |
|-------|-------------|
| `IN_AIR` | Cruise or climb / descent en-route. |
| `APPROACH` | Within 10 NM & 3 000 ft of destination. |
| `FINAL` | On localiser / glide path. |
| `DEVIATED` | Off published SID/STAR or > X σ from localiser / centreline. |
| `GATE` | Parked with chocks in. |
| `PUSHBACK` | Clearance received, push in progress. |
| `TAXI_OUT` | Taxiing to runway. |
| `HOLD_SHORT` | Holding at runway entry. |
| `TAKEOFF_ROLL` | Accelerating on runway. |
| `TAXI_IN` | Vacated runway, taxi to gate. |

---

## 3. Dataset & Simulation Harness

* **Audio Mix** – 10 parallel voice channels  
  * 3 airborne: **(1)** nominal, **(2)** *“Mayday”*, **(3)** trajectory deviation.  
  * 7 ground: transcript-only channels (pushback, taxi, gate ops).

* **ADS-B Live Feed** – Real-time positions for the same 10 call-signs; injected anomalies for #3.

* **Synthetic Weather** – METAR/TAF generator with variable wind-shear & ceiling.

* **Flight Schedule** – Randomised arrivals/departures incl. gate assignments.

---

## 4. Tech Stack at a Glance

| Layer | Tooling |
|-------|---------|
| Data Governance | **Unity Catalog** (separate schemas for Audio, ADS-B, Weather, Schedule) |
| Orchestration | **LangGraph** (multi-agent flow & tool-calling) |
| Serving | **Databricks Model Serving** (low-latency endpoints) |
| Experiment Tracking & Eval | **MLflow + mlflow.evaluate()** (precision/recall for each agent) |
| Compute | **Serverless Jobs** (audio STT, ADS-B ingestion) |
| Memory / Retrieval | **Vector Store** (FAISS on Delta, chunked transcripts & SOPs) |
| Rapid Prototyping | **Genie** notebooks for ad-hoc agent testing |

---

## 5. Demo Storyboard (for Hackathon Pitch)

1. **Scenario Load** – Ingest ADS-B feed & audio mix (2 min).  
2. **Live Dashboard** – Display GeoTracker map + Weather & Gate panels.  
3. **Interactive Exchange** – Pilot issues clearance request → SkyLink reads back.  
4. **Injected Deviation** – Simulate drift; GeoTracker fires alert, agent warns pilot.  
5. **Emergency Call** – Play “Mayday”; Comms Analyser auto-routes EMS message.  
6. **Metrics Slide** – Show eval dashboard: F1 scores, latency < 400 ms (95-pct).  
7. **Finish** – “One cockpit voice, four sub-agents, zero runway incursions.”

---

> **Next steps:**  
> • Fine-tune Comms Analyser on larger ATC corpora.  
> • Expand Schedule Tracker to support multi-airport sequencing.  
> • Integrate surface-radar for ground-movement conflict detection.  

“Cleared for the future—contact **SkyLink Navigator** on 121.5 for any further assistance.” 🛫