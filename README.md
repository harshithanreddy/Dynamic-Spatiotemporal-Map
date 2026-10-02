# Dynamic Spatiotemporal Graph Neural Network for Real-Time Earthquake-Triggered Landslide Susceptibility Mapping

[![Project Phase](https://img.shields.io/badge/Project%20Phase-Phase%201%20Complete-blue.svg)](https://github.com/)
[![Institution](https://img.shields.io/badge/CHRIST%20(Deemed%20to%20be%20University)-School%20of%20Engineering%20%26%20Technology-navy.svg)](https://christuniversity.in)
[![Department](https://img.shields.io/badge/Department-AI%20%26%20Data%20Science%20Engineering-orange.svg)](https://christuniversity.in)
[![Python Version](https://img.shields.io/badge/Python-3.11%2B-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-red.svg)](https://pytorch.org)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.5%2B-purple.svg)](https://pyg.org)

---

## 📌 Project Information

- **Project Title:** Dynamic Spatiotemporal Graph Neural Network for Real-Time Earthquake-Triggered Landslide Susceptibility Mapping
- **Academic Program:** Bachelor of Technology in Computer Science and Engineering (Data Science) / AI and Data Science Engineering
- **Institution:** School of Engineering and Technology, CHRIST (Deemed to be University), Kumbalgodu, Bengaluru 560 074
- **Project Phase:** Phase 1 Final Report & Baseline Implementation (September 2026)
- **Student Authors:**
  - **B K Nandeish** (Register No: `2362315`)
  - **Harshitha N Reddy** (Register No: `2362332`)
- **Project Supervisor:** **Dr. Amos Bortiew**, Department of AI and Data Science Engineering
- **Department Head:** **Dr. Michael Moses T** | **Associate Dean:** **Dr. E A Mary Anita**

---

## 📖 Executive Summary & Abstract

Earthquake-triggered landslides represent a critical cascading hazard in mountainous terrain. Strong ground motion weakens slope materials and creates fractures, leaving slopes vulnerable to failure during the mainshock and over weeks of subsequent aftershocks. Conventional landslide susceptibility maps utilized by disaster management agencies (e.g., JMA, MLIT) are **static products** computed from pre-event topography and single-snapshot shaking footprints. Consequently, they fail to update as seismic shaking evolves across an aftershock sequence.

This research project introduces a **Dynamic Spatiotemporal Graph Neural Network** framework based on a **Diffusion Convolutional Recurrent Neural Network (DCRNN)**. The study region is discretized into a uniform **Uber H3 hexagonal graph** (Resolution 8, average cell area ~0.74 km²), where nodes ingest multi-source terrain, geological, fault, and ground deformation features, alongside time-varying seismic station recordings (PGA and PGV interpolated via Ordinary Kriging).

### Phase 1 Deliverables (This Repository)
1. **Nine-Stage Reproducible Data & Feature Pipeline:** Converts raw Digital Elevation Models (DEM), lithology maps, fault traces, coseismic deformation fields (InSAR / PIV), and landslide inventories into attributed H3 graphs.
2. **Spatial Block Cross-Validation Framework:** Implements a balanced spatial block assignment algorithm (Algorithm 1) to eliminate optimistic spatial autocorrelation leakage (Tobler’s First Law).
3. **Static Baseline Benchmarking:** Compares Logistic Regression, Random Forest, and a two-layer Graph Convolutional Network (GCN) across two independent major Japanese earthquakes:
   - **2024 Noto Peninsula Earthquake** (Magnitude Mw 7.5, 3,311 hexagonal cells, 15.83% positive rate).
   - **2018 Hokkaido Eastern Iburi Earthquake** (Magnitude Mw 6.6, 12,394 hexagonal cells, 2.51% positive rate).
4. **Phase 2 Dynamic Architecture Design:** Mathematical formulation for temporal seismic signal ingestion via DCRNN and holdout transfer validation on the **2016 Kumamoto Earthquake** (Magnitude Mw 7.0).

---

## 🎯 Research Objectives & Status

| No. | Objective Description | Current Status | Details / Deliverable |
|:---:|:--- |:---:|:--- |
| **1** | Develop a dynamic spatiotemporal graph neural network (DCRNN) for earthquake-induced landslide prediction. | **Designed** | DCRNN architecture & diffusion GRU operator formulated; implementation scheduled for Phase 2. |
| **2** | Integrate multi-source geospatial and seismic datasets into a unified graph representation. | **Completed (Static)** | Multi-source static feature pipeline complete for Noto and Hokkaido on H3 Grid (Res 8). Seismic data collected. |
| **3** | Update landslide susceptibility maps in real time after each significant aftershock. | **In Progress** | Data feeds (NIED K-NET/KiK-net & JMA catalogue) acquired; kriging interpolation pipeline defined. |
| **4** | Improve prediction accuracy compared with traditional static susceptibility models. | **Baselines Established** | Established benchmark static metrics (Random Forest AUC: 0.947 Noto, 0.960 Hokkaido). |
| **5** | Support emergency management and evacuation planning with continuously updated hazard maps. | **Groundwork Complete** | System architecture, computational cost estimation, and disaster response operational requirements documented. |

---

## 🛠️ Multi-Source Geospatial Datasets

| Dataset | Noto Peninsula Event (Mw 7.5) | Hokkaido Eastern Iburi Event (Mw 6.6) | Data Source / Provider |
|:--- |:--- |:--- |:--- |
| **Digital Elevation Model (DEM)** | JAXA AW3D30 (30 m resolution) | JAXA AW3D30 (30 m resolution via OpenTopography) | JAXA |
| **Geology & Lithology** | GSJ Seamless Digital Geological Map V2 (1:200k) | GSJ Seamless Digital Geological Map V2 (Nationwide) | Geological Survey of Japan (AIST) |
| **Landslide Inventory** | GSI DEM-differencing product (68,844 polygons) | GSI slope-failure & deposition distribution (3 GeoJSONs) | Geospatial Information Authority of Japan |
| **Ground Deformation** | NIED PIV magnitude (local & wide correlation windows) | Reconstructed quasi-EW & quasi-UD components from raw GSI InSAR (Paths 018, 116, 122) | NIED / GSI |
| **Active Fault Traces** | Notokaigan segment (stub mode) | GEM Global Active Faults Database (~13,500 nationwide traces filtered by bounding box) | GEM Foundation |
| **Earthquake Catalogue** | JMA Catalogue (Jan 2024 – Jul 2025, Magnitude >= 2.4) | JMA Catalogue (2018) | Japan Meteorological Agency |
| **Strong Motion Records** | NIED K-NET and KiK-net station time series | NIED K-NET and KiK-net station summaries | NIED |

---

## 📐 Methodology & System Architecture

### 1. Spatial Discretization (Uber H3 Grid)
The study areas are discretized using Uber's H3 hierarchical hexagonal spatial index at **Resolution 8** (average hexagon area ~0.74 km², edge length ~461 m). 
- **Noto Grid:** 3,311 land cells clipped to geology polygon extents.
- **Hokkaido Grid:** 12,394 valid land cells after dropping 2,915 unsurveyed peripheral cells outside InSAR/GSI survey footprints.
- Node adjacency creates an undirected graph G = (V, E) where interior nodes possess degree k = 6.

```
       / \     / \
      /   \___/   \
     \  *  / \  *  /     Node (V): H3 Cell (Res 8, ~0.74 km²)
      \___/ * \___/      Edge (E): Shared Hexagon Boundary
      /   \___/   \      Degree: 6 (Interior), <6 (Coastal)
     /  *  / \  *  /
     \___/     \___/
```

### 2. Feature Engineering & Vector Wrapping
Each H3 hexagon accumulates terrain, lithological, geological, and deformation features:
- **Elevation (Mean z):** Zonal mean from AW3D30 DEM.
- **Slope (Angle in degrees):** Zonal mean of finite-difference gradient:
  ```
  Slope Angle = arctan( sqrt( (dz/dx)^2 + (dz/dy)^2 ) )
  ```
- **Aspect (Mean Direction):** Circular vector mean wrapped to 0° to 360° to prevent arithmetic wrapping artifacts:
  ```
  Mean Aspect = atan2( sum(sin(theta_k)) / n, sum(cos(theta_k)) / n )
  ```
- **Curvature:** Zonal mean of second-order elevation differences.
- **Lithology:** Dominant geological class by area, one-hot encoded (rare classes pooled).
- **Fault Distance:** Geodesic distance from cell centroid to nearest GEM fault trace (in meters).
- **Ground Deformation:** Pixel-offset correlation magnitude (NIED PIV) for Noto; least-squares reconstructed quasi-East-West (d_EW) and quasi-Up-Down (d_UD) InSAR fields for Hokkaido:
  ```
  d_LOS,i = (e_i * d_EW) + (u_i * d_UD)   for satellite paths 018, 116, and 122
  ```

### 3. Label Construction (5% Coverage Threshold)
Landslide target label y_c (binary 0 or 1) is defined by a 5% cell coverage threshold to mitigate scale mismatch between small landslide scars (average scar area ~827 m²) and H3 cells (~740,000 m²):
```
y_c = 1  if  (Landslide Area in Cell / Total Cell Area) >= 0.05  else  0
```

### 4. Spatial Block Cross-Validation (Algorithm 1)
To avoid spatial autocorrelation leakage (Tobler’s First Law), training and test splits use spatial blocking:
- **Balanced Assignment:** Assigns spatial blocks b to folds K by minimizing joint disparity in cell count (S_b) and positive landslide count (P_b):
  ```
  Best Fold k* = argmin_k [ (p_k + P_b) / Total_P  +  (s_k + S_b) / Total_S ]
  ```
- **Noto Configuration:** 5 spatial block folds.
- **Hokkaido Configuration:** 3 spatial block folds (ensuring >= 82 test positives per fold).

---

## 📊 Phase 1 Baseline Results & Benchmarks

All models were evaluated under strict Spatial Block Cross-Validation using class-balanced weights.

### Performance Summary Table

| Study Region | Evaluation Setup | Model Architecture | AUC-ROC (Mean ± SD) | F1 Score | Key Feature Importance Drivers |
|:--- |:--- |:--- |:---:|:---:|:--- |
| **Noto Peninsula**<br>(Mw 7.5, Total Cells = 3,311,<br>Positive Rate: 15.83%) | 5 Spatial Folds | Logistic Regression | 0.931 | N/A | Ground deformation (local & wide) combined ~46%, Slope ~22%, Elevation & Curvature ~32%. |
| | | **Random Forest** | **0.947** | N/A |
| | | Static 2-Layer GCN | 0.924 ± 0.041 | N/A |
| **Hokkaido Eastern Iburi**<br>(Mw 6.6, Total Cells = 12,394,<br>Positive Rate: 2.51%) | 3 Spatial Folds | Logistic Regression | 0.943 ± 0.023 | 0.396 | Quasi-vertical deformation ranked 1st, followed by fault distance, elevation, slope, and quasi-EW deformation. |
| | | **Random Forest** | **0.960 ± 0.012** | **0.495** |
| | | Static 2-Layer GCN | 0.962 | 0.510 | *(Single-snapshot reference without temporal signals)* |

> [!NOTE]
> **Key Analytical Takeaways:**
> 1. **Random Forest Benchmark:** Random Forest produced the most robust static baselines across both regions due to its capacity to capture non-linear thresholds and interactions (e.g., deformation x slope).
> 2. **Static GCN Behavior:** The static GCN did not surpass Random Forest on Noto (0.924 vs 0.947). On a static table, graph convolution acts as spatial smoothing; it adds no temporal signal until time-series seismic records are ingested in Phase 2.
> 3. **Imbalance & F1 Gap:** High AUC-ROC (> 0.94) coupled with moderate F1 (~0.50) is characteristic of severe class imbalance (2.51% positive rate on Hokkaido), underscoring the operational importance of F1-tuned decision thresholds.

---

## ⚡ Phase 2 Dynamic Architecture: DCRNN Formulation

Phase 2 models aftershock dynamics by extending the H3 graph into a **Diffusion Convolutional Recurrent Neural Network (DCRNN)**:

```
[ Static Features s_i ] ──┐
                         ├──> [ Node Feature Input X(t) ] ──> [ DCRNN Layer (Diffusion GRU) ] ──> [ Updated Susceptibility p_i(t) ]
[ Seismic Kriging d_i^t ] ──┘
```

### Diffusion Convolution Operator
Diffusion convolution models bidirectional spatial spread on graph G using random walks:
```
Diffusion_Convolution(X) = sum over k from 0 to K-1 of:
  [ theta_(k,1) * (D_out^(-1) * W)^k  +  theta_(k,2) * (D_in^(-1) * W^T)^k ] * X
```
Where `D_out` and `D_in` are out-degree and in-degree matrices, and `W` is the weighted slope-connectivity adjacency matrix.

### Dynamic Recurrent Update Rules
The Diffusion Gated Recurrent Unit (DGRU) updates hidden states H(t) following aftershock t:
```
Reset Gate:     r(t) = sigmoid( Conv( [X(t), H(t-1)] ) + b_r )
Update Gate:    u(t) = sigmoid( Conv( [X(t), H(t-1)] ) + b_u )
Candidate Cell: C(t) = tanh( Conv( [X(t), r(t) * H(t-1)] ) + b_c )
Hidden State:   H(t) = u(t) * H(t-1)  +  (1 - u(t)) * C(t)
```

---

## 💻 Repository Structure & Code Architecture

```
.
├── config.py                       # Central typed dataclass registry (RegionConfig) & global paths
├── requirements.txt                # Required Python dependencies
├── build_snapshot_timeline.py     # Timeline builder for seismic event sequences
├── check_jma_feasibility.py        # JMA catalogue feasibility verification
├── derive_hokkaido_deformation.py # OLS InSAR decomposition (quasi-EW & quasi-UD)
├── fetch_usgs_catalog.py           # USGS earthquake catalog fetcher tool
├── generate_baseline_v2_reports.py # Baseline metric report generator
├── spatial_cv_sensitivity.py       # Spatial block size sensitivity analysis
├── visualize_predictions.py        # Map rendering & confusion matrix plotter
├── data/
│   ├── raw/                        # Event-specific raw datasets (Noto, Hokkaido, Kumamoto)
│   └── processed/                  # Processed H3 feature tables (.parquet) and graph exports
├── src/
│   ├── grid/
│   │   └── build_h3_grid.py        # Stage 1: Hexagonal spatial indexing (H3 Resolution 8)
│   ├── features/
│   │   ├── terrain_features.py     # Stage 2: Elevation, slope, aspect, curvature calculation
│   │   ├── geology_features.py     # Stage 3: Dominant lithology overlay & encoding
│   │   ├── fault_features.py       # Stage 4: Centroid-to-fault distance extraction
│   │   ├── deformation_features.py # Stage 5: InSAR/PIV ground deformation processing
│   │   ├── labels.py               # Stage 6: 5% landslide coverage thresholding
│   │   └── build_feature_table.py  # Stage 7: Feature merging & missing data filter
│   ├── eval/
│   │   ├── spatial_cv.py           # Balanced Spatial Block Assignment (Algorithm 1)
│   │   └── check_folds.py          # Spatial fold balance & leak diagnostic check
│   └── models/
│       ├── train_baseline.py       # Stage 8: Logistic Regression & Random Forest training
│       └── train_gnn.py            # Stage 9: Static PyG 2-layer GCN training
└── outputs/                        # Prediction maps, metrics (.json), and visual artifacts
```

---

## 🚀 Environment Setup & Execution Guide

### 1. Prerequisites & Installation

Ensure Python >= 3.11 and standard geospatial libraries (GDAL/GEOS) are available.

```bash
# Clone repository
git clone https://github.com/your-username/noto-landslide-dgnn.git
cd noto-landslide-dgnn

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Running the Full Phase 1 Pipeline

The pipeline is fully parameterised via `--region` (`noto` or `hokkaido`). Each stage reads from and writes to `data/processed/`, allowing isolated execution.

```bash
# --- Step 1: Spatial Grid Construction ---
python src/grid/build_h3_grid.py --region noto

# --- Steps 2-6: Feature Engineering Pipeline ---
python src/features/terrain_features.py --region noto
python src/features/geology_features.py --region noto
python src/features/fault_features.py --region noto
python src/features/deformation_features.py --region noto
python src/features/labels.py --region noto

# --- Step 7: Assemble Feature Table & Clean Missing Data ---
python src/features/build_feature_table.py --region noto

# --- Step 8: Train & Benchmark Baseline ML Models (Spatial CV) ---
python src/models/train_baseline.py --region noto

# --- Step 9: Train Static Graph Convolutional Network (GCN) ---
python src/models/train_gnn.py --region noto
```

To run the complete static baseline pipeline for **Hokkaido**:
```bash
# Execute deformation derivation for Hokkaido raw InSAR
python derive_hokkaido_deformation.py

# Run pipeline for Hokkaido
python src/features/build_feature_table.py --region hokkaido
python src/models/train_baseline.py --region hokkaido
python src/models/train_gnn.py --region hokkaido
```

---

## 🛡️ Technical Safeguards & Defect Log

During Phase 1 execution, several critical data handling defects were identified and systematically resolved:

| Pipeline Stage | Observed Defect / Symptom | Root Cause | Implemented Safeguard / Fix |
|:--- |:--- |:--- |:--- |
| **H3 Grid Generation** | ~575,000 cells generated for Noto | API version mismatch (H3 v3 vs v4 coordinate order swap) | Enforced H3 v4 API contract; added grid bounding box validation printouts. |
| **Terrain Features** | Aspect values restricted to 96° to 278° | Standard arithmetic mean applied to circular angular variable | Implemented vector-based circular mean (atan2 of unit sine/cosine sums). |
| **Label Construction** | 62% positive rate on Noto (unrealistic) | Any-touch spatial intersection rule triggered on sub-pixel scars | Replaced with 5% cell area coverage threshold (Landslide Area / Cell Area >= 0.05). |
| **Hokkaido DEM Processing** | All terrain feature values missing | Stale fallback path pointing to non-overlapping Noto DEM | Corrected raster paths; added spatial envelope intersection checks. |
| **Hokkaido Feature Table** | 2,915 of 15,309 cells missing data | Bounding box padded beyond InSAR and GSI survey footprint | Dropped unsurveyed cells prior to imputation to prevent false negative bias. |
| **Spatial Cross-Validation** | Folds without positive cells / extreme imbalance | Standard spatial blocking on highly clustered landslide occurrences | Implemented Algorithm 1 (Balanced spatial block assignment balancing count & area). |
| **Data Acquisition** | Garbled file names & wrong catalog data | Shift-JIS encoding issues; station summary CSVs mistaken for catalog | Re-extracted archives with correct codepage; validated JMA earthquake catalog. |

---

## 🌍 Societal, Environmental & SDG Impact

- **Environmental Impact:** Identifies weakened slopes subject to secondary failure, assisting river basin management against landslide damming and sediment runoff. Computing footprint is minimal (~17 kWh GPU training, ~12 kg CO2).
- **Societal Impact:** Empowers disaster management agencies (JMA/MLIT) with dynamic risk indicators for targeted road closures, evacuation routing, and community shelter management.
- **Sustainable Development Goals (SDG Compliance):**
  - **SDG 11 (Sustainable Cities & Communities - Target 11.5 & 11.b):** Reduces disaster-induced casualties and economic losses through enhanced hazard preparedness.
  - **SDG 13 (Climate Action - Target 13.1):** Strengthens resilience against compound natural hazards (earthquake-weakened slopes vulnerable to heavy precipitation).
  - **SDG 9 (Industry, Innovation & Infrastructure - Target 9.1 & 9.5):** Safeguards critical lifeline infrastructure through spatiotemporal AI innovation.

---

## 📜 Project Roadmap (Phase 2 & Future Scope)

```
[Jul 2026] Literature Review & Requirement Analysis   (Completed)
[Aug 2026] Dataset Acquisition & Cleaning             (Completed)
[Sep 2026] Phase 1: Static Graph & Baselines         (Completed - This Report)
[Oct 2026] DCRNN Architecture & Seismic Features      (In Progress)
[Nov 2026] Dynamic Model Training & Spatial Tuning    (Upcoming)
[Dec 2026] DeLong Tests, Ablations & Kumamoto Transfer(Upcoming)
[Jan 2027] Map Generation & System Documentation      (Planned)
[Feb 2027] Final Thesis Submission & Defense          (Planned)
```

---

## 👥 Authors & Academic Credits

**Research Team:**
- **B K Nandeish** (Register No: 2362315) — *Department of AI & Data Science Engineering, CHRIST (Deemed to be University)*
- **Harshitha N Reddy** (Register No: 2362332) — *Department of AI & Data Science Engineering, CHRIST (Deemed to be University)*

**Under the Guidance of:**
- **Dr. Amos Bortiew**, *Project Supervisor & Faculty Member, Department of AI & Data Science Engineering*

**Institutional Acknowledgements:**
We extend our gratitude to the **Geospatial Information Authority of Japan (GSI)**, **Geological Survey of Japan (AIST)**, **National Research Institute for Earth Science and Disaster Resilience (NIED)**, **Japan Meteorological Agency (JMA)**, **Japan Aerospace Exploration Agency (JAXA)**, and the **GEM Foundation** for providing public access to open scientific datasets.

---

## 📄 License & Usage Note

This project is developed for academic research purposes as part of the Bachelor of Technology degree requirements at CHRIST (Deemed to be University). All open datasets referenced remain subject to their respective issuing agency terms of use.#   D y n a m i c - S p a t i o t e m p o r a l - M a p  
 