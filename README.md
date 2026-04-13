# Tripartite Causal Fairness Decomposition of 707 Indian Districts

**Structural vs. Agential Causal Pathways to Health Outcome Inequality**  
A Tripartite Causal Fairness Decomposition of 707 Indian Districts using NFHS-5 (2019–21)  
with Proof-of-Concept Transportability to Nigeria, Bangladesh, Kenya, and Cambodia

---

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![DOI](https://img.shields.io/badge/DOI-10.2471%2FBLT.23.290854-green.svg)](https://doi.org/10.2471/BLT.23.290854)
[![Walsh College DBA](https://img.shields.io/badge/Walsh%20College-DBA%20AI%2FML-navy.svg)](https://walshcollege.edu)

---

## Author

**Karthikeyan Venkatesan**  
Lead Systems Engineer II, FORVIA Hella India Automotive Ltd, Pune  
DBA Candidate (AI/ML) — Walsh College, USA & Deakin University, Australia  
MSc Data Science — Deakin University  
GL Mentor: Somak Sengupta  
Supervisor: Prof. Javad Katibai, Walsh College  

---

## Overview

This repository contains the complete data pipeline, analytical notebook, and supporting code for the DBA capstone study applying the **Tripartite Causal Fairness Decomposition (TCFD)** framework to district-level health outcome inequality in India.

### The TCFD Framework

The TCFD classifies all causal pathways driving health inequality into three analytically distinct types:

| Type | Category | Examples | Policy Implication |
|------|----------|----------|-------------------|
| **Type I** | Structural-Physical Constraints | Geographic remoteness, terrain, population density | Infrastructure investment, telemedicine |
| **Type II** | Historically-Produced Injustice | Caste exclusion, gender discrimination, colonial gaps | Redistributive policy, ASHA density |
| **Type III** | Policy-Actionable Levers | PM-JAY enrollment, institutional delivery, immunisation | Implementation reform, scheme expansion |

### Key Finding (Preliminary)
- Median UHCd = **44.75%** (range: 26.42%–69.40%) across 707 districts
- Type I (geography tax) dominates in **Thar Desert, Himalayan, and North-East** districts
- Type III (policy) dominates in **South India and Maharashtra** — precisely where PM-JAY impact is largest
- PM-JAY DiD ATT estimated at **+3–5 pp UHCd** in Type III-dominant clusters vs. minimal effect in Type I-dominant clusters

---

## Repository Structure

```
tcfd-india-707/
├── data/
│   ├── master_final_v3.csv          # 707 districts × 171 variables (primary dataset)
│   ├── uhcd_extracted.csv           # UHCd scores extracted from Mukherji et al. 2024 PDF
│   ├── gadm_geo_features.csv        # Geographic variables from GADM v4.1 shapefiles
│   └── tcfd_taxonomy.json           # Variable → Type I/II/III classification
├── notebooks/
│   └── TCFD_Selfcontained.ipynb     # Complete analysis pipeline (43 cells)
├── src/
│   ├── data_pipeline/
│   │   ├── master_merge.py          # Builds master_final_v3.csv from 10 sources
│   │   ├── uhcd_extractor.py        # Extracts UHCd from Mukherji et al. PDF
│   │   ├── geo_features.py          # Computes geography tax from GADM shapefiles
│   │   ├── pmjay_extractor.py       # PM-JAY state data from Parliament Q&A
│   │   └── nha_extractor.py         # NHA DIU/cards/hospitals data
│   ├── models/
│   │   ├── xgboost_shap.py          # Stage 1: XGBoost + SHAP attribution
│   │   ├── did_estimator.py         # Stage 2: Callaway-Sant'Anna DiD
│   │   ├── kmeans_clustering.py     # Stage 3: K-means on SHAP matrix
│   │   └── transportability.py      # Stage 4: Cross-country TCFD validation
│   └── utils/
│       ├── district_normaliser.py   # Standardises district name joins
│       └── validators.py            # Data quality validation checks
├── docs/
│   ├── data_dictionary.md           # Full 171-variable data dictionary
│   ├── data_sources.md              # Source citations with URLs
│   ├── tcfd_methodology.md          # TCFD framework specification
│   └── validation_report.md        # 10-point data validation results
├── results/
│   ├── figures/                     # TCFD choropleth maps, SHAP plots
│   └── tables/                      # Cluster profiles, DiD estimates
├── tests/
│   └── test_pipeline.py             # Unit tests for data pipeline
├── requirements.txt
├── environment.yml
├── .gitignore
└── README.md
```

---

## Data Sources

| Source | Coverage | Key Variables | Citation |
|--------|----------|---------------|----------|
| NFHS-5 (2019–21) | 707 districts | 109 health indicators | IIPS/MoHFW (2021) |
| Mukherji et al. 2024 UHCd | 525/707 districts | UHCd, 5 domain scores | doi:10.2471/BLT.23.290854 |
| PM-JAY (NHA/Parliament) | 36 States/UTs | Treatment flag, hospitalizations 2018–24 | NHA; data.gov.in RS-265 Q1723 |
| IHME GBD 2023 | 32 states | DALY rates (2021) | vizhub.healthdata.org |
| Census 2011 A-1 | 640/707 districts | Population, area, density | censusindia.gov.in |
| GADM v4.1 Shapefiles | 580/707 districts | Area, compactness, remoteness | gadm.org (CC BY 4.0) |
| NHA Annual Reports | 36 states | DIU, Ayushman cards, hospitals | nha.gov.in; Joseph et al. 2021 |
| NFHS Policy Tracker | 115 districts | Boundary change flag | Appendix A |

**Dataset note:** `master_final_v3.csv` is the primary analysis file (707 rows × 171 columns, 683 KB). Raw source files are not redistributed due to size; download links are in `docs/data_sources.md`.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/karthivenkatesan/tcfd-india-707.git
cd tcfd-india-707

# Create conda environment
conda env create -f environment.yml
conda activate tcfd-india

# Or install with pip
pip install -r requirements.txt
```

---

## Quick Start

```python
# Run the complete pipeline from scratch
python src/data_pipeline/master_merge.py

# Or open the self-contained notebook
jupyter notebook notebooks/TCFD_Selfcontained.ipynb
```

### Running Individual Stages

```python
# Stage 1: XGBoost-SHAP attribution
python src/models/xgboost_shap.py --data data/master_final_v3.csv --output results/

# Stage 2: Callaway-Sant'Anna DiD
python src/models/did_estimator.py --treatment pmjay_treatment_flag --outcome UHCd

# Stage 3: K-means clustering on SHAP matrix
python src/models/kmeans_clustering.py --k 5 --shap_matrix results/shap_matrix.npy

# Stage 4: Transportability validation
python src/models/transportability.py --countries NGA,BGD,KEN,KHM
```

---

## Reproducibility

All analyses use fixed random seeds:
```python
import numpy as np
import random
np.random.seed(42)
random.seed(42)
# XGBoost: random_state=42
# K-means: random_state=42, n_init=10
```

Python version: 3.11.x  
All package versions pinned in `requirements.txt`

---

## Citation

If you use this code or data, please cite:

```bibtex
@misc{venkatesan2026tcfd,
  author       = {Karthikeyan Venkatesan},
  title        = {Tripartite Causal Fairness Decomposition of 707 Indian Districts:
                  Structural vs. Agential Causal Pathways to Health Outcome Inequality},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/karthivenkatesan/tcfd-india-707},
  note         = {DBA Capstone, Walsh College \& Deakin University}
}
```

Also cite the primary UHCd data source:
> Mukherji A, Rao M, Desai S, Subramanian SV, Kang G, Patel V. District-level monitoring of universal health coverage, India. *Bull World Health Organ* 2024;102:630–638B. doi:10.2471/BLT.23.290854

---

## Licence

Code: MIT License  
Data (master_final_v3.csv): CC BY 4.0  
UHCd scores: CC BY IGO 3.0 (World Health Organization, per Mukherji et al. 2024)

---

## Contact

Karthikeyan Venkatesan — kar.venkat@gmail.com  
Issues and pull requests welcome.
