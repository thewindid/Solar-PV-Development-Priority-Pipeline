# ☀️ Urban Solar PV Prioritization: Geospatial Machine Learning Pipeline

An end-to-end automated geospatial data engineering and machine learning pipeline designed to evaluate, model, and prioritize building-scale Solar Photovoltaic (PV) deployment. By integrating physical irradiance modeling, urban economic valuation, and environmental degradation indices via satellite data, this system analyzes 36,642 urban buildings to provide highly reliable, data-driven pilot location recommendations.

## 🚀 Executive Summary & Project Impact

This project moves beyond simple spatial overlays by implementing a robust Multi-Criteria Decision Analysis (MCDA) framework reinforced by machine learning ensembles and probabilistic robustness testing.

*   **Total Validated Buildings:** 36,592 buildings (filtered from 36,642 raw geometries).
*   **Total Technical Capacity Projected:** 662.43 MWp.
*   **Total Annual Energy Production Projected:** 867.47 GWh/year.
*   **Robust Pilot Selection:** Successfully isolated 3,190 highly stable building candidates through 10,000 Monte Carlo simulations.

---

## 🛠️ Technology Stack & Dependencies

*   **Core Data Engineering:** Python, Pandas, NumPy
*   **Geospatial & Remote Sensing:** GeoPandas, Rasterio, Google Earth Engine (GEE), Folium, Inverse Distance Weighting (IDW)
*   **Machine Learning (scikit-learn):** Random Forest, Gradient Boosting Regressor (GBR), XGBoost, Principal Component Analysis (PCA), K-Means Clustering
*   **Decision Science (MCDA):** Analytical Hierarchy Process (AHP), Weighted Linear Combination (WLC), TOPSIS, Dirichlet Monte Carlo Simulations

---

## 🏗️ System Architecture & Pipeline Workflow

The automated pipeline is systematically divided into four consecutive operational tasks, executing complex geospatial fusions in under 15 minutes total processing time.

### Phase 1: Solar PV Potential Index (Task 1)
This module handles the extraction and physical modeling of solar energy potential across building footprints.
*   **Geospatial Pre-processing:** Transformed coordinate reference systems (CRS to EPSG:32648) and calculated effective roof areas. 
*   **Missing Data Imputation:** Resolved severe data sparsity by engineering a Random Forest spatial smoothing model to impute 36,579 missing building height values, achieving an R² of 0.2098 and an MAE of 0.12m.
*   **Advanced Irradiance Modeling:** Calculated Plane of Array (POA) irradiance using the Hay-Davies model, yielding a +13.5% tilt gain compared to standard Global Horizontal Irradiance (GHI).
*   **Performance:** Evaluated mutual building and cloud shading algorithms (SF_total mean: 0.919) across the entire dataset in just 26 seconds.

### Phase 2: Economic Value Index (Task 2)
This module builds an Urban Gravity Model to estimate the spatial economic viability of PV installations.
*   **Geospatial Feature Engineering:** Extracted and engineered advanced features from OpenStreetMap (OSM), including Point of Interest (POI) density, economic anchors, road networks, public transportation access, and land-use scoring.
*   **Machine Learning Valuation:** Trained a Random Forest Regressor (5-Fold Cross-Validation) to model spatial economic variance. The model achieved exceptional accuracy with an R² of 0.9970 and an MAE of 0.0348.
*   **Feature Importance:** Identified `landuse_score` (0.458), `area_m2` (0.391), and `poi_density` (0.143) as the primary drivers of urban economic value.
*   **Performance:** Executed feature extraction and model inference in 77 seconds.

### Phase 3: Environmental Degradation Index (Task 3)
This module maps environmental vulnerability by fusing multiple satellite imagery sources via Google Earth Engine.
*   **Satellite Data Fusion:** Interpolated high-resolution data points for Aerosol Optical Depth (AOD), Land Surface Temperature (LST), NDVI, NDBI, Cloud Fraction (CF), and Land Cover (LC). Data directionality was adjusted (e.g., NDVI inverted) to uniformly represent degradation.
*   **Dimensionality Reduction & Clustering:** Deployed PCA to reduce noise, successfully retaining 97.8% of the data variance across the first four principal components. Applied K-Means clustering (optimal k=2, silhouette score = 0.4619) to segment environmental degradation zones.
*   **Spatial Smoothing Ensemble:** Applied Gradient Boosting (GBR) to smooth the final environmental index, achieving an R² of 0.9980. The top contributing indicators were LST (0.550), AOD (0.254), and NDBI (0.091).
*   **Performance:** Processed and validated the environmental spatial model in 155 seconds.

### Phase 4: Multi-Criteria Integration & Prioritization (Task 4)
The final production module integrates outputs from Phase 1, 2, and 3 to finalize building rankings.
*   **MCDA Implementation:** Executed parallel scoring algorithms including 5 AHP scenarios (all maintaining Consistency Ratios < 0.01), 6 WLC scenarios, and TOPSIS base scoring.
*   **Ensemble ML Scoring:** Fused traditional MCDA with machine learning regressors (Random Forest R²=0.9984, GBR R²=0.9993, XGBoost R²=0.9958) to dynamically adjust priority weights based on spatial interactions.
*   **Robustness via Monte Carlo:** Ran 10,000 Dirichlet simulations to test scoring stability under extreme weight variations. Identified 3,190 "Stable" buildings that maintained a ≥70% probability of staying within the top-20% priority rank.
*   **Quadrant Analysis:** Mapped comparative operational viability (e.g., identifying 12,795 "Star" buildings in the PV vs. Economic quadrant).
*   **Performance:** Finalized the integration, simulations, and exported all ranking datasets in 400 seconds.

---

## 🏆 Final Results: Top Pilot Candidates

The pipeline systematically narrowed down 36,592 buildings to the most statistically robust candidates. The top 3 recommended buildings for Phase 1 pilot deployment are:

1.  **Bldg-94** | Final Score: `1.0000` | (PV: 1.000, Econ: 0.718, Env: 0.600) 🔒 *Highly Stable*
2.  **Bldg-36152** | Final Score: `0.9579` | (PV: 0.823, Econ: 0.845, Env: 0.746) 🔒 *Highly Stable*
3.  **Bldg-36346** | Final Score: `0.9162` | (PV: 0.790, Econ: 0.849, Env: 0.636) 🔒 *Highly Stable*

---

## 📂 Output Artifacts & Visualizations
The pipeline automatically generates multiple visual and analytical artifacts stored in the `/output/` directory:
*   **Interactive Maps:** Multi-layer Folium HTML maps for prioritized zones.
*   **Static Cartography:** High-resolution choropleth `.png` maps for each domain index.
*   **Data Dashboards:** 8-panel analysis charts, AHP network diagrams, Monte Carlo distribution plots, and sensitivity tornado charts.
*   **Geospatial Files:** Standardized `.geojson` and `.csv` files for direct GIS software ingestion.

---
**Author:** Bayu Kurniawan
*Driven by data, powered by machine learning, engineered for sustainable urban energy.*
