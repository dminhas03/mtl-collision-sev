# Montreal Collision Severity Project
This project investigates whether machine learning models can predict road collisions in Montréal using 200K+ real-world traffic reports.

The pipeline integrates spatial, temporal, and contextual features (engineered with OSMnx and domain knowledge) to model collision risk. Logistic Regression serves as a baseline, while Random Forest achieves 91% AUROC, significantly outperforming baseline methods.


---

## 📊 Results

### Geospatial Visualization
Historical collision density across Montréal (200K+ reports).  
This highlights high-risk areas such as downtown Montréal and major intersections.  

![Collision Heatmap](reports/figures/collision_heatmap.png)  

👉 [View interactive version (HTML)](reports/figures/collision_heatmap.html)

---

### Model Evaluation

**ROC & Precision–Recall Curves**  
![ROC & PR](reports/figures/roc_pr_curves.png)

**Confusion Matrices**  
![Confusion Matrices](reports/figures/confusion_matrices.png)

**Calibration Curve**  
![Calibration Curve](reports/figures/calibration_curve.png)

**Top Feature Importances**  
![Feature Importances](reports/figures/feature_importances.png)

---

## TTechnical Details
- **Language & Libraries**: Python, scikit-learn, pandas, numpy, matplotlib, OSMnx, Folium  
- **Models**: Logistic Regression, Random Forest  
- **Evaluation Metrics**: ROC-AUC, PR-AUC, Calibration, Confusion Matrices  
- **Features**:  
  - Temporal (hour, day, month, weekday, season, rush hour, is_night)  
  - Spatial (latitude, longitude, route categories)  
  - Contextual (environment, road conditions, weather, etc.)  

---

## 🚧 Ongoing Work
- Extending geospatial visualization to **predicted risk maps** (model-based, not just historical).  
- Adding temporal layers (e.g., rush hour vs late night, seasonal trends).  
- Experimenting with deep learning models for spatio-temporal prediction.  
- Possible deployment via Streamlit/Dash for interactive exploration.  

---

##  Why This Matters
- **Urban planning**: identify high-risk intersections and corridors.  
- **Public safety**: support proactive interventions based on risk profiles.  
- **Applied ML**: real-world example of classification, feature engineering, and model interpretability.  

---

##  Status
This is an **ongoing project**. Current focus is on robust evaluation and exploratory visualization; extensions to deployment and predictive risk mapping are planned.  

