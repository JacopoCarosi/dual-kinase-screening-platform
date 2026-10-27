#   Dual Kinase In-Silico Screening Platform (EGFR & HER2)

A machine learning-driven cheminformatics web application designed to predict compound bioactivity ($pIC_{50}$) and evaluate isoform selectivity against **EGFR** (`CHEMBL203`) and **HER2/ErbB2** (`CHEMBL1824`).

The application integrates automated molecular descriptor calculation, chemical space validation via **Applicability Domain (Tanimoto Similarity)**, and physicochemical drug-likeness profiling (**Lipinski's Rule of Five**).

---

##   Project Overview & Rational Design

EGFR and HER2 belong to the ErbB family of receptor tyrosine kinases, sharing over 80% catalytic kinase domain sequence identity. 
* **Target Duality & Resistance:** In oncological clinical settings, EGFR inhibition often triggers compensatory survival signaling via HER2. Dual EGFR/HER2 inhibitors (e.g., Lapatinib) mitigate this bypass pathway.
* **Selectivity Assessment:** Excessive EGFR inhibition leads to dose-limiting on-target cutaneous and gastrointestinal toxicities. This platform quantifies estimated selectivity ratios to identify dual vs. target-selective scaffolds.

---

##   Computational Methodology

1. **Data Curation & ChEMBL Pipeline:**
   - Bioactivity data extracted from ChEMBL database using `chembl_webresource_client`.
   - Filtered for homogenous biochemical assays: standard type `IC50`, relation `=`, and units `nM`.
   - Data deduplication and logarithmic conversion:
     $$pIC_{50} = 9 - \log_{10}(IC_{50})$$

2. **Featurization:**
   - 2D chemical structures processed via **RDKit**.
   - Generated Circular Fingerprints (**Morgan Fingerprints / ECFP4 equivalents**, radius = 2, 2048-bit density).

3. **Machine Learning Models & Performance:**
   - **EGFR:** Random Forest Regressor (`best_rf_model.pkl`).
   - **HER2:** Gradient Boosting via XGBoost (`best_XG_HER2.pkl`).

| Target | Algorithm | Test $R^2$ | Test RMSE |
| :--- | :--- | :---: | :---: |
| **EGFR** (`CHEMBL203`) | Random Forest | ~0.68 | ~0.66 |
| **HER2** (`CHEMBL1824`) | XGBoost Regressor | 0.691 | 0.660 |

4. **Rigorous Applicability Domain (AD):**
   - Implemented Bulk Tanimoto similarity against the training library fingerprints.
   - Molecules falling below the similarity threshold ($Tanimoto_{max} < 0.35$) are flagged as out-of-domain to prevent algorithmic extrapolation errors.

5. **Physicochemical Profiling (ADMET):**
   - Real-time screening of Lipinski's Rule of 5: Molecular Weight (MW), LogP, Hydrogen Bond Donors (HBD), and Hydrogen Bond Acceptors (HBA).

---

## Quickstart & Local Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<YOUR-USERNAME>/<YOUR-REPO-NAME>.git
   cd <YOUR-REPO-NAME>