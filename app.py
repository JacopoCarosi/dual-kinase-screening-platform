import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, DataStructs
from rdkit.Chem.Draw import MolToImage

# Configurazione della pagina
st.set_page_config(
    page_title="Dual Kinase Screening Platform (EGFR & HER2)",
    layout="wide"
)

st.title("Dual Kinase In-Silico Screening Platform")
st.markdown("Piattaforma di screening molecolare per l'inibizione di **EGFR** e **HER2** con valutazione del Dominio di Applicabilità e Proprietà ADMET.")

# ----------------------------------------------------
# 1. CARICAMENTO ASSET E MODELLI
# ----------------------------------------------------
@st.cache_resource
def load_assets():
    models = {}
    fps = {}

    # File paths
    egfr_model_path = "egfr_model.pkl"
    egfr_fps_path = "egfr_train_fps.pkl"
    her2_model_path = "best_XG_HER2.pkl"
    her2_fps_path = "HER2_train_fps.pkl"

    # Caricamento EGFR
    if os.path.exists(egfr_model_path) and os.path.exists(egfr_fps_path):
        models["EGFR"] = joblib.load(egfr_model_path)
        fps["EGFR"] = joblib.load(egfr_fps_path)
    
    # Caricamento HER2
    if os.path.exists(her2_model_path) and os.path.exists(her2_fps_path):
        models["HER2"] = joblib.load(her2_model_path)
        fps["HER2"] = joblib.load(her2_fps_path)

    return models, fps

loaded_models, loaded_fps = load_assets()

# Sidebar di stato
with st.sidebar:
    st.header("Stato dei Modelli")
    if "EGFR" in loaded_models:
        st.success("Modello EGFR (Random Forest): Caricato")
    else:
        st.error("Modello EGFR: File non trovato")

    if "HER2" in loaded_models:
        st.success("Modello HER2 (XGBoost): Caricato")
    else:
        st.error("Modello HER2: File non trovato")

    st.markdown("---")
    target_choice = st.selectbox(
        "Seleziona modalità predittiva:",
        ["Analisi Duale (EGFR + HER2 & Selettività)", "Solo EGFR", "Solo HER2"]
    )

# ----------------------------------------------------
# 2. FUNZIONI HELPER
# ----------------------------------------------------
def calculate_lipinski(mol):
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    return {
        "MW (Da)": round(mw, 2),
        "LogP": round(logp, 2),
        "H-Bond Donors": hbd,
        "H-Bond Acceptors": hba,
        "Lipinski Violations": violations
    }

def get_morgan_fp(mol, n_bits=2048, radius=2):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)

def check_applicability_domain(query_fp, train_fps, threshold=0.35):
    similarities = DataStructs.BulkTanimotoSimilarity(query_fp, train_fps)
    max_sim = max(similarities)
    mean_sim = np.mean(similarities)
    is_in_domain = max_sim >= threshold
    return round(max_sim, 3), round(mean_sim, 3), is_in_domain

# ----------------------------------------------------
# 3. INTERFACCIA E INFERENZA
# ----------------------------------------------------
smiles_input = st.text_input(
    "Inserisci SMILES canonico della molecola da analizzare:",
    "COc1cc2c(ncnc2cc1OCCCN3CCOCC3)Nc4ccc(c(c4)Cl)F"  # Gefitinib come default
)

if st.button("Esegui Screening", type="primary"):
    mol = Chem.MolFromSmiles(smiles_input)

    if mol is None:
        st.error("Stringa SMILES non valida. Verifica la struttura inserita.")
    else:
        col_img, col_admet = st.columns([1, 2])

        with col_img:
            st.subheader("Struttura 2D")
            img = MolToImage(mol, size=(320, 260))
            st.image(img, use_container_width=True)

        with col_admet:
            st.subheader("Profilo Fisico-Chimico (Lipinski)")
            admet_props = calculate_lipinski(mol)
            df_admet = pd.DataFrame([admet_props])
            st.dataframe(df_admet, hide_index=True)
            if admet_props["Lipinski Violations"] == 0:
                st.success("Tutte le regole di Lipinski sono rispettate (Drug-like).")
            else:
                st.warning(f"Rilevate {admet_props['Lipinski Violations']} violazioni delle regole di Lipinski.")

        st.markdown("---")
        st.subheader("Risultati Predittivi Bioattività & Applicability Domain")

        query_fp = get_morgan_fp(mol)
        query_fp_array = np.array(query_fp).reshape(1, -1)

        # Predizione EGFR
        egfr_pred = None
        if target_choice in ["Analisi Duale (EGFR + HER2 & Selettività)", "Solo EGFR"]:
            if "EGFR" in loaded_models:
                egfr_pred = float(loaded_models["EGFR"].predict(query_fp_array)[0])
                max_sim_egfr, _, in_domain_egfr = check_applicability_domain(query_fp, loaded_fps["EGFR"])
            else:
                st.error("Asset EGFR non caricati correttamente.")

        # Predizione HER2
        her2_pred = None
        if target_choice in ["Analisi Duale (EGFR + HER2 & Selettività)", "Solo HER2"]:
            if "HER2" in loaded_models:
                her2_pred = float(loaded_models["HER2"].predict(query_fp_array)[0])
                max_sim_her2, _, in_domain_her2 = check_applicability_domain(query_fp, loaded_fps["HER2"])
            else:
                st.error("Asset HER2 non caricati correttamente.")

        # Rendering Colonne Risultati
        c1, c2 = st.columns(2)

        if egfr_pred is not None:
            with c1:
                st.markdown("### Target: EGFR")
                st.metric("pIC50 stimato", f"{egfr_pred:.2f}")
                est_ic50_nm = 10 ** (9 - egfr_pred)
                st.caption(f"IC50 equivalente: **{est_ic50_nm:.1f} nM**")
                
                if in_domain_egfr:
                    st.success(f"In-Domain (Max Tanimoto: {max_sim_egfr})")
                else:
                    st.warning(f"Fuori Dominio (Max Tanimoto: {max_sim_egfr} < 0.35)")

        if her2_pred is not None:
            with c2:
                st.markdown("### Target: HER2")
                st.metric("pIC50 stimato", f"{her2_pred:.2f}")
                est_ic50_nm_her2 = 10 ** (9 - her2_pred)
                st.caption(f"IC50 equivalente: **{est_ic50_nm_her2:.1f} nM**")
                
                if in_domain_her2:
                    st.success(f"In-Domain (Max Tanimoto: {max_sim_her2})")
                else:
                    st.warning(f"Fuori Dominio (Max Tanimoto: {max_sim_her2} < 0.35)")

        # Calcolo Selettività Duale
        if target_choice == "Analisi Duale (EGFR + HER2 & Selettività)" and egfr_pred is not None and her2_pred is not None:
            st.markdown("---")
            st.subheader("Stima della Selettività Isoforme")
            delta_pic50 = egfr_pred - her2_pred
            fold_diff = 10 ** abs(delta_pic50)

            if abs(delta_pic50) < 0.5:
                st.info(f"**Profilo Duale Bilanciato**: Differenza di affinità limitata (ΔpIC50 = {delta_pic50:+.2f}). Il composto si comporta come inibitore duale EGFR/HER2 con selettività comparabile ({fold_diff:.1f}x).")
            elif delta_pic50 >= 0.5:
                st.success(f"**Selettivo per EGFR**: Il composto mostra una preferenza per EGFR di circa **{fold_diff:.1f} volte** rispetto a HER2 (ΔpIC50 = {delta_pic50:+.2f}).")
            else:
                st.success(f"**Selettivo per HER2**: Il composto mostra una preferenza per HER2 di circa **{fold_diff:.1f} volte** rispetto a EGFR (ΔpIC50 = {delta_pic50:+.2f}).")