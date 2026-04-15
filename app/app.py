import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetic Readmission Predictor",
    page_icon="🏥",
    layout="wide"
)

# ── CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stApp { font-family: 'Segoe UI', sans-serif; }
    .title-box {
        background: linear-gradient(135deg, #1a6eaa, #2196F3);
        padding: 20px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
    }
    .title-box h1 { margin: 0; font-size: 1.8rem; }
    .title-box p  { margin: 5px 0 0; opacity: 0.85; font-size: 0.95rem; }
    .section-header {
        background: #e3f2fd;
        border-left: 4px solid #2196F3;
        padding: 8px 14px;
        border-radius: 0 8px 8px 0;
        font-weight: 600;
        color: #1565C0;
        margin: 18px 0 12px;
    }
    .result-card {
        padding: 22px 26px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 14px;
    }
    .result-danger  { background: #FFEBEE; border: 2px solid #EF5350; }
    .result-safe    { background: #E8F5E9; border: 2px solid #66BB6A; }
    .result-warning { background: #FFF8E1; border: 2px solid #FFA726; }
    .result-card h2 { margin: 0 0 6px; font-size: 1.1rem; }
    .result-card .proba { font-size: 2.2rem; font-weight: 700; margin: 4px 0; }
    .result-card .label { font-size: 1rem; font-weight: 600; }
    .metric-row { display: flex; gap: 10px; margin-top: 16px; }
    .info-box {
        background: #fff;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.85rem;
        color: #555;
        margin-top: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model        = joblib.load("classifier_chain_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    thresholds   = joblib.load("best_thresholds.pkl")
    return model, preprocessor, thresholds

model, preprocessor, thresholds = load_artifacts()


def chain_predict_proba(chain_model, X):
    """
    Reconstruit predict_proba manuellement pour ClassifierChain.

    ClassifierChain avec 2 cibles [readmitted_bin, change_bin] :
      - estimators_[0] : prédit readmitted_bin sur X seul
      - estimators_[1] : prédit change_bin sur [X | pred_0]  (la chaîne)

    Retourne (proba_readmit, proba_change) — deux floats entre 0 et 1.
    """
    import scipy.sparse as sp

    # --- Estimateur 0 : readmitted_bin ---
    est0   = chain_model.estimators_[0]
    proba0 = float(est0.predict_proba(X)[0, 1])   # P(readmitted=1)

    # --- Concatène X + proba_0 pour alimenter l'estimateur 1 ---
    aug = np.array([[proba0]])
    if sp.issparse(X):
        X1 = sp.hstack([X, sp.csr_matrix(aug)])
    else:
        X1 = np.hstack([X, aug])

    # --- Estimateur 1 : change_bin ---
    est1   = chain_model.estimators_[1]
    proba1 = float(est1.predict_proba(X1)[0, 1])  # P(change=1)

    return proba0, proba1


# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div class="title-box">
    <h1>🏥 Diabetic Patient Readmission Predictor</h1>
    <p>Prédiction de réadmission hospitalière et changement de traitement — ClassifierChain (Random Forest)</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/caduceus.png", width=60)
    st.markdown("### À propos")
    st.markdown("""
Ce modèle prédit deux résultats cliniques :

🔴 **Réadmission < 30j**
Probabilité qu'un patient diabétique soit réhospitalisé dans les 30 jours.

🔵 **Changement de traitement**
Probabilité qu'un ajustement de traitement soit nécessaire.

---
**Modèle :** ClassifierChain (Random Forest)
**Dataset :** UCI Diabetes 130-US Hospitals
**Seuil readmitted :** `{:.3f}`
**Seuil change :** `{:.3f}`
""".format(thresholds['readmitted_bin'], thresholds['change_bin']))

# ── Démographiques ────────────────────────────────────────────────
st.markdown('<div class="section-header">👤 Informations démographiques</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    age = st.selectbox("Tranche d'âge", [
        '[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
        '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'
    ], index=6)
    age_num = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
               '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'].index(age)
with col2:
    gender = st.selectbox("Genre", ['Female', 'Male'])
with col3:
    race = st.selectbox("Race / Origine", [
        'AfricanAmerican', 'Asian', 'Caucasian', 'Hispanic', 'Other', 'Unknown'
    ], index=2)

# ── Hospitalisation ───────────────────────────────────────────────
st.markdown('<div class="section-header">🏨 Informations hospitalières</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

ADMISSION_TYPES = {
    1: "1 — Urgence", 2: "2 — Urgence", 3: "3 — Élective",
    4: "4 — Nouveau-né", 5: "5 — Pas d'information",
    6: "6 — NULL", 7: "7 — Traumatisme", 8: "8 — Pas d'information"
}
DISCHARGE_TYPES = {
    1: "1 — Domicile", 2: "2 — Soins courts", 3: "3 — Maison de soins",
    6: "6 — Domicile avec soins", 11: "11 — Décès", 13: "13 — Soins à domicile",
    18: "18 — Pas d'information", 22: "22 — Réadmission", 25: "25 — Psychiatrique"
}
ADMISSION_SOURCES = {
    1: "1 — Médecin référent", 2: "2 — Clinique",
    3: "3 — HMO", 4: "4 — Transfert hôpital",
    7: "7 — Urgences", 9: "9 — Pas d'information"
}

with col1:
    admission_type_id = st.selectbox("Type d'admission",
        list(ADMISSION_TYPES.keys()), format_func=lambda x: ADMISSION_TYPES[x])
with col2:
    discharge_disposition_id = st.selectbox("Type de sortie",
        list(DISCHARGE_TYPES.keys()), format_func=lambda x: DISCHARGE_TYPES[x])
with col3:
    admission_source_id = st.selectbox("Source d'admission",
        list(ADMISSION_SOURCES.keys()), format_func=lambda x: ADMISSION_SOURCES[x])
with col4:
    time_in_hospital = st.slider("Durée d'hospitalisation (jours)", 1, 14, 4)

# ── Examens & médicaments ─────────────────────────────────────────
st.markdown('<div class="section-header">🧪 Examens & Médicaments</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    num_lab_procedures = st.slider("Analyses de laboratoire", 1, 132, 43)
with col2:
    num_procedures     = st.slider("Procédures médicales", 0, 6, 1)
with col3:
    num_medications    = st.slider("Nombre de médicaments", 1, 81, 16)
with col4:
    number_diagnoses   = st.slider("Nombre de diagnostics", 1, 16, 7)

# ── Visites antérieures ───────────────────────────────────────────
st.markdown('<div class="section-header">📋 Historique des visites</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    number_outpatient = st.number_input("Visites ambulatoires", 0, 42, 0)
with col2:
    number_emergency  = st.number_input("Visites aux urgences", 0, 76, 0)
with col3:
    number_inpatient  = st.number_input("Hospitalisations antérieures", 0, 21, 0)

# ── Résultats biologiques ─────────────────────────────────────────
st.markdown('<div class="section-header">🔬 Résultats biologiques</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    A1Cresult     = st.selectbox("Résultat HbA1c (A1C)", ['None', 'Norm', '>7', '>8'])
with col2:
    max_glu_serum = st.selectbox("Glycémie max sérique", ['None', 'Norm', '>200', '>300'])

# ── Diagnostics ───────────────────────────────────────────────────
st.markdown('<div class="section-header">🩺 Diagnostics principaux</div>', unsafe_allow_html=True)
diag_options = ['Autre', 'Blessure', 'Circulatoire', 'Diabète', 'Digestif',
                'Génito-urinaire', 'Musculo-squelettique', 'Néoplasme', 'Respiratoire']

col1, col2, col3 = st.columns(3)
with col1:
    diag_1 = st.selectbox("Diagnostic principal (diag_1)", diag_options, index=3)
with col2:
    diag_2 = st.selectbox("Diagnostic secondaire (diag_2)", diag_options, index=2)
with col3:
    diag_3 = st.selectbox("Diagnostic tertiaire (diag_3)", diag_options, index=0)

# ── Feature engineering ───────────────────────────────────────────
is_senior          = int(age_num >= 6)
total_visits       = number_outpatient + number_emergency + number_inpatient
procedures_per_day = num_procedures / time_in_hospital if time_in_hospital > 0 else 0
meds_per_diag      = num_medications / number_diagnoses if number_diagnoses > 0 else 0
nb_visites_patient = total_visits

# ── Predict button ────────────────────────────────────────────────
st.markdown("---")
predict_btn = st.button("🔍 Lancer la prédiction", type="primary", use_container_width=True)

if predict_btn:

    input_data = pd.DataFrame([{
        'race'                      : race,
        'gender'                    : gender,
        'admission_type_id'         : admission_type_id,
        'discharge_disposition_id'  : discharge_disposition_id,
        'admission_source_id'       : admission_source_id,
        'time_in_hospital'          : time_in_hospital,
        'num_lab_procedures'        : num_lab_procedures,
        'num_procedures'            : num_procedures,
        'num_medications'           : num_medications,
        'number_outpatient'         : number_outpatient,
        'number_emergency'          : number_emergency,
        'number_inpatient'          : number_inpatient,
        'diag_1'                    : diag_1,
        'diag_2'                    : diag_2,
        'diag_3'                    : diag_3,
        'number_diagnoses'          : number_diagnoses,
        'max_glu_serum'             : max_glu_serum,
        'A1Cresult'                 : A1Cresult,
        'age_num'                   : age_num,
        'is_senior'                 : is_senior,
        'total_visits'              : total_visits,
        'procedures_per_day'        : procedures_per_day,
        'meds_per_diag'             : meds_per_diag,
        'nb_visites_patient'        : nb_visites_patient,
    }])

    # Preprocessing
    X_proc = preprocessor.transform(input_data)

    # Prédiction via les estimateurs internes (contourne le bug predict_proba)
    try:
        proba_readmit, proba_change = chain_predict_proba(model, X_proc)
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        st.stop()

    pred_readmit = int(proba_readmit >= thresholds['readmitted_bin'])
    pred_change  = int(proba_change  >= thresholds['change_bin'])

    # ── Résultats ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Résultats de la prédiction")

    col1, col2 = st.columns(2)

    with col1:
        cls   = "result-danger" if pred_readmit == 1 else "result-safe"
        icon  = "🔴" if pred_readmit == 1 else "🟢"
        lbl   = "RISQUE ÉLEVÉ" if pred_readmit == 1 else "RISQUE FAIBLE"
        color = "#c62828" if pred_readmit == 1 else "#2e7d32"
        st.markdown(f"""
        <div class="result-card {cls}">
            <h2>{icon} Réadmission dans les 30 jours</h2>
            <div class="proba" style="color:{color}">{proba_readmit*100:.1f}%</div>
            <div class="label" style="color:{color}">{lbl}</div>
            <div style="font-size:0.8rem;color:#777;margin-top:8px">
                Seuil : {thresholds['readmitted_bin']:.3f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        cls2   = "result-warning" if pred_change == 1 else "result-safe"
        icon2  = "🟡" if pred_change == 1 else "🟢"
        lbl2   = "CHANGEMENT PROBABLE" if pred_change == 1 else "PAS DE CHANGEMENT"
        color2 = "#e65100" if pred_change == 1 else "#2e7d32"
        st.markdown(f"""
        <div class="result-card {cls2}">
            <h2>{icon2} Changement de traitement</h2>
            <div class="proba" style="color:{color2}">{proba_change*100:.1f}%</div>
            <div class="label" style="color:{color2}">{lbl2}</div>
            <div style="font-size:0.8rem;color:#777;margin-top:8px">
                Seuil : {thresholds['change_bin']:.3f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Barres de probabilité ─────────────────────────────────────
    st.markdown("#### Probabilités détaillées")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Probabilité réadmission", f"{proba_readmit*100:.1f}%",
                  delta="Au-dessus du seuil" if pred_readmit else "En dessous du seuil")
        st.progress(float(proba_readmit))
    with col2:
        st.metric("Probabilité changement traitement", f"{proba_change*100:.1f}%",
                  delta="Au-dessus du seuil" if pred_change else "En dessous du seuil")
        st.progress(float(proba_change))

    # ── Récapitulatif ─────────────────────────────────────────────
    with st.expander("📋 Récapitulatif des données saisies"):
        recap = pd.DataFrame({
            'Variable': [
                'Âge', 'Genre', 'Race', 'Durée hospitalisation',
                'Analyses labo', 'Médicaments', 'Diagnostics',
                'Visites totales antérieures', 'HbA1c', 'Glycémie max',
                'Diagnostic principal', 'Type admission', 'Type sortie'
            ],
            'Valeur': [
                age, gender, race, f"{time_in_hospital} jours",
                num_lab_procedures, num_medications, number_diagnoses,
                total_visits, A1Cresult, max_glu_serum,
                diag_1, admission_type_id, discharge_disposition_id
            ]
        })
        st.dataframe(recap, use_container_width=True, hide_index=True)

    # ── Note clinique ─────────────────────────────────────────────
    st.markdown("""
    <div class="info-box">
    ⚠️ <strong>Note clinique :</strong> Ce modèle est un outil d'aide à la décision basé sur des données
    historiques. Il ne remplace pas le jugement médical. Les prédictions doivent être interprétées
    conjointement avec l'évaluation clinique du patient.
    </div>
    """, unsafe_allow_html=True)