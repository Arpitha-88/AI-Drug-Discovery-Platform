
import streamlit as st

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

import py3Dmol
import pubchempy as pcp

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import random
import plotly.graph_objects as go

from fpdf import FPDF

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="AI Drug Discovery",
    page_icon="🧬",
    layout="wide"
)

# ------------------------------------------------
# MODERN CSS
# ------------------------------------------------

st.markdown("""
<style>

.main {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b
    );
    color: white;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* TITLE */

.title {

    text-align: center;

    font-size: 55px;

    font-weight: bold;

    background: linear-gradient(
        90deg,
        #3b82f6,
        #8b5cf6
    );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;

    margin-bottom: 10px;
}

.subtitle {

    text-align: center;

    color: #cbd5e1;

    font-size: 22px;

    margin-bottom: 40px;
}

/* BUTTON */

.stButton>button {

    background: linear-gradient(
        90deg,
        #3b82f6,
        #8b5cf6
    );

    color: white;

    border: none;

    padding: 15px;

    border-radius: 12px;

    width: 100%;

    font-size: 18px;

    font-weight: bold;
}

/* INPUT */

.stTextInput>div>div>input {

    background-color: #1e293b;

    color: white;

    border-radius: 12px;

    padding: 14px;

    border: 1px solid rgba(255,255,255,0.1);
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# DATASET
# ------------------------------------------------

active_smiles = [
    "CCO","CCC","CCBr",
    "CC(C)O","CCCC","CCCO"
]

inactive_smiles = [
    "CCN","CCCl","CC(C)N",
    "CC(C)Cl","CCCCN","CCCN"
]

toxic_smiles = [
    "CCCl",
    "CCBr",
    "CCCCCl",
    "CCCCBr"
]

smiles_list = []
activity_list = []
toxicity_list = []

# Active molecules
for i in range(250):

    mol = random.choice(active_smiles)

    smiles_list.append(mol)

    activity_list.append(1)

    if mol in toxic_smiles:
        toxicity_list.append(1)
    else:
        toxicity_list.append(0)

# Inactive molecules
for i in range(250):

    mol = random.choice(inactive_smiles)

    smiles_list.append(mol)

    activity_list.append(0)

    if mol in toxic_smiles:
        toxicity_list.append(1)
    else:
        toxicity_list.append(0)

df = pd.DataFrame({
    "smiles": smiles_list,
    "activity": activity_list,
    "toxicity": toxicity_list
})

# ------------------------------------------------
# FEATURE EXTRACTION
# ------------------------------------------------

def extract_features(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return [0,0,0,0]

    return [
        Descriptors.MolWt(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol)
    ]

X = df["smiles"].apply(
    extract_features
).tolist()

y_activity = df["activity"]

y_toxicity = df["toxicity"]

# ------------------------------------------------
# TRAIN MODELS
# ------------------------------------------------

activity_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

toxicity_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

activity_model.fit(X, y_activity)

toxicity_model.fit(X, y_toxicity)

# ------------------------------------------------
# 3D MOLECULE VIEWER
# ------------------------------------------------

def show_molecule(smiles):

    mol = Chem.MolFromSmiles(smiles)

    mol = Chem.AddHs(mol)

    AllChem.EmbedMolecule(mol)

    AllChem.MMFFOptimizeMolecule(mol)

    mol_block = Chem.MolToMolBlock(mol)

    viewer = py3Dmol.view(
        width=700,
        height=500
    )

    viewer.addModel(
        mol_block,
        "mol"
    )

    viewer.setStyle({
        "stick": {},
        "sphere": {
            "scale": 0.25
        }
    })

    viewer.setBackgroundColor("black")

    viewer.zoomTo()

    return viewer._make_html()

# ------------------------------------------------
# DRUG LIKENESS
# ------------------------------------------------

def lipinski(smiles):

    mol = Chem.MolFromSmiles(smiles)

    mw = Descriptors.MolWt(mol)

    h_donors = Descriptors.NumHDonors(mol)

    h_acceptors = Descriptors.NumHAcceptors(mol)

    logp = Descriptors.MolLogP(mol)

    violations = 0

    if mw > 500:
        violations += 1

    if h_donors > 5:
        violations += 1

    if h_acceptors > 10:
        violations += 1

    if logp > 5:
        violations += 1

    if violations <= 1:
        result = "GOOD"
    else:
        result = "POOR"

    return result, violations

# ------------------------------------------------
# PUBCHEM DATA
# ------------------------------------------------

def fetch_pubchem_data(smiles):

    try:

        compounds = pcp.get_compounds(
            smiles,
            namespace='smiles'
        )

        if compounds:

            compound = compounds[0]

            return {
                "Name":
                    compound.iupac_name,

                "Formula":
                    compound.molecular_formula,

                "Weight":
                    compound.molecular_weight
            }

    except:
        return None

# ------------------------------------------------
# PDF REPORT
# ------------------------------------------------

def create_pdf(
    smiles,
    prediction,
    confidence,
    toxicity,
    drug_like,
    info
):

    pdf = FPDF()

    pdf.add_page()

    # TITLE
    pdf.set_font(
        "Arial",
        "B",
        20
    )

    pdf.cell(
        200,
        10,
        txt="AI Drug Discovery Report",
        ln=True,
        align='C'
    )

    pdf.ln(10)

    # CONTENT
    pdf.set_font(
        "Arial",
        "",
        14
    )

    clean_toxicity = toxicity.replace(
        "🔴",
        ""
    ).replace(
        "🟢",
        ""
    )

    pdf.cell(
        200,
        10,
        txt=f"Molecule: {smiles}",
        ln=True
    )

    pdf.cell(
        200,
        10,
        txt=f"Prediction: {prediction}",
        ln=True
    )

    pdf.cell(
        200,
        10,
        txt=f"Confidence: {confidence:.2f}",
        ln=True
    )

    pdf.cell(
        200,
        10,
        txt=f"Toxicity: {clean_toxicity}",
        ln=True
    )

    pdf.cell(
        200,
        10,
        txt=f"Drug-Likeness: {drug_like}",
        ln=True
    )

    pdf.ln(10)

    # MOLECULAR INFO
    pdf.set_font(
        "Arial",
        "B",
        16
    )

    pdf.cell(
        200,
        10,
        txt="Molecular Properties",
        ln=True
    )

    pdf.set_font(
        "Arial",
        "",
        12
    )

    for key, value in info.items():

        pdf.cell(
            200,
            10,
            txt=f"{key}: {value}",
            ln=True
        )

    pdf.output("drug_report.pdf")

# ------------------------------------------------
# HEADER
# ------------------------------------------------

st.markdown(
    '<div class="title">AI Drug Discovery Platform</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Research-Level Molecular Intelligence Dashboard</div>',
    unsafe_allow_html=True
)

# ------------------------------------------------
# INPUT
# ------------------------------------------------

smiles = st.text_input(
    "Enter SMILES Molecule"
)

predict = st.button(
    "Predict Molecule"
)

# ------------------------------------------------
# PREDICTION
# ------------------------------------------------

if predict:

    features = extract_features(smiles)

    prediction = activity_model.predict(
        [features]
    )[0]

    probability = activity_model.predict_proba(
        [features]
    )[0][1]

    toxicity = toxicity_model.predict(
        [features]
    )[0]

    drug_like, violations = lipinski(smiles)

    if prediction == 1:
        prediction_label = "ACTIVE"
    else:
        prediction_label = "INACTIVE"

    if toxicity == 1:
        toxicity_label = "HIGH"
    else:
        toxicity_label = "LOW"

    # ------------------------------------------------
    # METRICS
    # ------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Prediction",
            prediction_label
        )

    with col2:
        st.metric(
            "Confidence",
            f"{probability:.2f}"
        )

    with col3:
        st.metric(
            "Toxicity",
            toxicity_label
        )

    with col4:
        st.metric(
            "Drug-Likeness",
            drug_like
        )

    # ------------------------------------------------
    # MOLECULAR INFO
    # ------------------------------------------------

    st.markdown("## Molecular Information")

    mol = Chem.MolFromSmiles(smiles)

    info = {
        "Formula":
            CalcMolFormula(mol),

        "Molecular Weight":
            round(
                Descriptors.MolWt(mol),
                2
            ),

        "H-Bond Donors":
            Descriptors.NumHDonors(mol),

        "H-Bond Acceptors":
            Descriptors.NumHAcceptors(mol),

        "TPSA":
            round(
                Descriptors.TPSA(mol),
                2
            ),

        "Lipinski Violations":
            violations
    }

    info_df = pd.DataFrame({
        "Property":
            list(info.keys()),

        "Value":
            list(info.values())
    })

    st.table(info_df)

    # ------------------------------------------------
    # PDF DOWNLOAD
    # ------------------------------------------------

    create_pdf(
        smiles,
        prediction_label,
        probability,
        toxicity_label,
        drug_like,
        info
    )

    with open(
        "drug_report.pdf",
        "rb"
    ) as file:

        st.download_button(
            label="Download Research Report",

            data=file,

            file_name="drug_report.pdf",

            mime="application/pdf"
        )

    # ------------------------------------------------
    # PUBCHEM INFO
    # ------------------------------------------------

    st.markdown("## Real Drug Information")

    pubchem_data = fetch_pubchem_data(smiles)

    if pubchem_data:

        st.write(
            f"### Compound Name: {pubchem_data['Name']}"
        )

        st.write(
            f"Formula: {pubchem_data['Formula']}"
        )

        st.write(
            f"Molecular Weight: {pubchem_data['Weight']}"
        )

    else:

        st.warning(
            "No PubChem data found"
        )

    # ------------------------------------------------
    # CHART
    # ------------------------------------------------

    st.markdown("## Confidence Analytics")

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",

        value = probability * 100,

        title = {
            'text': "Confidence Score"
        },

        gauge = {
            'axis': {
                'range': [0,100]
            }
        }
    ))

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ------------------------------------------------
    # 3D STRUCTURE
    # ------------------------------------------------

    st.markdown("## 3D Molecular Structure")

    st.components.v1.html(
        show_molecule(smiles),
        height=550
    )
