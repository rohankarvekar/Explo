import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import joblib
import os

pio.templates["clean"] = go.layout.Template(
    layout=go.Layout(
        font=dict(color="#222222", family="Arial", size=13),
        title=dict(font=dict(color="#1F3864", size=16)),
        paper_bgcolor="white",
        plot_bgcolor="#f8f9fa",
        xaxis=dict(color="#222222", tickfont=dict(color="#222222"),
                   title_font=dict(color="#222222"), gridcolor="#e0e0e0"),
        yaxis=dict(color="#222222", tickfont=dict(color="#222222"),
                   title_font=dict(color="#222222"), gridcolor="#e0e0e0"),
        legend=dict(font=dict(color="#222222"), bgcolor="rgba(255,255,255,0.9)"),
    )
)
pio.templates.default = "clean"

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MelanoVax AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1F3864, #2E75B6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1F3864, #2E75B6);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.85;
        margin-top: 0.3rem;
    }
    .pipeline-step {
        background: #f8f9fa;
        border-left: 4px solid #2E75B6;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .result-high {
        background: #d4edda;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #28a745;
    }
    .result-low {
        background: #f8d7da;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #dc3545;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 600;
    }
    .badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# AMINO ACID PROPERTIES
# ─────────────────────────────────────────────
amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
kd_scale = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2, 'X': 0.0
}

def extract_position_features(sequence):
    sequence = str(sequence).upper()
    length = len(sequence)
    aa_comp = [sequence.count(aa) / length for aa in amino_acids]
    std_seq = sequence.ljust(11, 'X')[:11]
    position_features = []
    for pos in range(11):
        aa_at_pos = std_seq[pos]
        one_hot = [1 if aa_at_pos == aa else 0 for aa in amino_acids]
        position_features.extend(one_hot)
    pos_hydro = [kd_scale.get(std_seq[pos], 0.0) for pos in range(11)]
    p2_aa = std_seq[1]
    p9_aa = std_seq[8]
    p2_onehot = [1 if p2_aa == aa else 0 for aa in amino_acids]
    p9_onehot = [1 if p9_aa == aa else 0 for aa in amino_acids]
    hydrophobic = sum(kd_scale.get(aa, 0) for aa in sequence) / length
    positive = sum(1 for aa in sequence if aa in 'RHK') / length
    negative = sum(1 for aa in sequence if aa in 'DE') / length
    polar = sum(1 for aa in sequence if aa in 'NQSTY') / length
    return (aa_comp + position_features + pos_hydro +
            p2_onehot + p9_onehot + [hydrophobic, positive, negative, polar, length])

# ─────────────────────────────────────────────
# MOCK DATA (replace with real models/data)
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# REAL MODELS + DATA
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    alleles = {
        'HLA-A*02:01': 'model_HLA-A_02_01 (2).pkl',
        'HLA-A*24:02': 'model_HLA-A_24_02.pkl',
        'HLA-B*07:02': 'model_HLA-B_07_02.pkl',
        'HLA-B*57:01': 'model_HLA-B_57_01.pkl',
    }
    for allele, fname in alleles.items():
        models[allele] = joblib.load(fname)
    return models

@st.cache_data
def load_results():
    return pd.read_csv("melanoma_predictions_final.csv")

allele_models = load_models()
df = load_results()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 MelanoVax AI")
    st.markdown("*Computational Melanoma Epitope Predictor*")
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("", [
        "🏠 Home",
        "⚙️ ML Pipeline",
        "📊 Results Dashboard",
        "🔬 Predict Epitope",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **IIT (BHU) Varanasi**  
    Dept. of Pharmaceutical Engineering  
    B.Tech Project 2024-25  
    
    **Models:**  
    4 HLA-specific Random Forest  
    classifiers (AUC: 0.93-0.99)  
    
    **Data:** IEDB (~100k records)
    """)
    st.markdown("---")
    st.markdown("**Made by Rohan**")

# ─────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown('<div class="main-header">🧬 MelanoVax AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Based Identification of Melanoma T-Cell Epitopes<br>A Computational Immunoinformatics Approach to Cancer Vaccine Discovery</div>', unsafe_allow_html=True)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">27,366</div>
            <div class="metric-label">Peptides Analysed</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">199</div>
            <div class="metric-label">Promiscuous Epitopes</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">0.999</div>
            <div class="metric-label">Best Model AUC</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">80%+</div>
            <div class="metric-label">Population Coverage</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Problem & Solution
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 The Problem")
        st.markdown("""
        Melanoma cells express tumor-associated antigens (TAAs). 
        These proteins produce peptide fragments that — if they bind 
        MHC Class I molecules — can activate CD8+ T cells to destroy cancer.

        **The bottleneck:**
        - 27,366 peptide candidates to screen
        - Experimental testing: **225 years** of lab work
        - Cost: Crores of rupees
        """)
    with col2:
        st.markdown("### 💡 Our Solution")
        st.markdown("""
        A **machine learning pipeline** trained on 100,000 experimentally 
        validated IEDB records predicts MHC binding in **seconds**.

        **Key innovations:**
        - 296 position-specific features (vs 25 frequency-based)
        - 4 allele-specific models covering 80%+ world population
        - 3-layer biological validation
        - Promiscuous epitope scoring for broad coverage
        """)

    st.markdown("---")
    st.markdown("### 🔬 Biological Pathway")
    fig = go.Figure()
    steps = [
        ("Melanoma\nProteins", 0.5, 1.0, "#1F3864"),
        ("Proteasomal\nDegradation", 0.5, 0.75, "#2E75B6"),
        ("Peptide\nFragments\n(8-11 aa)", 0.5, 0.50, "#5BA3E0"),
        ("MHC Class I\nBinding", 0.5, 0.25, "#28a745"),
        ("CD8+ T Cell\nActivation", 0.5, 0.0, "#dc3545"),
    ]
    for label, x, y, color in steps:
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers+text',
            marker=dict(size=60, color=color, symbol='circle'),
            text=[label], textposition='middle center',
            textfont=dict(color='white', size=10, family='Arial'),
            showlegend=False
        ))
    for i in range(len(steps)-1):
        fig.add_annotation(
            x=steps[i+1][1], y=steps[i+1][2]+0.07,
            ax=steps[i][1], ay=steps[i][2]-0.07,
            xref='x', yref='y', axref='x', ayref='y',
            arrowhead=2, arrowsize=1.5, arrowcolor='#666'
        )
    fig.add_annotation(
        x=0.75, y=0.5, text="Our ML\nPipeline\nPredicts This ⬅",
        showarrow=True, arrowhead=2, ax=0.6, ay=0.5,
        font=dict(color='#e74c3c', size=12, family='Arial'),
        bgcolor='#fff3cd', bordercolor='#e74c3c', borderwidth=1
    )
    fig.update_layout(
        height=500, showlegend=False,
        xaxis=dict(visible=False, range=[0,1]),
        yaxis=dict(visible=False, range=[-0.15,1.15]),
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: ML PIPELINE
# ─────────────────────────────────────────────
elif page == "⚙️ ML Pipeline":
    st.markdown("## ⚙️ ML Pipeline")
    st.markdown("*How the model was built — step by step*")
    st.markdown("---")

    # Ablation Study
    st.markdown("### 📈 Ablation Study — The Key Finding")
    st.info("💡 **Core insight:** Data quality > Data quantity. Allele-specific training is the primary driver of performance.")

    ablation_data = {
        'Model': ['RF v1\n(50k mixed, 25 feat)', 'RF v2\n(50k mixed, 296 feat)',
                  'RF v3\n(HLA-specific, 296 feat)', 'XGBoost\n(HLA-specific, 296 feat)'],
        'AUC': [0.8723, 0.8741, 0.9849, 0.9710],
        'Accuracy': [0.7982, 0.8019, 0.9455, 0.9207],
        'Change': ['Baseline', '+0.002 (features)', '+0.11 (allele-specific!)', 'Comparison'],
        'Color': ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
    }
    abl_df = pd.DataFrame(ablation_data)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=abl_df['Model'], y=abl_df['AUC'],
        marker_color=abl_df['Color'],
        text=[f'{v:.4f}' for v in abl_df['AUC']],
        textposition='outside', name='AUC'
    ))
    fig.add_hline(y=0.9, line_dash='dash', line_color='red',
                  annotation_text='0.90 threshold', annotation_position='right')
    fig.update_layout(
        height=400, yaxis=dict(range=[0.75, 1.02], title='AUC Score'),
        title='Model Evolution — Ablation Study',
        plot_bgcolor='white', paper_bgcolor='white'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Feature Engineering
    st.markdown("### 🔧 Feature Engineering — 296 Features")
    feat_data = {
        'Feature Group': ['AA Frequency', 'Position One-Hot', 'Position Hydrophobicity',
                          'Anchor P2', 'Anchor P9', 'Global Properties'],
        'Count': [20, 220, 11, 20, 20, 5],
        'Description': [
            'Proportion of each amino acid',
            '11 positions × 20 amino acids',
            'Kyte-Doolittle per position',
            'Critical anchor — Pocket B',
            'Critical anchor — Pocket F',
            'Hydrophobicity, charge, length'
        ]
    }
    feat_df = pd.DataFrame(feat_data)
    fig = px.bar(feat_df, x='Count', y='Feature Group', orientation='h',
                 color='Count', color_continuous_scale='Blues',
                 text='Count', hover_data=['Description'])
    fig.update_layout(height=350, plot_bgcolor='white', paper_bgcolor='white',
                      showlegend=False, title='296 Features per Peptide')
    st.plotly_chart(fig, use_container_width=True)

    # Multi-Allele Performance
    st.markdown("### 🎯 Multi-Allele Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        allele_data = {
            'Allele': ['HLA-A*02:01', 'HLA-A*24:02', 'HLA-B*07:02', 'HLA-B*57:01'],
            'AUC': [0.9393, 0.9596, 0.9772, 0.9985],
            'Accuracy': [0.8733, 0.9135, 0.9249, 0.9728],
            'Population': [50, 20, 15, 10]
        }
        allele_df = pd.DataFrame(allele_data)
        fig = px.bar(allele_df, x='Allele', y='AUC',
                     color='AUC', color_continuous_scale='Greens',
                     text=[f'{v:.4f}' for v in allele_df['AUC']])
        fig.add_hline(y=0.9, line_dash='dash', line_color='red')
        fig.update_layout(height=350, yaxis=dict(range=[0.85,1.02]),
                          plot_bgcolor='white', paper_bgcolor='white',
                          title='AUC per HLA Allele')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(allele_df, values='Population', names='Allele',
                     color_discrete_sequence=['#3498db','#2ecc71','#e74c3c','#f39c12'],
                     title='World Population Coverage (4 Alleles = 80%+)')
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Biological Validation
    st.markdown("### ✅ Three-Layer Biological Validation")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("""**Layer 1 — Statistical**
        
AUC: 0.9393 to 0.9985  
All models above 0.90 threshold  
10,000 held-out test peptides""")
    with col2:
        st.success("""**Layer 2 — Known Epitope**
        
AAGIGILTV (MLANA)  
Published HLA-A\\*02:01 epitope  
Predicted with **98.7% confidence** ✅""")
    with col3:
        st.success("""**Layer 3 — Anchor Residues**
        
P2 = Leucine: **62.96%**  
P9 = Valine: **55.56%**  
Matches crystallographic studies ✅""")

# ─────────────────────────────────────────────
# PAGE: RESULTS DASHBOARD
# ─────────────────────────────────────────────
elif page == "📊 Results Dashboard":
    st.markdown("## 📊 Results Dashboard")
    st.markdown("*Interactive exploration of 27,366 melanoma peptide predictions*")
    st.markdown("---")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_genes = st.multiselect("Filter by Gene",
            options=sorted(df['Gene_name'].unique()),
            default=sorted(df['Gene_name'].unique()))
    with col2:
        min_alleles = st.slider("Minimum alleles bound", 0, 4, 0)
    with col3:
        min_prob = st.slider("Minimum promiscuous score", 0.0, 1.0, 0.0, 0.05)

    filtered = df[
        (df['Gene_name'].isin(selected_genes)) &
        (df['alleles_bound'] >= min_alleles) &
        (df['promiscuous_score'] >= min_prob)
    ]

    # Stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Peptides", f"{len(filtered):,}")
    col2.metric("Predicted Binders", f"{filtered['binding_prediction'].sum():,}")
    col3.metric("3+ Allele Binders", f"{(filtered['alleles_bound']>=3).sum():,}")
    col4.metric("Avg Promiscuous Score", f"{filtered['promiscuous_score'].mean():.3f}")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🧬 Gene Analysis", "🏆 Top Candidates", "📉 Distributions"])

    with tab1:
        gene_summary = filtered.groupby('Gene_name').agg(
            Total=('epitope_sequence','count'),
            Binders=('binding_prediction','sum'),
            Promiscuous_3plus=('alleles_bound', lambda x: (x>=3).sum()),
            Avg_Score=('promiscuous_score','mean')
        ).reset_index()
        gene_summary['Binding_Rate'] = (gene_summary['Binders']/gene_summary['Total']*100).round(1)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(gene_summary.sort_values('Promiscuous_3plus', ascending=True),
                         x='Promiscuous_3plus', y='Gene_name', orientation='h',
                         color='Promiscuous_3plus', color_continuous_scale='Viridis',
                         title='Promiscuous Epitopes (3+ alleles) per Gene',
                         text='Promiscuous_3plus')
            fig.update_layout(height=450, plot_bgcolor='white', paper_bgcolor='white',
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(gene_summary,
                             x='Avg_Score', y='Binding_Rate',
                             size='Total', color='Gene_name',
                             hover_name='Gene_name',
                             title='Gene Analysis — Score vs Binding Rate',
                             labels={'Avg_Score':'Avg Promiscuous Score',
                                     'Binding_Rate':'Binding Rate (%)'})
            fig.update_layout(height=450, plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        prob_cols = ['prob_HLA_A_02_01','prob_HLA_A_24_02','prob_HLA_B_07_02','prob_HLA_B_57_01']
        top_cands = filtered[filtered['alleles_bound']>=3]\
            .sort_values('promiscuous_score', ascending=False).head(30)

        if len(top_cands) > 0:
            # Heatmap
            heatmap_data = top_cands[prob_cols].values
            labels = [f"{row['epitope_sequence']} ({row['Gene_name']})"
                      for _, row in top_cands.iterrows()]

            fig = go.Figure(go.Heatmap(
                z=heatmap_data,
                x=['HLA-A*02:01','HLA-A*24:02','HLA-B*07:02','HLA-B*57:01'],
                y=labels,
                colorscale='RdYlGn', zmin=0, zmax=1,
                text=np.round(heatmap_data,2),
                texttemplate='%{text}',
                textfont=dict(size=9)
            ))
            fig.update_layout(
                title='Top Candidates — Per Allele Binding Probability',
                height=600,
                yaxis=dict(tickfont=dict(size=9))
            )
            st.plotly_chart(fig, use_container_width=True)

            # Table
            display_cols = ['Gene_name','epitope_sequence','peptide_length',
                           'promiscuous_score','alleles_bound'] + prob_cols
            display_df = top_cands[display_cols].reset_index(drop=True)
            for col in prob_cols + ['promiscuous_score']:
                display_df[col] = display_df[col].round(3)
            st.dataframe(display_df, use_container_width=True, height=300)
            
        else:
            st.warning("No 3+ allele binders found with current filters.")

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(filtered, x='promiscuous_score', color='Gene_name',
                               nbins=40, opacity=0.7,
                               title='Promiscuous Score Distribution per Gene')
            fig.add_vline(x=0.5, line_dash='dash', line_color='red',
                          annotation_text='Threshold (0.5)')
            fig.update_layout(height=400, plot_bgcolor='white', paper_bgcolor='white',
                              bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            alleles_dist = filtered['alleles_bound'].value_counts().sort_index().reset_index()
            alleles_dist.columns = ['Alleles Bound','Count']
            fig = px.bar(alleles_dist, x='Alleles Bound', y='Count',
                         color='Count', color_continuous_scale='Blues',
                         title='Distribution of Alleles Bound',
                         text='Count')
            fig.update_layout(height=400, plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

        # Violin
        fig = px.violin(filtered, x='Gene_name', y='promiscuous_score',
                        color='Gene_name', box=True, points=False,
                        title='Promiscuous Score Distribution (Violin) per Gene')
        fig.add_hline(y=0.5, line_dash='dash', line_color='red')
        fig.update_layout(height=450, plot_bgcolor='white', paper_bgcolor='white',
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: PREDICT EPITOPE
# ─────────────────────────────────────────────
elif page == "🔬 Predict Epitope":
    st.markdown("## 🔬 Predict Epitope Binding")
    st.markdown("*Enter a peptide sequence to predict MHC Class I binding probability*")
    st.markdown("---")

    col1, col2 = st.columns([2,1])
    with col1:
        sequence = st.text_input("Enter peptide sequence (8-11 amino acids)",
                                  placeholder="e.g. AAGIGILTV",
                                  help="Standard 1-letter amino acid code, uppercase")
        st.caption("Valid amino acids: A C D E F G H I K L M N P Q R S T V W Y")

    with col2:
        st.markdown("**Quick Examples:**")
        examples = {
            "AAGIGILTV (Known MLANA epitope)": "AAGIGILTV",
            "KVAELVHFL (MAGEA3 top candidate)": "KVAELVHFL",
            "FLRNQPLTFAL (PMEL candidate)": "FLRNQPLTFAL",
            "ALLAGLVSLL (TYR epitope)": "ALLAGLVSLL",
        }
        example_choice = st.selectbox("Or choose an example:", list(examples.keys()))
        if st.button("Load Example"):
            sequence = examples[example_choice]
            st.rerun()

    if st.button("🔬 Predict Binding", type="primary") or sequence:
        if sequence:
            sequence = sequence.upper().strip()
            valid_aa = set('ACDEFGHIKLMNPQRSTVWY')

            if len(sequence) < 8 or len(sequence) > 11:
                st.error(f"❌ Sequence length must be 8-11 aa. Your sequence has {len(sequence)} aa.")
            elif not all(c in valid_aa for c in sequence):
                invalid = [c for c in sequence if c not in valid_aa]
                st.error(f"❌ Invalid amino acids: {invalid}")
            else:
                features = [extract_position_features(sequence)]
                probs = []
                for allele in ['HLA-A*02:01','HLA-A*24:02',
                              'HLA-B*07:02','HLA-B*57:01']:
                    prob = allele_models[allele].predict_proba(features)[0][1]
                    probs.append(float(prob))  
                alleles = ['HLA-A*02:01','HLA-A*24:02','HLA-B*07:02','HLA-B*57:01']
                prom_score = np.mean(probs)
                alleles_bound = sum(p > 0.5 for p in probs)

                st.markdown("---")
                st.markdown(f"### Results for `{sequence}`")

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Length", f"{len(sequence)} aa")
                col2.metric("Promiscuous Score", f"{prom_score:.3f}")
                col3.metric("Alleles Bound (>0.5)", f"{alleles_bound}/4")
                col4.metric("Classification",
                    "🟢 Promiscuous" if alleles_bound >= 3 else
                    "🟡 Partial" if alleles_bound >= 1 else "🔴 Non-binder")

                # Per allele gauge charts
                st.markdown("#### Per-Allele Binding Probability")
                cols = st.columns(4)
                pop_coverage = [50, 20, 15, 10]
                for i, (allele, prob, pop) in enumerate(zip(alleles, probs, pop_coverage)):
                    with cols[i]:
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prob * 100,
                            title={'text': f"{allele}<br><span style='font-size:0.8em'>Pop: ~{pop}%</span>"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': '#2ecc71' if prob > 0.5 else '#e74c3c'},
                                'steps': [
                                    {'range': [0,50], 'color': '#f8d7da'},
                                    {'range': [50,100], 'color': '#d4edda'}
                                ],
                                'threshold': {'line': {'color': 'black', 'width': 3},
                                              'thickness': 0.75, 'value': 50}
                            },
                            number={'suffix': '%', 'font': {'size': 24}}
                        ))
                        fig.update_layout(height=220, margin=dict(l=10,r=10,t=50,b=10))
                        st.plotly_chart(fig, use_container_width=True)

                # Biochemical analysis
                st.markdown("#### Biochemical Feature Analysis")
                col1, col2 = st.columns(2)

                with col1:
                    # AA composition
                    aa_counts = {aa: sequence.count(aa) for aa in sequence}
                    aa_df = pd.DataFrame(list(aa_counts.items()), columns=['AA','Count'])
                    fig = px.bar(aa_df, x='AA', y='Count',
                                 color='Count', color_continuous_scale='Blues',
                                 title='Amino Acid Composition')
                    fig.update_layout(height=280, plot_bgcolor='white',
                                      paper_bgcolor='white', showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Position hydrophobicity
                    pos_hydro = [kd_scale.get(aa, 0) for aa in sequence]
                    pos_labels = [f'P{i+1}\n({aa})' for i, aa in enumerate(sequence)]
                    colors = ['#2ecc71' if h > 0 else '#e74c3c' for h in pos_hydro]

                    fig = go.Figure(go.Bar(
                        x=pos_labels, y=pos_hydro,
                        marker_color=colors,
                        text=[f'{h:.1f}' for h in pos_hydro],
                        textposition='outside'
                    ))
                    fig.add_hline(y=0, line_color='black', line_width=1)
                    fig.update_layout(
                        title='Per-Position Hydrophobicity (Kyte-Doolittle)',
                        height=280, plot_bgcolor='white', paper_bgcolor='white',
                        yaxis_title='Hydrophobicity Score'
                    )
                    # Highlight P2 and P9
                    if len(sequence) >= 2:
                        fig.add_vrect(x0=-0.5, x1=0.5, fillcolor='yellow', opacity=0.15,
                                      annotation_text='P2\nAnchor', annotation_position='top')
                    if len(sequence) >= 9:
                        fig.add_vrect(x0=7.5, x1=8.5, fillcolor='orange', opacity=0.15,
                                      annotation_text='P9\nAnchor', annotation_position='top')
                    st.plotly_chart(fig, use_container_width=True)

                # Interpretation
                st.markdown("#### Interpretation")
                if alleles_bound >= 3:
                    st.markdown(f"""<div class="result-high">
                    ✅ <strong>HIGH PRIORITY VACCINE CANDIDATE</strong><br>
                    This peptide binds {alleles_bound}/4 HLA alleles with probability > 0.5.<br>
                    Promiscuous score: <strong>{prom_score:.3f}</strong><br>
                    Recommended for: <em>In vitro MHC binding assay validation</em>
                    </div>""", unsafe_allow_html=True)
                elif alleles_bound >= 1:
                    st.markdown(f"""<div style='background:#fff3cd;border-radius:8px;
                    padding:1rem;border-left:4px solid #ffc107'>
                    🟡 <strong>PARTIAL BINDER</strong><br>
                    Binds {alleles_bound}/4 alleles. Limited population coverage.<br>
                    Consider for allele-specific vaccine formulations only.
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="result-low">
                    ❌ <strong>NON-BINDER</strong><br>
                    This peptide is predicted not to bind any of the 4 target alleles.<br>
                    Promiscuous score: <strong>{prom_score:.3f}</strong>
                    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888; font-size:0.85rem; padding:1rem'>
    🧬 MelanoVax AI | IIT (BHU) Varanasi | Dept. of Pharmaceutical Engineering & Technology<br>
    B.Tech Project 2024-25 | Data Source: IEDB | Models: Random Forest (AUC: 0.93-0.99)
</div>
""", unsafe_allow_html=True)
