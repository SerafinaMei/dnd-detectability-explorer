import csv
import itertools
from pathlib import Path

import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from upsetplot import plot, from_memberships
import streamlit as st

# page config
st.set_page_config(
    page_title="DnD Gene Detectability Explorer",
    layout="wide",
    page_icon="🧬"
)

st.markdown("""
    <style>
    [data-testid="stDataFrame"] [role="columnheader"] {
        font-weight: 700 !important;
        background-color: #f0f2f6 !important;
    }

    [data-testid="stDataFrame"] [role="rowheader"] {
        font-weight: 700 !important;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------
# File paths - adjust as needed for local testing
# -----------------------------------------------
CARDIAC_CSV = "detection_recheck_outputs/master_df_cardiac_qc3.csv"
NEURON_CSV = "detection_recheck_outputs/master_df_neuron_qc3.csv"
ANNOT_XLSX = "detection_recheck_outputs/genes_other_info.xlsx"
VARIANT_FILE = "detection_recheck_outputs/FINAL_Combined_Master_Variant_Table.xlsx"
SUMMARY_TSV = "detection_recheck_outputs/summary_counts.tsv" # Optional

ANNOT_SHEET = 0

# ------------------------------------------------
# Helper functions
# ------------------------------------------------
QC_LEVELS = ["Raw", "LooseQC", "StrongQC"]

BASE_METRICS = [
    "Detection_Rate_%",
    "Mean_Expr_All",
    "Mean_Expr_Detected",
    "Aggregated_CPM"
]

def qc_col(qc_level: str, metric: str) -> str:
    return f"{qc_level}_{metric}"

def safe_numeric(series, fill_value=0):
    return pd.to_numeric(series, errors="coerce").fillna(fill_value)

def add_dataset_label(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    df = df.copy()
    df["Dataset"] = dataset_name
    return df

def normalize_bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    s = df[col]
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])

def pick_annotation_key(df_metrics: pd.DataFrame, df_annot: pd.DataFrame):
    candidates = ["Gene_Symbol", "hgnc_symbol", "gene_symbol", "Symbol"]
    for c in candidates:
        if c in df_metrics.columns and c in df_annot.columns:
            return c, c
    if "Gene_Symbol" in df_metrics.columns and "hgnc_symbol" in df_annot.columns:
        return "Gene_Symbol", "hgnc_symbol"
    raise ValueError("Could not find a matching gene key between metrics files and annotation file.")

def load_optional_summary(summary_path: str) -> pd.DataFrame:
    p = Path(summary_path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p, sep="\t")
    except Exception:
        return pd.DataFrame()

def build_variant_gene_summary(variant_df: pd.DataFrame) -> pd.DataFrame:
    if variant_df.empty:
        return pd.DataFrame(columns=[
            "Gene_Symbol",
            "Unique_Variant_Sites",
            "Min_Population_Variant_Frequency",
            "Max_Population_Variant_Frequency",
            "Mean_Population_Variant_Frequency"
        ])

    tmp = variant_df.copy()
    tmp["Gene_Symbol"] = tmp["Gene"].astype(str)
    tmp["Variant_Site_Key"] = (
        tmp["Chromosome"].astype(str) + ":" +
        tmp["Position"].astype(str) + ":" +
        tmp["Ref_Allele"].astype(str) + ">" +
        tmp["Alt_Allele"].astype(str)
    )

    summary = (
        tmp.groupby("Gene_Symbol", dropna=False)
        .agg(
            Unique_Variant_Sites=("Variant_Site_Key", "nunique"),
            Min_Population_Variant_Frequency=("Population_Variant_Frequency", "min"),
            Max_Population_Variant_Frequency=("Population_Variant_Frequency", "max"),
            Mean_Population_Variant_Frequency=("Population_Variant_Frequency", "mean"),
        )
        .reset_index()
    )
    return summary

def build_pairwise_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    tmp = df.copy()
    tmp["Gene_Symbol"] = tmp["Gene"].astype(str)
    tmp["Variant_Site_Key"] = (
        tmp["Chromosome"].astype(str) + ":" +
        tmp["Position"].astype(str) + ":" +
        tmp["Ref_Allele"].astype(str) + ">" +
        tmp["Alt_Allele"].astype(str)
    )

    site_mapping = tmp.groupby(["Gene_Symbol", "Variant_Site_Key"]).apply(
        lambda x: list(set(zip(x["Cell_Line"], x["Editing_Strategy"])))
    ).reset_index(name="Line_Strats")

    pairwise_records = []
    for _, row in site_mapping.iterrows():
        gene = row["Gene_Symbol"]
        line_strats = row["Line_Strats"]
        if len(line_strats) < 2:
            continue

        for (l1, s1), (l2, s2) in itertools.combinations(line_strats, 2):
            if l1 == l2:
                continue 
            
            if l1 > l2:
                (l1, s1), (l2, s2) = (l2, s2), (l1, s1)
            
            pairwise_records.append({
                "Gene_Symbol": gene,
                "Cell_Line_A": l1,
                "Strategy_A": s1,
                "Cell_Line_B": l2,
                "Strategy_B": s2
            })
    
    if not pairwise_records:
        return pd.DataFrame()

    pair_df = pd.DataFrame(pairwise_records)
    summary = pair_df.groupby(
        ["Gene_Symbol", "Cell_Line_A", "Strategy_A", "Cell_Line_B", "Strategy_B"]
    ).size().reset_index(name="Shared_Targetable_Positions")
    
    return summary.sort_values(["Shared_Targetable_Positions", "Gene_Symbol"], ascending=[False, True])

def coerce_variant_df_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Position", "Population_Variant_Frequency"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["Cell_Line", "Editing_Strategy", "Gene", "Chromosome", "Ref_Allele", "Alt_Allele"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df

CELL_LINE_LABELS = {
    "KOLF2-ARID2-A02": "KOLF2",
    "WTB_variants_PASS": "WTB",
    "WTC_variants_PASS": "WTC",
    "cN8-hNIL": "WTD (cN8-hNIL)",
    "p47": "p47"
}

def prettify_cell_line(x: str) -> str:
    return CELL_LINE_LABELS.get(x, x)


# def render_bold_dataframe(df_in: pd.DataFrame, use_container_width: bool = True):
#     """
#     Keep the dataframe index as the real Streamlit index so it stays pinned
#     while still bolding column headers. Also attempts to bold index values.
#     """
#     if df_in is None:
#         st.dataframe(df_in, use_container_width=use_container_width)
#         return

#     if df_in.empty:
#         st.dataframe(df_in, use_container_width=use_container_width)
#         return

#     styler = (
#         df_in.style
#         .set_table_styles([
#             {"selector": "th", "props": [("font-weight", "700"), ("background-color", "#f0f2f6")]},
#             {"selector": ".row_heading", "props": [("font-weight", "700")]},
#             {"selector": ".index_name", "props": [("font-weight", "700")]},
#         ])
#     )

#     st.dataframe(styler, use_container_width=use_container_width)
def render_bold_dataframe(df_in: pd.DataFrame, use_container_width: bool = True):
    """
    Keep the real dataframe index pinned, while bolding
    column headers and index values as much as Streamlit allows.
    """
    if df_in is None or df_in.empty:
        st.dataframe(df_in, use_container_width=use_container_width)
        return

    styler = (
        df_in.style
        # bold all body cells if desired? leave off for now
        .set_table_styles([
            {"selector": "th", "props": [("font-weight", "700"), ("background-color", "#f0f2f6")]},
        ])
        .map_index(lambda _: "font-weight: 700;", axis=0)   # row index / gene names
        .map_index(lambda _: "font-weight: 700;", axis=1)   # column headers
    )

    st.dataframe(styler, use_container_width=use_container_width)

    
# ------------------------------------------------
# Data Loader
# ------------------------------------------------
@st.cache_data
def load_data():
    cardiac = pd.read_csv(CARDIAC_CSV)
    neuron = pd.read_csv(NEURON_CSV)
    annot = pd.read_excel(ANNOT_XLSX, sheet_name=ANNOT_SHEET)

    cardiac = add_dataset_label(cardiac, "Cardiac")
    neuron = add_dataset_label(neuron, "Neuron")
    metrics_df = pd.concat([cardiac, neuron], ignore_index=True)

    left_key, right_key = pick_annotation_key(metrics_df, annot)
    merged = metrics_df.merge(annot, how="left", left_on=left_key, right_on=right_key)

    for col in ["dominant_mutation_count", "Citation_Count", "s_het"]:
        if col in merged.columns:
            merged[col] = safe_numeric(merged[col], fill_value=0)

    for col in ["HPO_Cardiovascular", "HPO_Nervous", "HPO_Metabolism", "HPO_Musculoskeletal"]:
        if col in merged.columns:
            merged[col] = normalize_bool_col(merged, col)

    variant_df = pd.DataFrame()
    if Path(VARIANT_FILE).exists():
        variant_df = pd.read_excel(VARIANT_FILE, sheet_name="Targetable_&_Het_Var")
        variant_df = coerce_variant_df_types(variant_df)

    summary_df = load_optional_summary(SUMMARY_TSV)
    return merged, variant_df, summary_df

df, variant_master_df, summary_counts_df = load_data()

# ------------------------------------------------
# Dashboard UI - Main Header & Info Manual
# ------------------------------------------------
st.title("DnD Gene Detectability Explorer")

with st.expander("ℹ️ Information Manual & Glossary (Click to expand)"):
    st.markdown("""
    This dashboard filters candidate genes based on expression metrics, annotations, and overlapping sites between targetable variants and heterzygous regions.
    
    **Expression Metrics**:
    * **Detection Rate**: The percentage of cells in the dataset expressing the gene.
    * **Mean Expr Detected**: Average counts per million in average cell considering only the cells where the gene was actually detected.
    * **Aggregated CPM**: Counts per million when aggregate all the cells, representing total expression magnitude considering the cell's overall mRNA counts.
   

    **Literature & Annotations**:
    * **s_het**: (Zeng, Spence, et al. 2024) Higher values indicate less tolerance to heterozygous loss-of-function variants.
    * **Dominant Mutation Count/ Citation Count/ **: Grace kindly provided the Dominant Mutation Count. ClinVar Citation Count sourced from ClinVar(more mutation-related) and Citation Count sourced from PubMed.
    * **HPO Phenotypes**: Associated clinical phenotypes "Abnormality in ___" based on the Human Phenotype Ontology.

    **Variant Features**:
    * **Unique Variant Sites**: Number of unique genomic locations that are heterozygous and targetable in the selected cell line(s) and editing strategies.
    * **Filtered Variant Rows**: Number of rows in the variant table that match the current gene set and variant filters, which may include multiple rows per gene if there are multiple variants.
    * **Pairwise Shared Positions**: Number of targetable and heterozygous positions overlapping between any two cell lines (e.g., KOLF2 & WTB) and specific editing strategies. --- if a variant is in more than two cell lines/strategies, it will contribute to the shared position count for each pairwise combination of those lines. eg, if a variant is in KOLF2, WTB, and WTC, it will count as 1 shared position for KOLF2-WTB, 1 for KOLF2-WTC, and 1 for WTB-WTC.
    * **Population Variant Frequency**: Observed allele frequency in the population database.
    """)

# ------------------------------------------------
# Sidebar Filters
# ------------------------------------------------
st.sidebar.title("Apply Filters")

include_expression = st.sidebar.checkbox(
    "Include Expression Data & Filters",
    value=True,
    help="Uncheck this to explore genes strictly based on Annotations and Variants, ignoring Single-Cell Detectability."
)

datasets = sorted(df["Dataset"].dropna().unique().tolist())
cell_types = sorted(df["Cell_Type"].dropna().unique().tolist())
replicates = sorted(df["Replicate"].dropna().unique().tolist())

if include_expression:
    st.sidebar.subheader(
        "Expression Thresholds",
        help="First define the minimum expression / detectability thresholds, then specify which datasets, cell types, and replicates those thresholds should be evaluated in."
    )

    qc_level = st.sidebar.selectbox("QC Level", QC_LEVELS, index=1)

    det_col = qc_col(qc_level, "Detection_Rate_%")
    mean_all_col = qc_col(qc_level, "Mean_Expr_All")
    mean_detected_col = qc_col(qc_level, "Mean_Expr_Detected")
    agg_col = qc_col(qc_level, "Aggregated_CPM")

    required_cols = [det_col, mean_all_col, mean_detected_col, agg_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.error(f"Missing expected QC-specific columns: {missing_cols}")
        st.stop()

    min_det = st.sidebar.slider(
        f"Min Detection Rate (%) [{qc_level}]",
        0.0, 100.0, 0.0, step=1.0
    )

    max_agg = float(pd.to_numeric(df[agg_col], errors="coerce").fillna(0).max())
    min_cpm = st.sidebar.slider(
        "Min Aggregated CPM",
        0.0,
        max(0.0, max_agg),
        0.0,
        step=max(1.0, max_agg / 100 if max_agg > 100 else 1.0)
    )

    max_mean_detected = float(pd.to_numeric(df[mean_detected_col], errors="coerce").fillna(0).max())
    min_mean_detected = st.sidebar.slider(
        "Min Mean Expr Detected",
        0.0,
        max(0.0, max_mean_detected),
        0.0,
        step=max(0.01, max_mean_detected / 100 if max_mean_detected > 1 else 0.01)
    )

    st.sidebar.subheader(
        "Expression Conditions",
        help="After setting the numeric thresholds above, choose the dataset, cell type, and replicate context in which a gene must satisfy them."
    )

    selected_datasets = st.sidebar.multiselect("Dataset", datasets, default=datasets)
    selected_cells = st.sidebar.multiselect("Cell Type / Stage", cell_types, default=cell_types)

    expr_logic = st.sidebar.radio(
        "Match Logic for Cell Types:",
        ["OR (Passes thresholds in ANY selected cell type)", "AND (Passes thresholds in ALL selected cell types)"]
    )

    selected_replicates = st.sidebar.multiselect("Replicate", replicates, default=replicates)

st.sidebar.subheader(
    "Literature & Genetics",
    help="Filter genes by genetic constraint, dominant mutation evidence, citation counts, and phenotype annotations."
)

if "s_het" in df.columns:
    max_s_het = float(df["s_het"].max())
    min_s_het = st.sidebar.slider("Min s_het Threshold", 0.0, max(1.0, max_s_het), 0.0, step=0.01)
else:
    min_s_het = 0.0

if "dominant_mutation_count" in df.columns:
    max_mut = int(df["dominant_mutation_count"].fillna(0).max())
    min_mut = st.sidebar.number_input("Min Dominant Mutation Count", min_value=0, max_value=max_mut, value=0)
else:
    min_mut = 0

if "Citation_Count" in df.columns:
    max_cite = int(df["Citation_Count"].fillna(0).max())
    min_cite = st.sidebar.number_input("Min Citation Count", min_value=0, max_value=max_cite, value=0)
else:
    min_cite = 0

with st.sidebar.expander("Clinical Phenotypes (HPO)"):
    st.markdown("*Check to only show genes associated with:*")
    req_cardio = st.checkbox("Cardiovascular System")
    req_neuro = st.checkbox("Nervous System")
    req_metab = st.checkbox("Metabolism / Homeostasis")
    req_musculo = st.checkbox("Musculoskeletal System")

if variant_master_df.empty:
    st.sidebar.subheader("Variant Filters")
    st.sidebar.info("Variant table not found at VARIANT_FILE path.")
    selected_variant_cell_lines = []
    selected_variant_strategies = []
    min_pop_freq = 0.0
    var_logic = "OR"
    strat_logic = "OR"
else:
    variant_cell_lines = sorted(variant_master_df["Cell_Line"].dropna().unique().tolist())
    variant_strategies = sorted(variant_master_df["Editing_Strategy"].dropna().unique().tolist())

    st.sidebar.subheader(
        "Variant Thresholds",
        help="First define the variant frequency threshold, then choose which cell lines and editing strategies that threshold should be evaluated in."
    )

    max_popfreq = float(
        pd.to_numeric(variant_master_df["Population_Variant_Frequency"], errors="coerce").dropna().max()
    ) if variant_master_df["Population_Variant_Frequency"].notna().any() else 1.0

    min_pop_freq = st.sidebar.slider(
        "Min Population Variant Frequency",
        0.0,
        max(1.0, max_popfreq),
        0.0,
        step=0.01
    )

    st.sidebar.subheader(
        "Variant Conditions",
        help="After setting the frequency threshold above, choose the cell lines and editing strategies in which variants should satisfy it."
    )

    selected_variant_cell_lines = st.sidebar.multiselect(
        "Variant Cell Line",
        variant_cell_lines,
        default=variant_cell_lines,
        format_func=prettify_cell_line
    )

    var_logic = st.sidebar.radio(
        "Match Logic for Variant Cell Lines:",
        ["OR (Variants in ANY selected line)", "AND (Variants in ALL selected lines)"]
    )

    selected_variant_strategies = st.sidebar.multiselect(
        "Editing Strategy",
        variant_strategies,
        default=variant_strategies
    )

    strat_logic = st.sidebar.radio(
        "Match Logic for Editing Strategies:",
        ["OR (Variants in ANY selected strategy)", "AND (Variants in ALL selected strategies)"]
    )














# st.sidebar.title("Apply Filters")

# include_expression = st.sidebar.checkbox(
#     "Include Expression Data & Filters", 
#     value=True, 
#     help="Uncheck this to explore genes strictly based on Annotations and Variants, ignoring Single-Cell Detectability."
# )

# if include_expression:
#     st.sidebar.subheader("Dataset & Conditions",
#                          help="This section allows you to filter genes by their expression and detectabilty in specific datasets, cell types, and replicates. The Detection Rate and other expression metrics will be calculated based on the selected conditions.")
#     datasets = sorted(df["Dataset"].dropna().unique().tolist())
#     selected_datasets = st.sidebar.multiselect("Dataset", datasets, default=datasets)

#     cell_types = sorted(df["Cell_Type"].dropna().unique().tolist())
#     selected_cells = st.sidebar.multiselect("Cell Type / Stage", cell_types, default=cell_types)
    
#     expr_logic = st.sidebar.radio(
#         "Match Logic for Cell Types:", 
#         ["OR (Passes thresholds in ANY selected cell type)", "AND (Passes thresholds in ALL selected cell types)"]
#     )

#     replicates = sorted(df["Replicate"].dropna().unique().tolist())
#     selected_replicates = st.sidebar.multiselect("Replicate", replicates, default=replicates)

#     qc_level = st.sidebar.selectbox("QC Level", QC_LEVELS, index=1)

#     st.sidebar.subheader("Expression Thresholds",
#                          help="This is the section where you can set thresholds for filtering genes based on their detectability and expression levels.")
#     det_col = qc_col(qc_level, "Detection_Rate_%")
#     mean_all_col = qc_col(qc_level, "Mean_Expr_All")
#     mean_detected_col = qc_col(qc_level, "Mean_Expr_Detected")
#     agg_col = qc_col(qc_level, "Aggregated_CPM")

#     required_cols = [det_col, mean_all_col, mean_detected_col, agg_col]
#     missing_cols = [c for c in required_cols if c not in df.columns]
#     if missing_cols:
#         st.error(f"Missing expected QC-specific columns: {missing_cols}")
#         st.stop()

#     min_det = st.sidebar.slider(f"Min Detection Rate (%) [{qc_level}]", 0.0, 100.0, 0.0, step=1.0)
    
#     max_agg = float(pd.to_numeric(df[agg_col], errors="coerce").fillna(0).max())
#     min_cpm = st.sidebar.slider("Min Aggregated CPM", 0.0, max(0.0, max_agg), 0.0, step=max(1.0, max_agg / 100 if max_agg > 100 else 1.0))

#     max_mean_detected = float(pd.to_numeric(df[mean_detected_col], errors="coerce").fillna(0).max())
#     min_mean_detected = st.sidebar.slider("Min Mean Expr Detected", 0.0, max(0.0, max_mean_detected), 0.0, step=max(0.01, max_mean_detected / 100 if max_mean_detected > 1 else 0.01))

# st.sidebar.subheader("Literature & Genetics",
#                      help="This is the section where you can set thresholds for filtering genes based on their genetic constraint metrics, known dominant mutations, literature citation counts, and associated clinical phenotypes.")

# if "s_het" in df.columns:
#     max_s_het = float(df["s_het"].max())
#     min_s_het = st.sidebar.slider("Min s_het Threshold", 0.0, max(1.0, max_s_het), 0.0, step=0.01)
# else:
#     min_s_het = 0.0

# if "dominant_mutation_count" in df.columns:
#     max_mut = int(df["dominant_mutation_count"].fillna(0).max())
#     min_mut = st.sidebar.number_input("Min Dominant Mutation Count", min_value=0, max_value=max_mut, value=0)
# else:
#     min_mut = 0

# if "Citation_Count" in df.columns:
#     max_cite = int(df["Citation_Count"].fillna(0).max())
#     min_cite = st.sidebar.number_input("Min Citation Count", min_value=0, max_value=max_cite, value=0)
# else:
#     min_cite = 0

# with st.sidebar.expander("Clinical Phenotypes (HPO)"):
#     st.markdown("*Check to only show genes associated with:*")
#     req_cardio = st.checkbox("Cardiovascular System")
#     req_neuro = st.checkbox("Nervous System")
#     req_metab = st.checkbox("Metabolism / Homeostasis")
#     req_musculo = st.checkbox("Musculoskeletal System")

# st.sidebar.subheader("Variant Filters",
#                     help="This is the section where you can filter genes based on the presence of common targetable variants overlapped with heterozygous sites in specific cell lines and editing strategies, as well as set a minimum population variant frequency threshold to exclude common variants.")
# if variant_master_df.empty:
#     st.sidebar.info("Variant table not found at VARIANT_FILE path.")
#     selected_variant_cell_lines = []
#     selected_variant_strategies = []
#     min_pop_freq = 0.0
# else:
#     variant_cell_lines = sorted(variant_master_df["Cell_Line"].dropna().unique().tolist())
#     selected_variant_cell_lines = st.sidebar.multiselect(
#         "Variant Cell Line", variant_cell_lines, default=variant_cell_lines, format_func=prettify_cell_line
#     )
    
#     var_logic = st.sidebar.radio(
#         "Match Logic for Variant Cell Lines:", 
#         ["OR (Variants in ANY selected line)", "AND (Variants in ALL selected lines)"]
#     )

#     variant_strategies = sorted(variant_master_df["Editing_Strategy"].dropna().unique().tolist())
#     selected_variant_strategies = st.sidebar.multiselect(
#         "Editing Strategy", variant_strategies, default=variant_strategies
#     )
    
#     strat_logic = st.sidebar.radio(
#         "Match Logic for Editing Strategies:", 
#         ["OR (Variants in ANY selected strategy)", "AND (Variants in ALL selected strategies)"]
#     )

#     max_popfreq = float(pd.to_numeric(variant_master_df["Population_Variant_Frequency"], errors="coerce").dropna().max()) if variant_master_df["Population_Variant_Frequency"].notna().any() else 1.0
#     min_pop_freq = st.sidebar.slider("Min Population Variant Frequency", 0.0, max(1.0, max_popfreq), 0.0, step=0.01)

# ------------------------------------------------
# Filter Applications
# ------------------------------------------------
annot_cols = ["Gene_Symbol", "dominant_mutation_count", "Citation_Count", "s_het", "HPO_Cardiovascular", "HPO_Nervous", "HPO_Metabolism", "HPO_Musculoskeletal"]
available_annot_cols = [c for c in annot_cols if c in df.columns]
gene_level_df = df[available_annot_cols].drop_duplicates()

if "dominant_mutation_count" in gene_level_df.columns:
    gene_level_df = gene_level_df[gene_level_df["dominant_mutation_count"] >= min_mut]
if "Citation_Count" in gene_level_df.columns:
    gene_level_df = gene_level_df[gene_level_df["Citation_Count"] >= min_cite]
if "s_het" in gene_level_df.columns:
    gene_level_df = gene_level_df[gene_level_df["s_het"] >= min_s_het]
if req_cardio and "HPO_Cardiovascular" in gene_level_df.columns:
    gene_level_df = gene_level_df[gene_level_df["HPO_Cardiovascular"] == True]
if req_neuro and "HPO_Nervous" in gene_level_df.columns:
    gene_level_df = gene_level_df[gene_level_df["HPO_Nervous"] == True]
if req_metab and "HPO_Metabolism" in gene_level_df.columns:
    gene_level_df = gene_level_df[gene_level_df["HPO_Metabolism"] == True]
if req_musculo and "HPO_Musculoskeletal" in gene_level_df.columns:
    gene_level_df = gene_level_df[gene_level_df["HPO_Musculoskeletal"] == True]

filtered_gene_set = set(gene_level_df["Gene_Symbol"].dropna().unique())

# Expression Application
if include_expression:
    base_df = df[
        (df["Dataset"].isin(selected_datasets)) &
        (df["Cell_Type"].isin(selected_cells)) &
        (df["Replicate"].isin(selected_replicates))
    ].copy()

    for c in [det_col, mean_all_col, mean_detected_col, agg_col]:
        base_df[c] = safe_numeric(base_df[c], fill_value=0)

    passed_expr_df = base_df[
        (base_df[det_col] >= min_det) &
        (base_df[agg_col] >= min_cpm) &
        (base_df[mean_detected_col] >= min_mean_detected)
    ].copy()
    
    if expr_logic.startswith("AND") and len(selected_cells) > 0:
        gene_cell_counts = passed_expr_df.groupby("Gene_Symbol")["Cell_Type"].nunique()
        valid_genes = gene_cell_counts[gene_cell_counts == len(selected_cells)].index
        passed_expr_df = passed_expr_df[passed_expr_df["Gene_Symbol"].isin(valid_genes)]
        
    filtered_df = passed_expr_df[passed_expr_df["Gene_Symbol"].isin(filtered_gene_set)].copy()
    filtered_gene_set = set(filtered_df["Gene_Symbol"].dropna().unique())
else:
    filtered_df = pd.DataFrame()

# Variant Application
if variant_master_df.empty:
    filtered_variant_df = pd.DataFrame()
    gene_variant_summary = pd.DataFrame()
    pairwise_variant_summary = pd.DataFrame()
else:
    filtered_variant_df = variant_master_df.copy()

    if selected_variant_cell_lines:
        filtered_variant_df = filtered_variant_df[filtered_variant_df["Cell_Line"].isin(selected_variant_cell_lines)]
    if selected_variant_strategies:
        filtered_variant_df = filtered_variant_df[filtered_variant_df["Editing_Strategy"].isin(selected_variant_strategies)]

    if filtered_gene_set:
        filtered_variant_df = filtered_variant_df[filtered_variant_df["Gene"].astype(str).isin(filtered_gene_set)]
    else:
        filtered_variant_df = filtered_variant_df.iloc[0:0].copy()

    filtered_variant_df = filtered_variant_df[
        filtered_variant_df["Population_Variant_Frequency"].isna() | 
        (filtered_variant_df["Population_Variant_Frequency"] >= min_pop_freq)
    ]
    
    if var_logic.startswith("AND") and len(selected_variant_cell_lines) > 0:
        var_counts = filtered_variant_df.groupby("Gene")["Cell_Line"].nunique()
        valid_var_genes = var_counts[var_counts == len(selected_variant_cell_lines)].index
        filtered_variant_df = filtered_variant_df[filtered_variant_df["Gene"].isin(valid_var_genes)]

    if strat_logic.startswith("AND") and len(selected_variant_strategies) > 0:
        strat_counts = filtered_variant_df.groupby("Gene")["Editing_Strategy"].nunique()
        valid_strat_genes = strat_counts[strat_counts == len(selected_variant_strategies)].index
        filtered_variant_df = filtered_variant_df[filtered_variant_df["Gene"].isin(valid_strat_genes)]

    gene_variant_summary = build_variant_gene_summary(filtered_variant_df)
    pairwise_variant_summary = build_pairwise_summary(filtered_variant_df)

# ------------------------------------------------
# UI Dashboards
# ------------------------------------------------
col1, col2, col3 = st.columns(3)
unique_gene_count = len(filtered_gene_set)
col1.metric("Genes Matching Expression Criteria", unique_gene_count)

if include_expression and not filtered_df.empty:
    col2.metric("Highest Detection Rate", f"{filtered_df[det_col].max():.2f}%")
    # col3.metric("Highest Aggregated CPM", f"{filtered_df[agg_col].max():.2f}")
    col3.metric("Highest Mean Expr Detected", f"{filtered_df[mean_detected_col].max():.2f}")

st.subheader("Variant Summary for Currently Filtered Genes")
v1, v2, v3 = st.columns(3)
if filtered_variant_df.empty:
    v1.metric("Genes Matching ALL Criteria", 0)
    v2.metric("Filtered Variant Rows", 0)
    v3.metric("Unique Variant Sites", 0)
    #v4.metric("Max Population Var Freq", "NA")
else:
    v1.metric("Genes Matching ALL Criteria", filtered_variant_df["Gene"].nunique())
    v2.metric("Filtered Variant Rows", len(filtered_variant_df))
    filtered_variant_df["Variant_Site_Key"] = (
        filtered_variant_df["Chromosome"].astype(str) + ":" +
        filtered_variant_df["Position"].astype(str) + ":" +
        filtered_variant_df["Ref_Allele"].astype(str) + ">" +
        filtered_variant_df["Alt_Allele"].astype(str)
    )
    v3.metric("Unique Variant Sites", filtered_variant_df["Variant_Site_Key"].nunique())
    max_pop_val = pd.to_numeric(filtered_variant_df["Population_Variant_Frequency"], errors="coerce")
    # if max_pop_val.notna().any():
    #     v4.metric("Max Population Var Freq", f"{max_pop_val.max():.3f}")
    # else:
    #     v4.metric("Max Population Var Freq", "NA")

if include_expression:
    st.subheader("Detection Rate Distribution")
    if not base_df.empty:
        fig_hist = px.histogram(
            base_df, x=det_col, color="Dataset", nbins=50, barmode="overlay", opacity=0.65,
            title=f"Distribution within Selected Conditions ({qc_level})",
            labels={det_col: f"{qc_level} Detection Rate (%)"}
        )
        fig_hist.add_vline(x=min_det, line_dash="dash", line_color="red", annotation_text=f"Cutoff: {min_det}%", annotation_position="top right")
        st.plotly_chart(fig_hist, use_container_width=True)

# ------------------------------------------------
# Tables & Visualizations
# ------------------------------------------------
# ------------------------------------------------
# UPSET PLOTS SECTION
# ------------------------------------------------
st.subheader("Position Intersections")

# PLOTLY HEATMAP
st.subheader("Pairwise Shared Targetable Positions")
st.markdown("Shows the exact number of overlapping targetable variant positions between any two specific cell line and strategy combinations.")
if not 'pairwise_variant_summary' in locals() or pairwise_variant_summary.empty:
    st.info("No pairwise overlaps found. Ensure you have at least 2 cell lines selected in your Variant Filters.")
else:
    
    heat_df = pairwise_variant_summary.copy()
    heat_df["Condition_A"] = heat_df["Cell_Line_A"] + " (" + heat_df["Strategy_A"] + ")"
    heat_df["Condition_B"] = heat_df["Cell_Line_B"] + " (" + heat_df["Strategy_B"] + ")"
    matrix = heat_df.pivot_table(index="Condition_A", columns="Condition_B", values="Shared_Targetable_Positions", aggfunc="sum").fillna(0)
    fig_heat = px.imshow(
        matrix, text_auto=True, color_continuous_scale="Blues", 
        title="Heatmap of Pairwise Shared Variants", labels=dict(x="Condition B", y="Condition A", color="Shared Variants")
    )
    st.plotly_chart(fig_heat, use_container_width=True)


if filtered_variant_df.empty:
    st.info("No variants available for UpSet plots.")
else:
    tmp_upset = filtered_variant_df.copy()
    tmp_upset["Cell_Line_Short"] = tmp_upset["Cell_Line"].apply(prettify_cell_line)
    tmp_upset["Site_Key"] = (
        tmp_upset["Gene"] + "_" +
        tmp_upset["Chromosome"].astype(str) + ":" +
        tmp_upset["Position"].astype(str) + ":" +
        tmp_upset["Ref_Allele"] + ">" + tmp_upset["Alt_Allele"]
    )
    tmp_upset["Condition"] = tmp_upset["Cell_Line_Short"] + " (" + tmp_upset["Editing_Strategy"] + ")"

    site_conditions = tmp_upset.groupby("Site_Key")["Condition"].apply(lambda x: tuple(sorted(set(x))))
    combination_counts = site_conditions.value_counts()

    pairwise_data = combination_counts[combination_counts.index.map(len) == 2]
    complex_data = combination_counts[combination_counts.index.map(len) > 2]

    st.markdown("### 1. Pairwise Intersections (Exactly 2)")
    if pairwise_data.empty:
        st.info("No pairwise overlaps found.")
    else:
        n_pair = st.slider("Show Top N Pairwise Intersections", 5, 40, 15, key="n_pair_slider")
        top_pairwise = pairwise_data.head(n_pair)

        upset_pair = from_memberships(top_pairwise.index, data=top_pairwise.values)
        fig_pair = plt.figure(figsize=(8, 4.5))
        plot(upset_pair, fig=fig_pair, show_counts="%d", element_size=28)
        st.pyplot(fig_pair)
        plt.close(fig_pair)

    st.markdown("### 2. Complex Overlaps (Shared by > 2)")
    if complex_data.empty:
        st.info("No variants are shared by 3 or more combinations.")
    else:
        n_complex = st.slider("Show Top N Complex Intersections", 5, 40, 15, key="n_complex_slider")
        top_complex = complex_data.head(n_complex)

        upset_complex = from_memberships(top_complex.index, data=top_complex.values)
        fig_complex = plt.figure(figsize=(8, 4.5))
        plot(upset_complex, fig=fig_complex, show_counts="%d", element_size=28)
        st.pyplot(fig_complex)
        plt.close(fig_complex)
    render_bold_dataframe(pairwise_variant_summary.set_index("Gene_Symbol"))


# BAR PLOT & GENE SUMMARY TABLE
st.subheader("Per-Gene Variant Count Table")
if gene_variant_summary.empty:
    st.info("No variant rows remain after the current expression + variant filters.")
else:
    gene_variant_summary = gene_variant_summary.sort_values(by=["Unique_Variant_Sites"], ascending=[False])
    render_bold_dataframe(gene_variant_summary.set_index("Gene_Symbol"))

    fig_gene_bar = px.bar(
        gene_variant_summary.head(30), x="Gene_Symbol", y="Unique_Variant_Sites",
        hover_data=[c for c in ["Unique_Variant_Sites", "Shared_Variant_Sites_>=2_CellLines", "Min_Population_Variant_Frequency", "Max_Population_Variant_Frequency", "Mean_Population_Variant_Frequency", "Raw_Overlap_Count"] if c in gene_variant_summary.columns],
        title="Top Genes by Filtered Variant Count",
        labels={"Gene_Symbol": "Gene", "Unique_Variant_Sites": "# Unique Variants"}
    )
    st.plotly_chart(fig_gene_bar, use_container_width=True)

st.subheader("Variant-Level Detail Table")
if filtered_variant_df.empty:
    st.info("No variant-level records match the current filters.")
else:
    preferred_variant_cols = ["Gene", "Cell_Line", "Editing_Strategy", "Chromosome", "Position", "Ref_Allele", "Alt_Allele", "Population_Variant_Frequency"]
    disp_variant_df = filtered_variant_df[[c for c in preferred_variant_cols if c in filtered_variant_df.columns]]
    render_bold_dataframe(disp_variant_df.set_index("Gene"))

if include_expression:
    st.subheader("Filtered Expression Data Table")
    preferred_cols = ["Gene_Symbol", "Dataset", "Cell_Type", "Replicate", det_col, mean_all_col, mean_detected_col, agg_col, "s_het", "dominant_mutation_count", "Citation_Count", "HPO_Cardiovascular", "HPO_Nervous", "HPO_Metabolism", "HPO_Musculoskeletal"]
    table_cols = [c for c in preferred_cols if c in filtered_df.columns]
    other_cols = [c for c in filtered_df.columns if c not in table_cols]
    disp_expr_df = filtered_df[table_cols + other_cols]
    render_bold_dataframe(disp_expr_df.set_index("Gene_Symbol"))

st.markdown("---")
download_col1, download_col2, download_col3 = st.columns(3)

if include_expression:
    with download_col1:
        st.download_button(label="Download filtered expression table", data=filtered_df.to_csv(index=False).encode("utf-8"), file_name=f"dnd_expression_filtered_{qc_level}.csv", mime="text/csv")

with download_col2:
    st.download_button(label="Download gene-level variant summary", data=gene_variant_summary.to_csv(index=False).encode("utf-8") if not gene_variant_summary.empty else b"", file_name="dnd_gene_variant_summary.csv", mime="text/csv")

with download_col3:
    st.download_button(label="Download variant-level filtered table", data=filtered_variant_df.to_csv(index=False).encode("utf-8") if not filtered_variant_df.empty else b"", file_name="dnd_variant_filtered.csv", mime="text/csv")


















# import csv
# from pathlib import Path

# import pandas as pd
# import plotly.express as px
# import streamlit as st

# # page config
# # must be set before any other Streamlit commands
# st.set_page_config(
#     page_title="DnD Gene Detectability Explorer",
#     layout="wide",
#     page_icon="🧬"
# )



# # -----------------------------------------------
# # File paths - adjust as needed on github
# CARDIAC_CSV = "detection_recheck_outputs/master_df_cardiac_qc3.csv"
# NEURON_CSV = "detection_recheck_outputs/master_df_neuron_qc3.csv"
# ANNOT_XLSX = "detection_recheck_outputs/genes_other_info.xlsx"

# # variant-level files
# VARIANT_CSV = "detection_recheck_outputs/master_variant_table_with_population_freq.csv"
# SUMMARY_TSV = "detection_recheck_outputs/summary_counts.tsv"

# ANNOT_SHEET = 0


# # ------------------------------------------------

# # some helper functions
# QC_LEVELS = ["Raw", "LooseQC", "StrongQC"]

# BASE_METRICS = [
#     "Detection_Rate_%",
#     "Mean_Expr_All",
#     "Mean_Expr_Detected",
#     "Aggregated_CPM"
# ]

# VARIANT_EXPECTED_COLUMNS = [
#     "Cell_Line",
#     "Editing_Strategy",
#     "Gene",
#     "Chromosome",
#     "Position",
#     "Ref_Allele",
#     "Alt_Allele",
#     "Allele_Frequency",
#     "Population_Variant_Frequency",
#     "Population_Frequency_Matched"
# ]

# def qc_col(qc_level: str, metric: str) -> str:
#     return f"{qc_level}_{metric}"


# def safe_numeric(series, fill_value=0):
#     return pd.to_numeric(series, errors="coerce").fillna(fill_value)


# def add_dataset_label(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
#     df = df.copy()
#     df["Dataset"] = dataset_name
#     return df


# def normalize_bool_col(df: pd.DataFrame, col: str) -> pd.Series:
#     if col not in df.columns:
#         return pd.Series([False] * len(df), index=df.index)
#     s = df[col]
#     if s.dtype == bool:
#         return s.fillna(False)
#     return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


# def pick_annotation_key(df_metrics: pd.DataFrame, df_annot: pd.DataFrame):
#     candidates = ["Gene_Symbol", "hgnc_symbol", "gene_symbol", "Symbol"]
#     for c in candidates:
#         if c in df_metrics.columns and c in df_annot.columns:
#             return c, c

#     if "Gene_Symbol" in df_metrics.columns and "hgnc_symbol" in df_annot.columns:
#         return "Gene_Symbol", "hgnc_symbol"

#     raise ValueError(
#         "Could not find a matching gene key between metrics files and annotation file."
#     )


# def robust_read_variant_csv(path: str) -> pd.DataFrame:
#     """
#     Robust reader for variant CSVs derived from VCF fields.
#     Handles multiallelic rows where ALT and AF may contain commas.

#     Assumes:
#     - first 6 columns are fixed
#     - last 1 column is fixed:
#         Population_Variant_Frequency
#     """
#     repaired_rows = []

#     with open(path, "r", newline="", encoding="utf-8") as f:
#         reader = csv.reader(f)
#         header = next(reader)

#         # If header already matches perfectly, still keep robust parsing for rows
#         if len(header) < 8:
#             raise ValueError(f"Unexpected header in variant CSV: {header}")

#         for line_num, row in enumerate(reader, start=2):
#             if len(row) == len(VARIANT_EXPECTED_COLUMNS):
#                 repaired_rows.append(row)
#                 continue

#             if len(row) < 5:
#                 raise ValueError(
#                     f"Line {line_num}: too few fields ({len(row)}): {row}"
#                 )

#             prefix = row[:6]
#             suffix = row[-1:]
#             middle = row[6:-1]

#             if len(middle) < 2 or len(middle) % 2 != 0:
#                 raise ValueError(
#                     f"Line {line_num}: cannot split ALT/AF fields cleanly: {row}"
#                 )

#             half = len(middle) // 2
#             alt = ",".join(middle[:half])
#             af = ",".join(middle[half:])

#             repaired_rows.append(prefix + [alt, af] + suffix)

#     df = pd.DataFrame(repaired_rows, columns=VARIANT_EXPECTED_COLUMNS)
#     return df


# def load_optional_summary(summary_path: str) -> pd.DataFrame:
#     p = Path(summary_path)
#     if not p.exists():
#         return pd.DataFrame()

#     try:
#         df = pd.read_csv(p, sep="\t")
#         return df
#     except Exception:
#         return pd.DataFrame()


# def build_variant_gene_summary(variant_df: pd.DataFrame) -> pd.DataFrame:
#     if variant_df.empty:
#         return pd.DataFrame(columns=[
#             "Gene_Symbol",
#             #Filtered_Variant_Count",
#             "Unique_Variant_Sites",
#             "Min_Population_Variant_Frequency",
#             "Max_Population_Variant_Frequency",
#             "Mean_Population_Variant_Frequency"
#         ])

#     tmp = variant_df.copy()
#     tmp["Gene_Symbol"] = tmp["Gene"].astype(str)
#     tmp["Variant_Site_Key"] = (
#         tmp["Chromosome"].astype(str) + ":" +
#         tmp["Position"].astype(str) + ":" +
#         tmp["Ref_Allele"].astype(str) + ">" +
#         tmp["Alt_Allele"].astype(str)
#     )

#     summary = (
#         tmp.groupby("Gene_Symbol", dropna=False)
#         .agg(
#             #Filtered_Variant_Count=("Gene_Symbol", "size"),
#             Unique_Variant_Sites=("Variant_Site_Key", "nunique"),
#             Min_Population_Variant_Frequency=("Population_Variant_Frequency", "min"),
#             Max_Population_Variant_Frequency=("Population_Variant_Frequency", "max"),
#             Mean_Population_Variant_Frequency=("Population_Variant_Frequency", "mean"),
#         )
#         .reset_index()
#     )

#     return summary


# def coerce_variant_df_types(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()

#     for col in [
#         "Position",
#         #"Allele_Frequency",
#         "Population_Variant_Frequency",
#     ]:
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors="coerce")

#     if "Population_Frequency_Matched" in df.columns:
#         df["Population_Frequency_Matched"] = (
#             df["Population_Frequency_Matched"]
#             .astype(str)
#             .str.lower()
#             .isin(["true", "1", "yes", "y"])
#         )

#     for col in ["Cell_Line", "Editing_Strategy", "Gene", "Chromosome", "Ref_Allele", "Alt_Allele"]:
#         if col in df.columns:
#             df[col] = df[col].astype(str)

#     return df



# # display variant labels to concise cell lines
# CELL_LINE_LABELS = {
#     "KOLF2-ARID2-A02": "KOLF2",
#     "WTB_variants_PASS": "WTB",
#     "WTC_variants_PASS": "WTC",
#     "cN8-hNIL": "WTD (cN8-hNIL)",
# }

# def prettify_cell_line(x: str) -> str:
#     return CELL_LINE_LABELS.get(x, x)




# # load data
# @st.cache_data
# def load_data():
#     cardiac = pd.read_csv(CARDIAC_CSV)
#     neuron = pd.read_csv(NEURON_CSV)
#     annot = pd.read_excel(ANNOT_XLSX, sheet_name=ANNOT_SHEET)

#     cardiac = add_dataset_label(cardiac, "Cardiac")
#     neuron = add_dataset_label(neuron, "Neuron")

#     metrics_df = pd.concat([cardiac, neuron], ignore_index=True)

#     left_key, right_key = pick_annotation_key(metrics_df, annot)
#     merged = metrics_df.merge(
#         annot,
#         how="left",
#         left_on=left_key,
#         right_on=right_key
#     )

#     for col in ["dominant_mutation_count", "Citation_Count"]:
#         if col in merged.columns:
#             merged[col] = safe_numeric(merged[col], fill_value=0)

#     for col in [
#         "HPO_Cardiovascular",
#         "HPO_Nervous",
#         "HPO_Metabolism",
#         "HPO_Musculoskeletal"
#     ]:
#         if col in merged.columns:
#             merged[col] = normalize_bool_col(merged, col)

#     variant_df = pd.DataFrame()
#     if Path(VARIANT_CSV).exists():
#         variant_df = robust_read_variant_csv(VARIANT_CSV)
#         variant_df = coerce_variant_df_types(variant_df)

#     summary_df = load_optional_summary(SUMMARY_TSV)

#     return merged, variant_df, summary_df


# df, variant_master_df, summary_counts_df = load_data()



# # sidebar filters
# st.sidebar.title("Apply Filters")

# # -----------------------------
# # Experimental subset
# # -----------------------------
# st.sidebar.subheader("Dataset & Conditions")

# datasets = sorted(df["Dataset"].dropna().unique().tolist())
# selected_datasets = st.sidebar.multiselect(
#     "Dataset",
#     datasets,
#     default=datasets
# )

# cell_types = sorted(df["Cell_Type"].dropna().unique().tolist())
# selected_cells = st.sidebar.multiselect(
#     "Cell Type / Stage",
#     cell_types,
#     default=cell_types
# )

# replicates = sorted(df["Replicate"].dropna().unique().tolist())
# selected_replicates = st.sidebar.multiselect(
#     "Replicate",
#     replicates,
#     default=replicates
# )

# qc_level = st.sidebar.selectbox(
#     "QC Level",
#     QC_LEVELS,
#     index=1
# )

# # -----------------------------
# # Metric column selection
# # -----------------------------
# st.sidebar.subheader("Expression Metrics")

# det_col = qc_col(qc_level, "Detection_Rate_%")
# mean_all_col = qc_col(qc_level, "Mean_Expr_All")
# mean_detected_col = qc_col(qc_level, "Mean_Expr_Detected")
# agg_col = qc_col(qc_level, "Aggregated_CPM")

# required_cols = [det_col, mean_all_col, mean_detected_col, agg_col]
# missing_cols = [c for c in required_cols if c not in df.columns]

# if missing_cols:
#     st.error(f"Missing expected QC-specific columns: {missing_cols}")
#     st.stop()

# min_det = st.sidebar.slider(
#     f"Min Detection Rate (%) [{qc_level}]",
#     0.0,
#     100.0,
#     0.0,
#     step=1.0
# )

# max_agg = float(pd.to_numeric(df[agg_col], errors="coerce").fillna(0).max())
# min_cpm = st.sidebar.slider(
#     f"Min Aggregated CPM [{qc_level}]",
#     0.0,
#     max(0.0, max_agg),
#     0.0,
#     step=max(1.0, max_agg / 100 if max_agg > 100 else 1.0)
# )

# max_mean_detected = float(pd.to_numeric(df[mean_detected_col], errors="coerce").fillna(0).max())
# min_mean_detected = st.sidebar.slider(
#     f"Min Mean Expr Detected [{qc_level}]",
#     0.0,
#     max(0.0, max_mean_detected),
#     0.0,
#     step=max(0.01, max_mean_detected / 100 if max_mean_detected > 1 else 0.01)
# )

# # -----------------------------
# # Phenotype filters
# # -----------------------------
# st.sidebar.subheader("Clinical Phenotypes (HPO)")
# st.sidebar.markdown("*Check to only show genes associated with: Abnormality in*")

# req_cardio = st.sidebar.checkbox("Cardiovascular System")
# req_neuro = st.sidebar.checkbox("Nervous System")
# req_metab = st.sidebar.checkbox("Metabolism / Homeostasis")
# req_musculo = st.sidebar.checkbox("Musculoskeletal System")

# # -----------------------------
# # Literature / genetics filters
# # -----------------------------
# st.sidebar.subheader("Literature & Genetics")

# if "dominant_mutation_count" in df.columns:
#     max_mut = int(df["dominant_mutation_count"].fillna(0).max())
#     min_mut = st.sidebar.number_input(
#         "Min Dominant Mutation Count",
#         min_value=0,
#         max_value=max_mut,
#         value=0
#     )
# else:
#     min_mut = 0

# if "Citation_Count" in df.columns:
#     max_cite = int(df["Citation_Count"].fillna(0).max())
#     min_cite = st.sidebar.number_input(
#         "Min Citation Count",
#         min_value=0,
#         max_value=max_cite,
#         value=0
#     )
# else:
#     min_cite = 0

# # -----------------------------
# # Variant filters
# # -----------------------------
# st.sidebar.subheader("Variant Filters")

# if variant_master_df.empty:
#     st.sidebar.info("Variant table not found at VARIANT_CSV path.")
#     selected_variant_cell_lines = []
#     selected_variant_strategies = []
#     min_pop_freq = 0.0
#     keep_missing_popfreq = True
# else:
#     variant_cell_lines = sorted(variant_master_df["Cell_Line"].dropna().unique().tolist())

#     selected_variant_cell_lines = st.sidebar.multiselect(
#         "Variant Cell Line",
#         variant_cell_lines,
#         default=variant_cell_lines,
#         format_func=prettify_cell_line,
#         key="variant_cell_line_multiselect"
#     )

#     variant_strategies = sorted(variant_master_df["Editing_Strategy"].dropna().unique().tolist())
#     selected_variant_strategies = st.sidebar.multiselect(
#         "Editing Strategy",
#         variant_strategies,
#         default=variant_strategies,
#         key="variant_strategy_multiselect"
#     )

#     max_popfreq = float(
#         pd.to_numeric(
#             variant_master_df["Population_Variant_Frequency"],
#             errors="coerce"
#         ).dropna().max()
#     ) if variant_master_df["Population_Variant_Frequency"].notna().any() else 1.0

#     min_pop_freq = st.sidebar.slider(
#         "Min Population Variant Frequency",
#         0.0,
#         max(1.0, max_popfreq),
#         0.0,
#         step=0.01,
#         key="variant_population_freq_slider"
#     )

#     # keep_missing_popfreq = st.sidebar.checkbox(
#     #     "Keep variants with missing population frequency",
#     #     value=True,
#     #     key="variant_keep_missing_popfreq_checkbox"
#     # )










# # -------------------------------------------
# # filter for expression
# base_df = df[
#     (df["Dataset"].isin(selected_datasets)) &
#     (df["Cell_Type"].isin(selected_cells)) &
#     (df["Replicate"].isin(selected_replicates))
# ].copy()

# for c in [det_col, mean_all_col, mean_detected_col, agg_col]:
#     base_df[c] = safe_numeric(base_df[c], fill_value=0)

# filtered_df = base_df[
#     (base_df[det_col] >= min_det) &
#     (base_df[agg_col] >= min_cpm) &
#     (base_df[mean_detected_col] >= min_mean_detected)
# ].copy()

# if "dominant_mutation_count" in filtered_df.columns:
#     filtered_df = filtered_df[filtered_df["dominant_mutation_count"] >= min_mut]

# if "Citation_Count" in filtered_df.columns:
#     filtered_df = filtered_df[filtered_df["Citation_Count"] >= min_cite]

# if req_cardio and "HPO_Cardiovascular" in filtered_df.columns:
#     filtered_df = filtered_df[filtered_df["HPO_Cardiovascular"] == True]

# if req_neuro and "HPO_Nervous" in filtered_df.columns:
#     filtered_df = filtered_df[filtered_df["HPO_Nervous"] == True]

# if req_metab and "HPO_Metabolism" in filtered_df.columns:
#     filtered_df = filtered_df[filtered_df["HPO_Metabolism"] == True]

# if req_musculo and "HPO_Musculoskeletal" in filtered_df.columns:
#     filtered_df = filtered_df[filtered_df["HPO_Musculoskeletal"] == True]

# filtered_gene_set = set(filtered_df["Gene_Symbol"].dropna().astype(str).unique().tolist()) \
#     if "Gene_Symbol" in filtered_df.columns else set()




# # -------------------------------------------------
# # filter for variant
# if variant_master_df.empty:
#     filtered_variant_df = pd.DataFrame()
#     gene_variant_summary = pd.DataFrame()
# else:
#     filtered_variant_df = variant_master_df.copy()

#     if selected_variant_cell_lines:
#         filtered_variant_df = filtered_variant_df[
#             filtered_variant_df["Cell_Line"].isin(selected_variant_cell_lines)
#         ]

#     if selected_variant_strategies:
#         filtered_variant_df = filtered_variant_df[
#             filtered_variant_df["Editing_Strategy"].isin(selected_variant_strategies)
#         ]

#     if filtered_gene_set:
#         filtered_variant_df = filtered_variant_df[
#             filtered_variant_df["Gene"].astype(str).isin(filtered_gene_set)
#         ]
#     else:
#         filtered_variant_df = filtered_variant_df.iloc[0:0].copy()

#     # if keep_missing_popfreq:
#     #     filtered_variant_df = filtered_variant_df[
#     #         filtered_variant_df["Population_Variant_Frequency"].isna() |
#     #         (filtered_variant_df["Population_Variant_Frequency"] >= min_pop_freq)
#     #     ]
#     # else:
#     #     filtered_variant_df = filtered_variant_df[
#     #         filtered_variant_df["Population_Variant_Frequency"] >= min_pop_freq
#     #     ]

#     gene_variant_summary = build_variant_gene_summary(filtered_variant_df)

#     # merge optional raw overlap counts if the TSV has the expected schema
#     expected_summary_cols = {"Cell_Line", "Editing_Strategy", "Gene", "Matched_Variant_Count"}
#     if not summary_counts_df.empty and expected_summary_cols.issubset(summary_counts_df.columns):
#         raw_counts = (
#             summary_counts_df[
#                 summary_counts_df["Cell_Line"].isin(selected_variant_cell_lines)
#                 if selected_variant_cell_lines else summary_counts_df["Cell_Line"].notna()
#             ]
#             .copy()
#         )
#         if selected_variant_strategies:
#             raw_counts = raw_counts[raw_counts["Editing_Strategy"].isin(selected_variant_strategies)]

#         raw_counts = (
#             raw_counts.groupby("Gene", dropna=False)["Matched_Variant_Count"]
#             .sum()
#             .reset_index()
#             .rename(columns={
#                 "Gene": "Gene_Symbol",
#                 "Matched_Variant_Count": "Raw_Overlap_Count"
#             })
#         )

#         gene_variant_summary = gene_variant_summary.merge(
#             raw_counts,
#             how="left",
#             on="Gene_Symbol"
#         )

# # -----------------------------------------------
# # dashboard
# st.title("DnD Gene Detectability Explorer")
# st.caption(f"Currently viewing metrics from: {qc_level}")

# col1, col2, col3, col4 = st.columns(4)
# unique_gene_count = filtered_df["Gene_Symbol"].nunique() if "Gene_Symbol" in filtered_df.columns else 0
# col1.metric("Genes Matching Expression Criteria", unique_gene_count)

# if not filtered_df.empty:
#     col2.metric("Highest Detection Rate", f"{filtered_df[det_col].max():.2f}%")
#     col3.metric("Highest Aggregated CPM", f"{filtered_df[agg_col].max():.2f}")
#     col4.metric("Highest Mean Expr Detected", f"{filtered_df[mean_detected_col].max():.2f}")

# # NEW: variant summary cards
# st.subheader("Variant Summary for Currently Filtered Genes")

# v1, v2, v3, v4 = st.columns(4)
# if filtered_variant_df.empty:
#     v1.metric("Genes With Filtered Variants", 0)
#     v2.metric("Filtered Variant Rows", 0)
#     v3.metric("Unique Variant Sites", 0)
#     v4.metric("Max Population Var Freq", "NA")
# else:
#     tmp_var = filtered_variant_df.copy()
#     tmp_var["Variant_Site_Key"] = (
#         tmp_var["Chromosome"].astype(str) + ":" +
#         tmp_var["Position"].astype(str) + ":" +
#         tmp_var["Ref_Allele"].astype(str) + ">" +
#         tmp_var["Alt_Allele"].astype(str)
#     )
#     v1.metric("Genes With Filtered Variants", tmp_var["Gene"].nunique())
#     v2.metric("Filtered Variant Rows", len(tmp_var))
#     v3.metric("Unique Variant Sites", tmp_var["Variant_Site_Key"].nunique())

#     max_pop_val = pd.to_numeric(tmp_var["Population_Variant_Frequency"], errors="coerce")
#     if max_pop_val.notna().any():
#         v4.metric("Max Population Var Freq", f"{max_pop_val.max():.3f}")
#     else:
#         v4.metric("Max Population Var Freq", "NA")



# # ---------------------------------------------------
# #histogram
# st.subheader("Detection Rate Distribution")

# if not base_df.empty:
#     fig_hist = px.histogram(
#         base_df,
#         x=det_col,
#         color="Dataset",
#         nbins=50,
#         barmode="overlay",
#         opacity=0.65,
#         title=f"Distribution within Selected Conditions ({qc_level})",
#         labels={det_col: f"{qc_level} Detection Rate (%)"}
#     )

#     fig_hist.add_vline(
#         x=min_det,
#         line_dash="dash",
#         line_color="red",
#         annotation_text=f"Cutoff: {min_det}%",
#         annotation_position="top right"
#     )

#     st.plotly_chart(fig_hist, use_container_width=True)
# else:
#     st.warning("No data match the current categorical filters.")




# # st.subheader("Expression Distribution")

# # if not filtered_df.empty:
# #     fig = px.scatter(
# #         filtered_df,
# #         x=det_col,
# #         y=agg_col,
# #         color="Cell_Type",
# #         symbol="Dataset",
# #         hover_data=[
# #             c for c in [
# #                 "Gene_Symbol",
# #                 "Dataset",
# #                 "Cell_Type",
# #                 "Replicate",
# #                 "dominant_mutation_count",
# #                 "Citation_Count",
# #                 mean_detected_col,
# #                 mean_all_col
# #             ] if c in filtered_df.columns
# #         ],
# #         title=f"Detection Rate vs Aggregated CPM ({qc_level})",
# #         labels={
# #             det_col: "Detection Rate (%)",
# #             agg_col: "Aggregated CPM"
# #         },
# #         log_y=True
# #     )
# #     fig.update_traces(marker=dict(size=8, opacity=0.75))
# #     st.plotly_chart(fig, use_container_width=True)
# # else:
# #     st.warning("No genes match the current filter criteria.")

# # Scatterplot for expression
# st.subheader("Detection Rate vs Mean Expression Among Detected Cells")

# if not filtered_df.empty:
#     fig2 = px.scatter(
#         filtered_df,
#         x=det_col,
#         y=mean_detected_col,
#         color="Cell_Type",
#         symbol="Dataset",
#         hover_data=[
#             c for c in [
#                 "Gene_Symbol",
#                 "Dataset",
#                 "Cell_Type",
#                 "Replicate",
#                 agg_col,
#                 "dominant_mutation_count",
#                 "Citation_Count"
#             ] if c in filtered_df.columns
#         ],
#         title=f"Detection Rate vs Mean Expr Detected ({qc_level})",
#         labels={
#             det_col: "Detection Rate (%)",
#             mean_detected_col: "Mean Expr Detected"
#         }
#     )
#     fig2.update_traces(marker=dict(size=8, opacity=0.75))
#     st.plotly_chart(fig2, use_container_width=True)

# # -----------------------------------------------
# # Variant table
# st.subheader("Per-Gene Variant Count Table")

# if gene_variant_summary.empty:
#     st.info("No variant rows remain after the current expression + variant filters.")
# else:
#     gene_variant_summary = gene_variant_summary.sort_values(
#         by=[#"Filtered_Variant_Count", 
#             "Unique_Variant_Sites"],
#         ascending=[False]
#     )
#     st.dataframe(gene_variant_summary, use_container_width=True)

#     fig_gene_bar = px.bar(
#         gene_variant_summary.head(30),
#         x="Gene_Symbol",
#         y="Unique_Variant_Sites",
#         hover_data=[
#             c for c in [
#                 "Unique_Variant_Sites",
#                 "Min_Population_Variant_Frequency",
#                 "Max_Population_Variant_Frequency",
#                 "Mean_Population_Variant_Frequency",
#                 "Raw_Overlap_Count"
#             ] if c in gene_variant_summary.columns
#         ],
#         title="Top Genes by Filtered Variant Count",
#         labels={
#             "Gene_Symbol": "Gene",
#             "Unique_Variant_Sites": "# Unique Variants"
#         }
#     )
#     st.plotly_chart(fig_gene_bar, use_container_width=True)


#     # TODO: group by cell line and strategy on the barplot.? but it will be different genes...



# # detailed variant table?
# st.subheader("Variant-Level Detail Table")

# if filtered_variant_df.empty:
#     st.info("No variant-level records match the current filters.")
# else:
#     preferred_variant_cols = [
#         "Gene",
#         "Cell_Line",
#         "Editing_Strategy",
#         "Chromosome",
#         "Position",
#         "Ref_Allele",
#         "Alt_Allele",
#         #"Allele_Frequency",
#         "Population_Variant_Frequency"
#     ]

#     st.dataframe(
#         filtered_variant_df[[c for c in preferred_variant_cols if c in filtered_variant_df.columns]],  # only the preferred columns that exist
#         use_container_width=True
#     )

# # -------------------------------------------------
# # Filtered expression table
# st.subheader("Filtered Expression Data Table")

# preferred_cols = [
#     "Gene_Symbol",
#     "Dataset",
#     "Cell_Type",
#     "Replicate",
#     det_col,
#     mean_all_col,
#     mean_detected_col,
#     agg_col,
#     "dominant_mutation_count",
#     "Citation_Count",
#     "HPO_Cardiovascular",
#     "HPO_Nervous",
#     "HPO_Metabolism",
#     "HPO_Musculoskeletal"
# ]

# table_cols = [c for c in preferred_cols if c in filtered_df.columns]
# other_cols = [c for c in filtered_df.columns if c not in table_cols]

# st.dataframe(filtered_df[table_cols + other_cols], use_container_width=True)


# # ----------------------------------------------------------------
# # Download
# download_col1, download_col2, download_col3 = st.columns(3)

# with download_col1:
#     st.download_button(
#         label="Download filtered expression table",
#         data=filtered_df.to_csv(index=False).encode("utf-8"),
#         file_name=f"dnd_expression_filtered_{qc_level}.csv",
#         mime="text/csv"
#     )

# with download_col2:
#     st.download_button(
#         label="Download gene-level variant summary",
#         data=gene_variant_summary.to_csv(index=False).encode("utf-8") if not gene_variant_summary.empty else b"",
#         file_name="dnd_gene_variant_summary.csv",
#         mime="text/csv"
#     )

# with download_col3:
#     st.download_button(
#         label="Download variant-level filtered table",
#         data=filtered_variant_df.to_csv(index=False).encode("utf-8") if not filtered_variant_df.empty else b"",
#         file_name="dnd_variant_filtered.csv",
#         mime="text/csv"
#     )
    
    
    
    
    
    
 