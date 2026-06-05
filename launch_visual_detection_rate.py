
import csv
import itertools
import pickle
from pathlib import Path
import warnings

import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from upsetplot import plot, from_memberships
from venn import venn as draw_venn_diagram
import streamlit as st

warnings.filterwarnings(
    "ignore",
    message="A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.",
    category=FutureWarning,
    module="upsetplot.plotting",
)

# page config
st.set_page_config(
    page_title="D&D Gene Detectability Explorer",
    layout="wide",
    page_icon="🧬"
)

st.markdown("""
    <style>
    :root {
        --capra-accent: #2f6f73;
        --capra-header-bg: color-mix(in srgb, currentColor 12%, transparent);
        --capra-highlight-bg: color-mix(in srgb, var(--capra-accent) 16%, transparent);
        --capra-highlight-text: inherit;
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --capra-header-bg: color-mix(in srgb, currentColor 18%, transparent);
            --capra-highlight-bg: color-mix(in srgb, var(--capra-accent) 28%, transparent);
            --capra-highlight-text: #f3f6f7;
        }
    }

    [data-testid="stDataFrame"] [role="columnheader"] {
        font-weight: 700 !important;
        background-color: var(--capra-header-bg) !important;
    }

    [data-testid="stDataFrame"] [role="rowheader"] {
        font-weight: 700 !important;
    }

    [data-testid="stSidebar"] {
        border-right: 3px solid var(--capra-accent);
        background: inherit;
    }

    [data-testid="stSidebar"] > div:first-child {
        background: inherit;
    }

    .tutorial-filter-highlight {
        border: 2px solid var(--capra-accent);
        border-radius: 6px;
        background: var(--capra-highlight-bg);
        color: var(--capra-highlight-text);
        padding: 0.55rem 0.7rem;
        margin: 0.75rem 0 0.35rem 0;
        font-weight: 800;
        text-decoration: underline;
    }

    .sidebar-section-title {
        margin: 0.75rem 0 0.2rem 0;
        font-weight: 800;
    }
    </style>
""", unsafe_allow_html=True)

CARDIAC_CSV = "data/master_new_cardiac_qc3.csv"
NEURON_CSV = "data/master_new_neuron_qc3.csv"
IPSC_HEPATOCYTE_EXPRESSION_CSV = "data/iPSC-liver_expression_stats.csv"
IPSC_HEPATOCYTE_QC_CSV = "data/iPSC-liver_detection_qc_thresholds.csv"
IPSC_HEPATOCYTE_QC_PREFIX = "iPSC_Hepatocyte_QC"
ANNOT_XLSX = "data/genes_other_info.xlsx"
VARIANT_FILE = "data/pre_pam_variant_table.xlsx"
LEGACY_VARIANT_FILE = "data/FINAL_Combined_Master_Variant_Table.xlsx"
VARIANT_CSV_DIR = "data/pre_pam_variant_table_csv"
VARIANT_CSV_FALLBACK_DIRS = [
    "data/pre_pam_variant_table_test_csv",
]
SUMMARY_TSV = "data/summary_counts.tsv" 
EPI_PROM_GENE_CSV = "data/epi_silenceable_100_200_prom.csv"
EPI_PROM_SITE_PKL = "data/epi_silenceable_100_200_sites.pkl"
CLINGEN_DOSAGE_CSV = "data/Clingen-Curation-Summary.csv"
EDITOR_ASSIGNMENT_CSV = "data/master_spreadsheets/editor_assignment_info.csv"

EDITOR_CATEGORY_OPTIONS = [
    "ABE",
    "CBE",
    "both ABE and CBE",
    "No editor assignment info",
]


INPUT_SUMMARY_GENE_COUNT = 593
INPUT_SUMMARY_CELL_LINE_COUNT = 5
INPUT_SUMMARY_EDITING_STRATEGY_COUNT = 3


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

IPSC_HEPATOCYTE_DATASET_LABELS = {
    "GSE141183": "GSE141183 (2021 day 30)",
    "GSM5009365": "GSE164417 (2021 day 21, less ideal)",
    "GSM7903846": "GSE247961 (2025 10x, post-day 21)",
}

def qc_col(qc_level: str, metric: str) -> str:
    return f"{qc_level}_{metric}"

def safe_numeric(series, fill_value=0):
    return pd.to_numeric(series, errors="coerce").fillna(fill_value)

def add_dataset_label(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    df = df.copy()
    df["Dataset"] = dataset_name
    return df

def normalize_single_qc_expression(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]

    metric_aliases = {
        "Detection_Rate_%": "Detection_Rate_%",
        "Mean_CPM_percell_All": "Mean_Expr_All",
        "Mean_Log1pCPM_percell_All": "Mean_Expr_All",
        "Mean_CPM_percell_Only_Detected": "Mean_Expr_Detected",
        "Mean_Log1pCPM_percell_Only_Detected": "Mean_Expr_Detected",
        "Aggregated_CPM": "Aggregated_CPM",
    }

    normalized = df.copy()
    for source_col, metric_name in metric_aliases.items():
        if source_col not in normalized.columns:
            continue
        for qc_level in QC_LEVELS:
            normalized[qc_col(qc_level, metric_name)] = normalized[source_col]

    if "Cell_Type" not in normalized.columns:
        normalized["Cell_Type"] = "iPSC-liver/hepatocyte"
    if "Source_File" not in normalized.columns:
        source_cols = [c for c in ["Sample_ID", "Dataset"] if c in normalized.columns]
        if source_cols:
            normalized["Source_File"] = normalized[source_cols[0]].astype(str)
        else:
            normalized["Source_File"] = "iPSC-liver_expression_stats.csv"

    return normalized

def ipsc_hepatocyte_label(dataset_id) -> str:
    dataset_id = str(dataset_id)
    return IPSC_HEPATOCYTE_DATASET_LABELS.get(dataset_id, dataset_id)

def normalize_ipsc_hepatocyte_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]
    df["QC_Min_Genes"] = pd.to_numeric(df["QC_Min_Genes"], errors="coerce").astype("Int64")

    metric_aliases = {
        "Detection_Rate_%": "Detection_Rate_%",
        "Mean_CPM_percell_All": "Mean_Expr_All",
        "Mean_Log1pCPM_percell_All": "Mean_Expr_All",
        "Mean_CPM_percell_Only_Detected": "Mean_Expr_Detected",
        "Mean_Log1pCPM_percell_Only_Detected": "Mean_Expr_Detected",
        "Aggregated_CPM": "Aggregated_CPM",
    }
    available_aliases = {
        source_col: metric_name
        for source_col, metric_name in metric_aliases.items()
        if source_col in df.columns
    }
    index_cols = ["Gene_Symbol", "Dataset", "Sample_ID", "Cell_Type", "Replicate"]

    tidy = df[index_cols + ["QC_Min_Genes"] + list(available_aliases)].rename(columns=available_aliases)
    pivoted = tidy.pivot_table(
        index=index_cols,
        columns="QC_Min_Genes",
        values=sorted(set(available_aliases.values())),
        aggfunc="first",
    )
    pivoted.columns = [
        f"{IPSC_HEPATOCYTE_QC_PREFIX}_{int(qc_min)}_{metric}"
        for metric, qc_min in pivoted.columns
    ]
    normalized = pivoted.reset_index()

    normalized["Source_Dataset"] = normalized["Dataset"].astype(str)
    normalized["Source_File"] = normalized["Sample_ID"].astype(str)
    normalized["Cell_Type"] = "iPSC-liver/hepatocyte"
    normalized["Detectability_Condition"] = normalized["Source_Dataset"].map(ipsc_hepatocyte_label)

    default_qc_by_level = {
        "Raw": 0,
        "LooseQC": 200,
        "StrongQC": 500,
    }
    qc_values = ipsc_hepatocyte_qc_min_gene_options(normalized)
    for qc_level, default_qc in default_qc_by_level.items():
        selected_qc = default_qc if default_qc in qc_values else (qc_values[-1] if qc_values else None)
        if selected_qc is None:
            continue
        for metric in BASE_METRICS:
            source_col = f"{IPSC_HEPATOCYTE_QC_PREFIX}_{selected_qc}_{metric}"
            if source_col in normalized.columns:
                normalized[qc_col(qc_level, metric)] = normalized[source_col]

    return normalized

def ipsc_hepatocyte_qc_min_gene_options(df: pd.DataFrame) -> list[int]:
    prefix = f"{IPSC_HEPATOCYTE_QC_PREFIX}_"
    suffix = "_Detection_Rate_%"
    options = []
    for col in df.columns:
        if not isinstance(col, str) or not col.startswith(prefix) or not col.endswith(suffix):
            continue
        qc_value = col[len(prefix):-len(suffix)]
        try:
            options.append(int(qc_value))
        except ValueError:
            pass
    return sorted(set(options))

def load_ipsc_hepatocyte_metrics() -> pd.DataFrame:
    expression_path = Path(IPSC_HEPATOCYTE_EXPRESSION_CSV)
    qc_path = Path(IPSC_HEPATOCYTE_QC_CSV)

    if qc_path.exists():
        liver = pd.read_csv(qc_path)
        liver = normalize_ipsc_hepatocyte_thresholds(liver)
    elif expression_path.exists():
        liver = pd.read_csv(expression_path)
        liver = normalize_single_qc_expression(liver)
        if "Dataset" in liver.columns:
            liver["Source_Dataset"] = liver["Dataset"].astype(str)
            liver["Cell_Type"] = "iPSC-liver/hepatocyte"
            liver["Detectability_Condition"] = liver["Source_Dataset"].map(ipsc_hepatocyte_label)
    else:
        return pd.DataFrame()

    return add_dataset_label(liver, "liver")

def add_detectability_context(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Cell_Type" not in df.columns:
        df["Cell_Type"] = df.get("Dataset", "Unknown")

    for col in ["Cell_Line", "Developmental_Stage"]:
        if col not in df.columns:
            df[col] = pd.NA

    cardiac_split = (
        df["Dataset"].astype(str).eq("Cardiac") &
        df["Cell_Line"].notna() &
        df["Developmental_Stage"].notna()
    )

    df["Detectability_Cell_Type"] = df["Dataset"].astype(str)
    if "Detectability_Condition" not in df.columns:
        df["Detectability_Condition"] = df["Cell_Type"]
    else:
        df["Detectability_Condition"] = df["Detectability_Condition"].fillna(df["Cell_Type"])
    df["Detectability_Condition"] = df["Detectability_Condition"].astype(str)
    df["Detectability_Context"] = df["Detectability_Condition"].astype(str)
    df.loc[cardiac_split, "Detectability_Context"] = (
        df.loc[cardiac_split, "Cell_Line"].astype(str) + " " +
        df.loc[cardiac_split, "Developmental_Stage"].astype(str) +
        " cardiomyocytes"
    )
    df.loc[cardiac_split, "Detectability_Condition"] = (
        df.loc[cardiac_split, "Cell_Line"].astype(str) + " " +
        df.loc[cardiac_split, "Developmental_Stage"].astype(str)
    )

    df["Detectability_Context"] = df["Dataset"].astype(str) + " | " + df["Detectability_Context"].astype(str)
    df["Detectability_Label"] = (
        df["Detectability_Cell_Type"].astype(str) + " | " +
        df["Detectability_Condition"].astype(str)
    )
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

def parse_clingen_score(assertion_value) -> float:
    if pd.isna(assertion_value):
        return pd.NA

    text = str(assertion_value).strip()
    if not text:
        return pd.NA

    prefix = text.split("-", 1)[0].strip()
    try:
        return int(prefix)
    except Exception:
        return pd.NA

def simplify_clingen_assertion(assertion_value) -> str:
    if pd.isna(assertion_value):
        return "Missing"

    text = str(assertion_value).strip()
    if not text:
        return "Missing"

    if " - " in text:
        label = text.split(" - ", 1)[1]
    else:
        label = text

    if " (" in label:
        label = label.split(" (", 1)[0]

    return label.strip() or "Missing"

def load_clingen_dosage_summary(csv_path: str) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        return pd.DataFrame(columns=[
            "Gene_Symbol_norm",
            "ClinGen_HI_Score",
            "ClinGen_HI_Label",
            "ClinGen_TS_Score",
            "ClinGen_TS_Label",
        ])

    try:
        raw = pd.read_csv(p, skiprows=3)
    except Exception:
        return pd.DataFrame(columns=[
            "Gene_Symbol_norm",
            "ClinGen_HI_Score",
            "ClinGen_HI_Label",
            "ClinGen_TS_Score",
            "ClinGen_TS_Label",
        ])

    required_cols = {
        "gene_symbol",
        "dosage_haploinsufficiency_assertion",
        "dosage_triplosensitivity_assertion",
    }
    if not required_cols.issubset(raw.columns):
        return pd.DataFrame(columns=[
            "Gene_Symbol_norm",
            "ClinGen_HI_Score",
            "ClinGen_HI_Label",
            "ClinGen_TS_Score",
            "ClinGen_TS_Label",
        ])

    raw = raw.copy()
    raw["Gene_Symbol_norm"] = raw["gene_symbol"].map(normalize_gene_symbol)
    raw = raw[raw["Gene_Symbol_norm"] != ""].copy()

    hi_df = raw[["Gene_Symbol_norm", "dosage_haploinsufficiency_assertion"]].copy()
    hi_df["ClinGen_HI_Score"] = hi_df["dosage_haploinsufficiency_assertion"].map(parse_clingen_score)
    hi_df["ClinGen_HI_Label"] = hi_df["dosage_haploinsufficiency_assertion"].map(simplify_clingen_assertion)
    hi_df = hi_df.sort_values(
        by=["Gene_Symbol_norm", "ClinGen_HI_Score"],
        ascending=[True, False],
        na_position="last"
    ).drop_duplicates(subset=["Gene_Symbol_norm"], keep="first")
    hi_df = hi_df[["Gene_Symbol_norm", "ClinGen_HI_Score", "ClinGen_HI_Label"]]

    ts_df = raw[["Gene_Symbol_norm", "dosage_triplosensitivity_assertion"]].copy()
    ts_df["ClinGen_TS_Score"] = ts_df["dosage_triplosensitivity_assertion"].map(parse_clingen_score)
    ts_df["ClinGen_TS_Label"] = ts_df["dosage_triplosensitivity_assertion"].map(simplify_clingen_assertion)
    ts_df = ts_df.sort_values(
        by=["Gene_Symbol_norm", "ClinGen_TS_Score"],
        ascending=[True, False],
        na_position="last"
    ).drop_duplicates(subset=["Gene_Symbol_norm"], keep="first")
    ts_df = ts_df[["Gene_Symbol_norm", "ClinGen_TS_Score", "ClinGen_TS_Label"]]

    merged = hi_df.merge(ts_df, how="outer", on="Gene_Symbol_norm")
    return merged


def normalize_gene_symbol(x):
    return str(x).strip().upper() if pd.notna(x) else ""


def normalize_truthy(x) -> bool:
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"true", "1", "yes", "y"}

def infer_is_pam_filtered(row: pd.Series) -> bool:
    raw_flag = row.get("Is_PAM_Filtered", pd.NA)
    if pd.notna(raw_flag):
        return normalize_truthy(raw_flag)

    status = str(row.get("PAM_Filter_Status", "")).strip().lower()
    if status:
        if "pre-pam" in status or "pre pam" in status:
            return False
        if "pam-filtered" in status or "pam filtered" in status:
            return True

    source = str(row.get("Variant_Source", "")).strip().lower()
    if source:
        if "pre-pam" in source or "pre pam" in source:
            return False
        if "pam-only" in source or "pam only" in source or "legacy" in source:
            return True

    return True

def normalize_pam_filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "PAM_Filter_Status" in df.columns:
        df["PAM_Filter_Status"] = df["PAM_Filter_Status"].astype(str)
    if "Variant_Source" in df.columns:
        df["Variant_Source"] = df["Variant_Source"].astype(str)

    df["Is_PAM_Filtered"] = df.apply(infer_is_pam_filtered, axis=1)

    if "PAM_Filter_Status" not in df.columns:
        df["PAM_Filter_Status"] = df["Is_PAM_Filtered"].map(
            lambda is_pam: "PAM-filtered targetable" if is_pam else "Pre-PAM expanded"
        )

    if "Variant_Source" not in df.columns:
        df["Variant_Source"] = df["Is_PAM_Filtered"].map(
            lambda is_pam: "legacy PAM-only table" if is_pam else "pre-PAM expanded table"
        )

    return df

def normalize_position_value(x):
    try:
        return int(float(x))
    except Exception:
        return pd.NA


def load_variant_master_table(
    preferred_xlsx_path: str,
    preferred_csv_dir: str,
    legacy_xlsx_path: str,
) -> pd.DataFrame:
    preferred_xlsx = Path(preferred_xlsx_path)
    preferred_csv_paths = [
        Path(preferred_csv_dir) / "Targetable_and_Het_Var.csv",
        *[
            Path(fallback_dir) / "Targetable_and_Het_Var.csv"
            for fallback_dir in VARIANT_CSV_FALLBACK_DIRS
        ],
    ]
    legacy_xlsx = Path(legacy_xlsx_path)

    if preferred_xlsx.exists():
        return pd.read_excel(preferred_xlsx, sheet_name="Targetable_&_Het_Var")
    for preferred_csv in preferred_csv_paths:
        if preferred_csv.exists():
            return pd.read_csv(preferred_csv)
    if legacy_xlsx.exists():
        return pd.read_excel(legacy_xlsx, sheet_name="Targetable_&_Het_Var")
    return pd.DataFrame()


def normalize_chromosome_value(x) -> str:
    if pd.isna(x):
        return ""
    text = str(x).strip()
    if text.lower().startswith("chr"):
        text = text[3:]
    return text.upper()


def join_unique_text(values) -> str:
    vals = pd.Series(values).dropna().astype(str)
    vals = vals[~vals.isin(["", "nan", "None"])]
    return "; ".join(sorted(vals.unique().tolist()))


def load_editor_assignment_summary(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    columns = [
        "_Editor_Gene_norm",
        "_Editor_Chrom_norm",
        "_Editor_Position_norm",
        "Editor_Categories",
        "Editor_Has_Assignment",
        "Editor_ABE",
        "Editor_CBE",
        "Editor_Both_ABE_CBE",
    ]
    if not path.exists():
        return pd.DataFrame(columns=columns)

    editor_df = pd.read_csv(path)
    required_cols = {"gene", "snp", "editor"}
    if not required_cols.issubset(editor_df.columns):
        return pd.DataFrame(columns=columns)

    chrom_col = "chrom" if "chrom" in editor_df.columns else None
    editor_df = editor_df.copy()
    editor_df["_Editor_Gene_norm"] = editor_df["gene"].map(normalize_gene_symbol)
    editor_df["_Editor_Position_norm"] = pd.to_numeric(editor_df["snp"], errors="coerce").astype("Int64")
    editor_df["_Editor_Chrom_norm"] = (
        editor_df[chrom_col].map(normalize_chromosome_value) if chrom_col else ""
    )
    editor_df["editor"] = editor_df["editor"].fillna("").astype(str).str.strip()

    summary = editor_df.groupby(
        ["_Editor_Gene_norm", "_Editor_Chrom_norm", "_Editor_Position_norm"],
        dropna=False,
    ).agg(
        Editor_Categories=("editor", join_unique_text),
        Editor_ABE=("editor", lambda s: bool(pd.Series(s).astype(str).eq("ABE").any())),
        Editor_CBE=("editor", lambda s: bool(pd.Series(s).astype(str).eq("CBE").any())),
        Editor_Both_ABE_CBE=("editor", lambda s: bool(pd.Series(s).astype(str).eq("both ABE and CBE").any())),
    ).reset_index()
    summary["Editor_Has_Assignment"] = True
    return summary[columns]


def add_editor_annotations_to_variants(variant_df: pd.DataFrame, editor_summary_df: pd.DataFrame) -> pd.DataFrame:
    out = variant_df.copy()
    if out.empty:
        return out

    if "Gene" not in out.columns or "Position" not in out.columns:
        out["Editor_Has_Assignment"] = False
        return out

    chrom_source = "Chromosome_norm" if "Chromosome_norm" in out.columns else "Chromosome"
    position_source = "Position_norm" if "Position_norm" in out.columns else "Position"

    out["_Editor_Gene_norm"] = out["Gene"].map(normalize_gene_symbol)
    out["_Editor_Chrom_norm"] = out[chrom_source].map(normalize_chromosome_value) if chrom_source in out.columns else ""
    out["_Editor_Position_norm"] = pd.to_numeric(out[position_source], errors="coerce").astype("Int64")

    if editor_summary_df.empty:
        out["Editor_Categories"] = ""
        out["Editor_Has_Assignment"] = False
        out["Editor_ABE"] = False
        out["Editor_CBE"] = False
        out["Editor_Both_ABE_CBE"] = False
    else:
        out = out.merge(
            editor_summary_df,
            how="left",
            on=["_Editor_Gene_norm", "_Editor_Chrom_norm", "_Editor_Position_norm"],
        )
        out["Editor_Categories"] = out["Editor_Categories"].fillna("")
        out["Editor_Has_Assignment"] = out["Editor_Has_Assignment"].fillna(False).astype(bool)
        for col in ["Editor_ABE", "Editor_CBE", "Editor_Both_ABE_CBE"]:
            out[col] = out[col].fillna(False).astype(bool)

    return out.drop(columns=["_Editor_Gene_norm", "_Editor_Chrom_norm", "_Editor_Position_norm"], errors="ignore")


def editor_category_options(variant_df: pd.DataFrame) -> list:
    if variant_df.empty or "Editor_Has_Assignment" not in variant_df.columns:
        return EDITOR_CATEGORY_OPTIONS.copy()

    options = []
    if "Editor_ABE" in variant_df.columns and variant_df["Editor_ABE"].fillna(False).astype(bool).any():
        options.append("ABE")
    if "Editor_CBE" in variant_df.columns and variant_df["Editor_CBE"].fillna(False).astype(bool).any():
        options.append("CBE")
    if "Editor_Both_ABE_CBE" in variant_df.columns and variant_df["Editor_Both_ABE_CBE"].fillna(False).astype(bool).any():
        options.append("both ABE and CBE")
    if (~variant_df["Editor_Has_Assignment"].fillna(False).astype(bool)).any():
        options.append("No editor assignment info")
    return options or EDITOR_CATEGORY_OPTIONS.copy()


def apply_editor_category_filter(variant_df: pd.DataFrame, selected_categories: list) -> pd.DataFrame:
    if variant_df.empty or not selected_categories or "Editor_Has_Assignment" not in variant_df.columns:
        return variant_df

    selected = set(map(str, selected_categories))
    keep = pd.Series(False, index=variant_df.index)
    if "ABE" in selected and "Editor_ABE" in variant_df.columns:
        keep = keep | variant_df["Editor_ABE"].fillna(False).astype(bool)
    if "CBE" in selected and "Editor_CBE" in variant_df.columns:
        keep = keep | variant_df["Editor_CBE"].fillna(False).astype(bool)
    if "both ABE and CBE" in selected and "Editor_Both_ABE_CBE" in variant_df.columns:
        keep = keep | variant_df["Editor_Both_ABE_CBE"].fillna(False).astype(bool)
    if "No editor assignment info" in selected:
        keep = keep | ~variant_df["Editor_Has_Assignment"].fillna(False).astype(bool)
    return variant_df[keep].copy()

def load_epi_promoter_gene_flags(csv_path: str) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        return pd.DataFrame(columns=["Gene_Symbol", "targetable_epi_silencing_100_200_prom"])

    epi_df = pd.read_csv(p)
    rename_map = {}
    if "gene" in epi_df.columns:
        rename_map["gene"] = "Gene_Symbol"
    if "targetable_epi_silencing_100/200_prom" in epi_df.columns:
        rename_map["targetable_epi_silencing_100/200_prom"] = "targetable_epi_silencing_100_200_prom"
    epi_df = epi_df.rename(columns=rename_map)

    if "Gene_Symbol" not in epi_df.columns:
        return pd.DataFrame(columns=["Gene_Symbol", "targetable_epi_silencing_100_200_prom"])

    if "targetable_epi_silencing_100_200_prom" not in epi_df.columns:
        epi_df["targetable_epi_silencing_100_200_prom"] = 0

    epi_df = epi_df[["Gene_Symbol", "targetable_epi_silencing_100_200_prom"]].copy()
    epi_df["Gene_Symbol"] = epi_df["Gene_Symbol"].astype(str).str.strip()
    epi_df["Gene_Symbol_norm"] = epi_df["Gene_Symbol"].map(normalize_gene_symbol)
    epi_df["targetable_epi_silencing_100_200_prom"] = pd.to_numeric(
        epi_df["targetable_epi_silencing_100_200_prom"], errors="coerce"
    ).fillna(0).astype(int)
    epi_df = epi_df.drop_duplicates(subset=["Gene_Symbol_norm"], keep="last")
    return epi_df

def load_epi_promoter_site_map(pkl_path: str) -> dict:
    p = Path(pkl_path)
    if not p.exists():
        return {}

    with open(p, "rb") as f:
        raw = pickle.load(f)

    site_map = {}
    if not isinstance(raw, dict):
        return site_map

    for gene, positions in raw.items():
        gene_norm = normalize_gene_symbol(gene)
        norm_positions = set()
        try:
            iterable = positions if positions is not None else []
        except Exception:
            iterable = []
        for pos in iterable:
            norm_pos = normalize_position_value(pos)
            if pd.notna(norm_pos):
                norm_positions.add(int(norm_pos))
        site_map[gene_norm] = norm_positions

    return site_map

def annotate_variant_df_with_epi_promoter(variant_df: pd.DataFrame, epi_gene_df: pd.DataFrame, epi_site_map: dict) -> pd.DataFrame:
    if variant_df.empty:
        out = variant_df.copy()
        out["targetable_epi_silencing_100_200_prom"] = pd.Series(dtype="int")
        out["targetable_epi_silencing_100_200_prom_variant"] = pd.Series(dtype="bool")
        return out

    out = variant_df.copy()
    out["Gene_norm"] = out["Gene"].map(normalize_gene_symbol)
    out["Position_norm"] = out["Position"].map(normalize_position_value)

    out["targetable_epi_silencing_100_200_prom_variant"] = out.apply(
        lambda row: (
            pd.notna(row["Position_norm"]) and
            int(row["Position_norm"]) in epi_site_map.get(row["Gene_norm"], set())
        ),
        axis=1
    )

    if epi_gene_df is not None and not epi_gene_df.empty:
        out = out.merge(
            epi_gene_df[["Gene_Symbol_norm", "targetable_epi_silencing_100_200_prom"]],
            how="left",
            left_on="Gene_norm",
            right_on="Gene_Symbol_norm"
        )
    else:
        out["targetable_epi_silencing_100_200_prom"] = 0

    out["targetable_epi_silencing_100_200_prom"] = pd.to_numeric(
        out["targetable_epi_silencing_100_200_prom"], errors="coerce"
    ).fillna(0).astype(int)

    drop_cols = [c for c in ["Gene_norm", "Position_norm", "Gene_Symbol_norm"] if c in out.columns]
    out = out.drop(columns=drop_cols)
    return out
def build_variant_gene_summary(variant_df: pd.DataFrame) -> pd.DataFrame:
    if variant_df.empty:
        return pd.DataFrame(columns=[
            "Gene_Symbol",
            "Unique_Variant_Sites",
            "Min_Population_Variant_Frequency",
            "Max_Population_Variant_Frequency",
            "Mean_Population_Variant_Frequency",
            "Has_EpiSilencing_100_200_Promoter_Gene",
            "EpiSilencing_100_200_Promoter_Variant_Row_Count",
            "EpiSilencing_100_200_Promoter_Unique_Site_Count",
            "Has_EpiSilencing_100_200_Promoter_Variant",
            "Any_PAM_Filtered",
            "Any_PrePAM_Only",
        ])

    tmp = variant_df.copy()
    tmp["Gene_Symbol"] = tmp["Gene"].astype(str)
    tmp["Is_PAM_Filtered"] = tmp.get("Is_PAM_Filtered", True).map(normalize_truthy)
    tmp["Variant_Site_Key"] = (
        tmp["Chromosome"].astype(str) + ":" +
        tmp["Position"].astype(str) + ":" +
        tmp["Ref_Allele"].astype(str) + ">" +
        tmp["Alt_Allele"].astype(str)
    )
    tmp["Position_norm"] = pd.to_numeric(tmp["Position"], errors="coerce").astype("Int64")

    summary = (
        tmp.groupby("Gene_Symbol", dropna=False)
        .agg(
            Unique_Variant_Sites=("Variant_Site_Key", "nunique"),
            Min_Population_Variant_Frequency=("Population_Variant_Frequency", "min"),
            Max_Population_Variant_Frequency=("Population_Variant_Frequency", "max"),
            Mean_Population_Variant_Frequency=("Population_Variant_Frequency", "mean"),
            Has_EpiSilencing_100_200_Promoter_Gene=("targetable_epi_silencing_100_200_prom", "max"),
            EpiSilencing_100_200_Promoter_Variant_Row_Count=(
                "targetable_epi_silencing_100_200_prom_variant", "sum"
            ),
            Any_PAM_Filtered=("Is_PAM_Filtered", "max"),
            Any_PrePAM_Only=("Is_PAM_Filtered", lambda x: (~pd.Series(x).fillna(False).astype(bool)).any()),
        )
        .reset_index()
    )

    promoter_sites = (
        tmp[tmp["targetable_epi_silencing_100_200_prom_variant"] == True]
        .groupby("Gene_Symbol")["Position_norm"]
        .nunique()
        .reset_index(name="EpiSilencing_100_200_Promoter_Unique_Site_Count")
    )

    summary = summary.merge(promoter_sites, on="Gene_Symbol", how="left")
    summary["EpiSilencing_100_200_Promoter_Unique_Site_Count"] = (
        summary["EpiSilencing_100_200_Promoter_Unique_Site_Count"].fillna(0).astype(int)
    )
    summary["EpiSilencing_100_200_Promoter_Variant_Row_Count"] = (
        summary["EpiSilencing_100_200_Promoter_Variant_Row_Count"].fillna(0).astype(int)
    )
    summary["Has_EpiSilencing_100_200_Promoter_Gene"] = (
        pd.to_numeric(summary["Has_EpiSilencing_100_200_Promoter_Gene"], errors="coerce")
        .fillna(0).astype(int)
    )
    summary["Has_EpiSilencing_100_200_Promoter_Variant"] = (
        summary["EpiSilencing_100_200_Promoter_Unique_Site_Count"] > 0
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

def build_pairwise_gene_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    tmp = df.copy()
    tmp["Gene_Symbol"] = tmp["Gene"].astype(str)
    tmp["Line_Strat"] = list(zip(tmp["Cell_Line"], tmp["Editing_Strategy"]))

    gene_mapping = (
        tmp.groupby("Gene_Symbol")["Line_Strat"]
        .apply(lambda x: sorted(set(x)))
        .reset_index(name="Line_Strats")
    )

    pairwise_records = []
    for _, row in gene_mapping.iterrows():
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

    pair_df = pd.DataFrame(pairwise_records).drop_duplicates()
    summary = pair_df.groupby(
        ["Cell_Line_A", "Strategy_A", "Cell_Line_B", "Strategy_B"]
    )["Gene_Symbol"].nunique().reset_index(name="Shared_Targetable_Genes")

    return summary.sort_values("Shared_Targetable_Genes", ascending=False)

def add_variant_site_key(df: pd.DataFrame, gene_col: str = "Gene") -> pd.DataFrame:
    out = df.copy()
    out["Variant_Site_Key"] = (
        out[gene_col].astype(str) + "_" +
        out["Chromosome"].astype(str) + ":" +
        out["Position"].astype(str) + ":" +
        out["Ref_Allele"].astype(str) + ">" +
        out["Alt_Allele"].astype(str)
    )
    return out

def build_variant_universe_context_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "Cell_Line",
            "Editing_Strategy",
            "Condition",
            "Genes_with_Filtered_Variants",
            "Unique_Variant_Sites",
            "Filtered_Variant_Rows",
        ])

    tmp = add_variant_site_key(df)
    summary = (
        tmp.groupby(["Cell_Line", "Editing_Strategy"], dropna=False)
        .agg(
            Genes_with_Filtered_Variants=("Gene", "nunique"),
            Unique_Variant_Sites=("Variant_Site_Key", "nunique"),
            Filtered_Variant_Rows=("Variant_Site_Key", "size"),
        )
        .reset_index()
    )
    summary["Condition"] = summary.apply(
        lambda row: condition_label(row["Cell_Line"], row["Editing_Strategy"]),
        axis=1
    )
    return summary.sort_values(
        by=["Unique_Variant_Sites", "Genes_with_Filtered_Variants", "Condition"],
        ascending=[False, False, True]
    )

def build_variant_universe_line_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "Cell_Line",
            "Genes_with_Filtered_Variants",
            "Unique_Variant_Sites",
            "Editing_Strategies_Present",
        ])

    tmp = add_variant_site_key(df)
    summary = (
        tmp.groupby("Cell_Line", dropna=False)
        .agg(
            Genes_with_Filtered_Variants=("Gene", "nunique"),
            Unique_Variant_Sites=("Variant_Site_Key", "nunique"),
            Editing_Strategies_Present=("Editing_Strategy", "nunique"),
        )
        .reset_index()
    )
    summary["Cell_Line_Display"] = summary["Cell_Line"].map(prettify_cell_line)
    return summary.sort_values(
        by=["Unique_Variant_Sites", "Genes_with_Filtered_Variants", "Cell_Line_Display"],
        ascending=[False, False, True]
    )

def build_variant_universe_strategy_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "Editing_Strategy",
            "Genes_with_Filtered_Variants",
            "Unique_Variant_Sites",
            "Cell_Lines_Present",
        ])

    tmp = add_variant_site_key(df)
    summary = (
        tmp.groupby("Editing_Strategy", dropna=False)
        .agg(
            Genes_with_Filtered_Variants=("Gene", "nunique"),
            Unique_Variant_Sites=("Variant_Site_Key", "nunique"),
            Cell_Lines_Present=("Cell_Line", "nunique"),
        )
        .reset_index()
    )
    summary["Editing_Strategy_Display"] = summary["Editing_Strategy"].map(prettify_strategy)
    return summary.sort_values(
        by=["Unique_Variant_Sites", "Genes_with_Filtered_Variants", "Editing_Strategy_Display"],
        ascending=[False, False, True]
    )

def build_condition_position_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "Condition",
            "Total_Unique_Variant_Sites",
            "Private_Unique_Variant_Sites",
        ])

    tmp = add_variant_site_key(df)
    tmp["Condition"] = tmp.apply(
        lambda row: condition_label(row["Cell_Line"], row["Editing_Strategy"]),
        axis=1
    )

    site_condition_counts = (
        tmp.groupby("Variant_Site_Key")["Condition"]
        .nunique()
        .reset_index(name="Condition_Count")
    )
    tmp = tmp.merge(site_condition_counts, how="left", on="Variant_Site_Key")

    total_summary = (
        tmp.groupby("Condition", dropna=False)
        .agg(Total_Unique_Variant_Sites=("Variant_Site_Key", "nunique"))
        .reset_index()
    )

    private_summary = (
        tmp[tmp["Condition_Count"] == 1]
        .groupby("Condition", dropna=False)
        .agg(Private_Unique_Variant_Sites=("Variant_Site_Key", "nunique"))
        .reset_index()
    )

    summary = total_summary.merge(private_summary, how="left", on="Condition")
    summary["Private_Unique_Variant_Sites"] = (
        summary["Private_Unique_Variant_Sites"].fillna(0).astype(int)
    )
    return summary.sort_values(
        by=["Total_Unique_Variant_Sites", "Private_Unique_Variant_Sites", "Condition"],
        ascending=[False, False, True]
    )

def build_shared_position_only_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.iloc[0:0].copy()

    tmp = add_variant_site_key(df)
    tmp["Condition"] = tmp.apply(
        lambda row: condition_label(row["Cell_Line"], row["Editing_Strategy"]),
        axis=1
    )
    shared_sites = (
        tmp.groupby("Variant_Site_Key")["Condition"]
        .nunique()
        .reset_index(name="Condition_Count")
    )
    shared_sites = shared_sites[shared_sites["Condition_Count"] >= 2]["Variant_Site_Key"]
    return tmp[tmp["Variant_Site_Key"].isin(shared_sites)].drop(columns=["Condition"]).copy()

def build_context_matrix(summary_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    tmp = summary_df.copy()
    tmp["Cell_Line_Display"] = tmp["Cell_Line"].map(prettify_cell_line)
    tmp["Editing_Strategy_Display"] = tmp["Editing_Strategy"].map(prettify_strategy)
    matrix = tmp.pivot_table(
        index="Cell_Line_Display",
        columns="Editing_Strategy_Display",
        values=value_col,
        aggfunc="sum",
        fill_value=0
    )
    return matrix.reindex(sorted(matrix.index), axis=0).reindex(sorted(matrix.columns), axis=1)

def render_summary_heatmap(matrix: pd.DataFrame, title: str, color_title: str, color_scale: str):
    if matrix.empty:
        st.info(f"No data available for {title.lower()}.")
        return

    fig = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale=color_scale,
        title=title,
        labels=dict(x="Editing strategy", y="Cell line", color=color_title)
    )
    fig.update_layout(height=max(420, 80 * len(matrix.index) + 180))
    fig.update_xaxes(tickangle=-25, automargin=True)
    fig.update_yaxes(automargin=True)
    st.plotly_chart(fig, width="stretch")

def build_cell_line_uniqueness_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "Cell_Line",
            "Total_Unique_Common_Variant_Sites",
            "Private_to_This_Cell_Line",
            "Shared_with_Other_Cell_Lines",
            "Genes_with_Private_Sites",
        ])

    tmp = add_variant_site_key(df)
    site_lines = (
        tmp.groupby("Variant_Site_Key")["Cell_Line"]
        .nunique()
        .reset_index(name="Cell_Line_Count")
    )
    tmp = tmp.merge(site_lines, how="left", on="Variant_Site_Key")

    total_summary = (
        tmp.groupby("Cell_Line", dropna=False)
        .agg(
            Total_Unique_Common_Variant_Sites=("Variant_Site_Key", "nunique"),
        )
        .reset_index()
    )

    private_tmp = tmp[tmp["Cell_Line_Count"] == 1].copy()
    private_summary = (
        private_tmp.groupby("Cell_Line", dropna=False)
        .agg(
            Private_to_This_Cell_Line=("Variant_Site_Key", "nunique"),
            Genes_with_Private_Sites=("Gene", "nunique"),
        )
        .reset_index()
    )

    summary = total_summary.merge(private_summary, how="left", on="Cell_Line")
    summary["Private_to_This_Cell_Line"] = summary["Private_to_This_Cell_Line"].fillna(0).astype(int)
    summary["Genes_with_Private_Sites"] = summary["Genes_with_Private_Sites"].fillna(0).astype(int)
    summary["Shared_with_Other_Cell_Lines"] = (
        summary["Total_Unique_Common_Variant_Sites"] - summary["Private_to_This_Cell_Line"]
    )
    summary["Cell_Line_Display"] = summary["Cell_Line"].map(prettify_cell_line)
    return summary.sort_values(
        by=["Private_to_This_Cell_Line", "Total_Unique_Common_Variant_Sites", "Cell_Line_Display"],
        ascending=[False, False, True]
    )

def build_strategy_overlap_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "Strategy_Combination",
            "Unique_CellLine_Sites",
            "Unique_Genes",
            "Cell_Lines_With_This_Pattern",
        ])

    tmp = add_variant_site_key(df)
    strategy_sets = (
        tmp.groupby(["Cell_Line", "Variant_Site_Key"], dropna=False)
        .agg(
            Gene=("Gene", "first"),
            Strategy_Set=("Editing_Strategy", lambda x: tuple(sorted(set(map(str, x)))))
        )
        .reset_index()
    )
    strategy_sets["Strategy_Combination"] = strategy_sets["Strategy_Set"].map(
        lambda combo: " + ".join(prettify_strategy(x) for x in combo)
    )
    summary = (
        strategy_sets.groupby(["Strategy_Set", "Strategy_Combination"], dropna=False)
        .agg(
            Unique_CellLine_Sites=("Variant_Site_Key", "nunique"),
            Unique_Genes=("Gene", "nunique"),
            Cell_Lines_With_This_Pattern=("Cell_Line", "nunique"),
        )
        .reset_index()
        .drop(columns=["Strategy_Set"])
    )
    return summary.sort_values(
        by=["Unique_CellLine_Sites", "Unique_Genes", "Strategy_Combination"],
        ascending=[False, False, True]
    )

CLINGEN_CATEGORY_OPTIONS = [
    "Sensitivity unlikely",
    "Strong sensitivity",
    "Emerging sensitivity",
    "Little evidence",
    "No evidence",
    "Autosomal recessive",
    "Missing annotation",
]

def clingen_score_to_category(score_value) -> str:
    if pd.isna(score_value):
        return "Missing annotation"

    try:
        score = int(score_value)
    except Exception:
        return "Missing annotation"

    if score == 40:
        return "Sensitivity unlikely"
    if score == 3:
        return "Strong sensitivity"
    if score == 2:
        return "Emerging sensitivity"
    if score == 1:
        return "Little evidence"
    if score == 0:
        return "No evidence"
    if score == 30:
        return "Autosomal recessive"
    return "Missing annotation"

def clingen_score_filter_mask(df: pd.DataFrame, score_col: str, selected_categories: list) -> pd.Series:
    if score_col not in df.columns:
        return pd.Series(True, index=df.index)

    if not selected_categories or set(selected_categories) == set(CLINGEN_CATEGORY_OPTIONS):
        return pd.Series(True, index=df.index)

    categories = pd.to_numeric(df[score_col], errors="coerce").map(clingen_score_to_category)
    return categories.isin(selected_categories)

def coerce_variant_df_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Position", "Population_Variant_Frequency"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["Cell_Line", "Editing_Strategy", "Gene", "Chromosome", "Ref_Allele", "Alt_Allele"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return normalize_pam_filter_columns(df)

CELL_LINE_LABELS = {
    "KOLF2-ARID2-A02": "KOLF2",
    "WTB_variants_PASS": "WTB",
    "WTC_variants_PASS": "WTC",
    "cN8-hNIL": "WTD",
    "p47": "p47"
}

def prettify_cell_line(x: str) -> str:
    return CELL_LINE_LABELS.get(x, x)

CRISPROFF_HIGH_CONF_OPTION = "CRISPRoff high confidence"

STRATEGY_LABELS = {
    CRISPROFF_HIGH_CONF_OPTION: "CRISPRoff HC",
}

def prettify_strategy(x: str) -> str:
    return STRATEGY_LABELS.get(str(x), str(x))

def condition_label(cell_line: str, strategy: str) -> str:
    return f"{prettify_cell_line(str(cell_line))} ({prettify_strategy(str(strategy))})"

def condition_order_from_variants(variant_df: pd.DataFrame) -> list:
    required_cols = {"Cell_Line", "Editing_Strategy"}
    if variant_df.empty or not required_cols.issubset(variant_df.columns):
        return []

    condition_df = variant_df[["Cell_Line", "Editing_Strategy"]].drop_duplicates()
    labels = [
        condition_label(row.Cell_Line, row.Editing_Strategy)
        for row in condition_df.itertuples(index=False)
    ]
    return sorted(labels)

def symmetric_pairwise_matrix(pairwise_df: pd.DataFrame, value_col: str, condition_order: list) -> pd.DataFrame:
    if not condition_order:
        return pd.DataFrame()

    matrix = pd.DataFrame(0, index=condition_order, columns=condition_order)
    if pairwise_df.empty:
        return matrix

    tmp = pairwise_df.copy()
    tmp["Condition_A"] = tmp.apply(
        lambda row: condition_label(row["Cell_Line_A"], row["Strategy_A"]),
        axis=1
    )
    tmp["Condition_B"] = tmp.apply(
        lambda row: condition_label(row["Cell_Line_B"], row["Strategy_B"]),
        axis=1
    )

    observed = tmp.pivot_table(
        index="Condition_A",
        columns="Condition_B",
        values=value_col,
        aggfunc="sum"
    ).fillna(0)
    observed = observed.reindex(index=condition_order, columns=condition_order, fill_value=0)
    matrix = observed.add(observed.T, fill_value=0)

    for condition in condition_order:
        matrix.loc[condition, condition] = 0

    return matrix.astype(int)

def render_pairwise_heatmap(matrix: pd.DataFrame, title: str, color_title: str, color_scale: str):
    if matrix.empty:
        st.info(f"No data available for {title.lower()}.")
        return

    display_matrix = matrix.astype(float).copy()
    for row_idx in range(display_matrix.shape[0]):
        for col_idx in range(display_matrix.shape[1]):
            if col_idx >= row_idx:
                display_matrix.iat[row_idx, col_idx] = None

    n_conditions = max(matrix.shape)
    fig = px.imshow(
        display_matrix,
        text_auto=True,
        color_continuous_scale=color_scale,
        title=title,
        labels=dict(x="Condition B", y="Condition A", color=color_title)
    )
    fig.update_traces(textfont_size=10)
    fig.update_layout(
        height=max(650, 46 * n_conditions + 320),
        margin=dict(l=300, r=150, t=90, b=260),
        font=dict(size=12),
        coloraxis_colorbar=dict(title=color_title),
    )
    fig.update_xaxes(tickangle=-35, automargin=True, tickfont=dict(size=12))
    fig.update_yaxes(automargin=True, tickfont=dict(size=12))
    st.plotly_chart(fig, width="stretch")

def variant_strategy_options(variant_df: pd.DataFrame) -> list:
    if variant_df.empty or "Editing_Strategy" not in variant_df.columns:
        return []
    options = sorted(variant_df["Editing_Strategy"].dropna().astype(str).unique().tolist())
    if "targetable_epi_silencing_100_200_prom_variant" in variant_df.columns:
        options.append(CRISPROFF_HIGH_CONF_OPTION)
    return options

def apply_variant_strategy_filter(variant_df: pd.DataFrame, selected_options: list) -> pd.DataFrame:
    if variant_df.empty or not selected_options:
        return variant_df.iloc[0:0].copy()

    selected_options = [str(x) for x in selected_options]
    regular_options = [x for x in selected_options if x != CRISPROFF_HIGH_CONF_OPTION]
    keep = pd.Series(False, index=variant_df.index)

    if regular_options and "Editing_Strategy" in variant_df.columns:
        keep = keep | variant_df["Editing_Strategy"].astype(str).isin(regular_options)

    if (
        CRISPROFF_HIGH_CONF_OPTION in selected_options and
        "targetable_epi_silencing_100_200_prom_variant" in variant_df.columns
    ):
        keep = keep | variant_df["targetable_epi_silencing_100_200_prom_variant"].fillna(False).astype(bool)

    return variant_df[keep].copy()

def genes_matching_all_variant_strategy_options(variant_df: pd.DataFrame, selected_options: list) -> set:
    if variant_df.empty or not selected_options:
        return set()

    selected_options = [str(x) for x in selected_options]
    gene_to_options = {}

    for _, row in variant_df.iterrows():
        gene = str(row.get("Gene", ""))
        if not gene:
            continue

        seen = gene_to_options.setdefault(gene, set())
        strategy = str(row.get("Editing_Strategy", ""))
        if strategy in selected_options:
            seen.add(strategy)

        if (
            CRISPROFF_HIGH_CONF_OPTION in selected_options and
            bool(row.get("targetable_epi_silencing_100_200_prom_variant", False))
        ):
            seen.add(CRISPROFF_HIGH_CONF_OPTION)

    required = set(selected_options)
    return {gene for gene, seen in gene_to_options.items() if required.issubset(seen)}

def expand_variant_rows_for_selected_strategy_options(variant_df: pd.DataFrame, selected_options: list) -> pd.DataFrame:
    if variant_df.empty or not selected_options:
        return variant_df.iloc[0:0].copy()

    selected_options = [str(x) for x in selected_options]
    regular_options = [x for x in selected_options if x != CRISPROFF_HIGH_CONF_OPTION]
    pieces = []

    if regular_options and "Editing_Strategy" in variant_df.columns:
        pieces.append(variant_df[variant_df["Editing_Strategy"].astype(str).isin(regular_options)].copy())

    if (
        CRISPROFF_HIGH_CONF_OPTION in selected_options and
        "targetable_epi_silencing_100_200_prom_variant" in variant_df.columns
    ):
        high_conf = variant_df[
            variant_df["targetable_epi_silencing_100_200_prom_variant"].fillna(False).astype(bool)
        ].copy()
        high_conf["Editing_Strategy"] = CRISPROFF_HIGH_CONF_OPTION
        pieces.append(high_conf)

    if not pieces:
        return variant_df.iloc[0:0].copy()

    return pd.concat(pieces, ignore_index=True).drop_duplicates()


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
    width = "stretch" if use_container_width else "content"
    if df_in is None or df_in.empty:
        st.dataframe(df_in, width=width)
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

    st.dataframe(styler, width=width)

def render_update_notice():
    st.markdown(
        """
        <div style="
            border: 1px solid rgba(47, 111, 115, 0.35);
            border-left: 6px solid #2f6f73;
            border-radius: 10px;
            padding: 0.9rem 1rem;
            margin: 0.2rem 0 1rem 0;
            background: color-mix(in srgb, #2f6f73 10%, transparent);
        ">
            <div style="font-weight: 800; margin-bottom: 0.35rem;">2026/06/04 update</div>
            <div>- Added cell-line intersection venn diagram.</div>
            <div>- This message will disappear if you click <strong>OK</strong>↓.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    dismiss_cols = st.columns([1, 6])
    with dismiss_cols[0]:
        if st.button("OK", key="dismiss_update_notice", width="stretch"):
            st.session_state["hide_update_notice"] = True
            st.rerun()

    
# ------------------------------------------------
# Data Loader
# ------------------------------------------------
@st.cache_data
def load_data():
    cardiac = pd.read_csv(CARDIAC_CSV)
    neuron = pd.read_csv(NEURON_CSV)
    hepatocyte = load_ipsc_hepatocyte_metrics()
    annot = pd.read_excel(ANNOT_XLSX, sheet_name=ANNOT_SHEET)
    clingen_df = load_clingen_dosage_summary(CLINGEN_DOSAGE_CSV)

    cardiac = add_dataset_label(cardiac, "Cardiac")
    neuron = add_dataset_label(neuron, "Neuron")
    metrics_parts = [cardiac, neuron]
    if not hepatocyte.empty:
        metrics_parts.append(hepatocyte)
    metrics_df = pd.concat(metrics_parts, ignore_index=True)
    metrics_df = add_detectability_context(metrics_df)

    left_key, right_key = pick_annotation_key(metrics_df, annot)
    merged = metrics_df.merge(annot, how="left", left_on=left_key, right_on=right_key)

    if not clingen_df.empty:
        merged["Gene_Symbol_norm"] = merged["Gene_Symbol"].map(normalize_gene_symbol)
        merged = merged.merge(clingen_df, how="left", on="Gene_Symbol_norm")
    else:
        merged["ClinGen_HI_Score"] = pd.NA
        merged["ClinGen_HI_Label"] = "Missing"
        merged["ClinGen_TS_Score"] = pd.NA
        merged["ClinGen_TS_Label"] = "Missing"

    for col in ["dominant_mutation_count", "Citation_Count", "s_het"]:
        if col in merged.columns:
            merged[col] = safe_numeric(merged[col], fill_value=0)

    for col in ["ClinGen_HI_Score", "ClinGen_TS_Score"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    for col in ["ClinGen_HI_Label", "ClinGen_TS_Label"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna("Missing").astype(str)

    if "ClinGen_HI_Score" in merged.columns:
        merged["ClinGen_HI_Category"] = merged["ClinGen_HI_Score"].map(clingen_score_to_category)
    if "ClinGen_TS_Score" in merged.columns:
        merged["ClinGen_TS_Category"] = merged["ClinGen_TS_Score"].map(clingen_score_to_category)

    for col in ["HPO_Cardiovascular", "HPO_Nervous", "HPO_Metabolism", "HPO_Musculoskeletal"]:
        if col in merged.columns:
            merged[col] = normalize_bool_col(merged, col)

    epi_gene_df = load_epi_promoter_gene_flags(EPI_PROM_GENE_CSV)
    if not epi_gene_df.empty:
        merged = merged.merge(
            epi_gene_df[["Gene_Symbol_norm", "targetable_epi_silencing_100_200_prom"]],
            how="left",
            on="Gene_Symbol_norm"
        )
        merged = merged.drop(columns=["Gene_Symbol_norm"])
    else:
        merged["targetable_epi_silencing_100_200_prom"] = 0
    merged["targetable_epi_silencing_100_200_prom"] = pd.to_numeric(
        merged["targetable_epi_silencing_100_200_prom"], errors="coerce"
    ).fillna(0).astype(int)

    variant_df = load_variant_master_table(
        preferred_xlsx_path=VARIANT_FILE,
        preferred_csv_dir=VARIANT_CSV_DIR,
        legacy_xlsx_path=LEGACY_VARIANT_FILE,
    )
    if not variant_df.empty:
        variant_df = coerce_variant_df_types(variant_df)
        if "PAM_Filter_Status" not in variant_df.columns:
            variant_df["PAM_Filter_Status"] = "PAM-filtered targetable"
        if "Is_PAM_Filtered" not in variant_df.columns:
            variant_df["Is_PAM_Filtered"] = True
        if "Variant_Source" not in variant_df.columns:
            variant_df["Variant_Source"] = "legacy PAM-only table"
        epi_site_map = load_epi_promoter_site_map(EPI_PROM_SITE_PKL)
        variant_df = annotate_variant_df_with_epi_promoter(variant_df, epi_gene_df, epi_site_map)
        editor_summary_df = load_editor_assignment_summary(EDITOR_ASSIGNMENT_CSV)
        variant_df = add_editor_annotations_to_variants(variant_df, editor_summary_df)

    summary_df = load_optional_summary(SUMMARY_TSV)
    return merged, variant_df, summary_df

df, variant_master_df, summary_counts_df = load_data()

# ------------------------------------------------
# Dashboard UI - Main Header & Info Manual
# ------------------------------------------------
st.title("D&D Gene Detectability Explorer")

if "hide_update_notice" not in st.session_state:
    st.session_state["hide_update_notice"] = False

if not st.session_state["hide_update_notice"]:
    render_update_notice()

TUTORIAL_STEPS = [
    {
        "title": "1. Start here",
        "sidebar": "Use the sidebar from top to bottom. Begin broad, then tighten one filter at a time.",
        "body": """
        This app filters Dominant and Dispensible (D&D) candidate genes by single-cell detectability, gene-level annotations, phenotype evidence, and targetable heterozygous variants. The left sidebar is the control panel. The main page shows the genes and variant records that survive the current settings.
        """,
    },
    {
        "title": "2. Expression on or off",
        "sidebar": "Toggle whether single-cell detectability should be part of the gene filter.",
        "body": """
        **Include Expression Data & Filters** decides whether expression/detectability is required.

        - On: a gene must pass the expression thresholds in the selected cell line contexts.
        - Off: expression is ignored, and genes are filtered by annotation, phenotype, promoter-window status, and variant evidence.
        """,
    },
    {
        "title": "3. QC level",
        "sidebar": "Choose which expression table columns are used.",
        "body": """
        The Quality Check level selects one matched set of expression columns.

        - **Raw** uses all cells/barcodes.
        - **LooseQC** uses a moderate quality filter.
        - **StrongQC** uses a stricter quality filter, including mitochondrial percentage.

        If you are exploring, start with **LooseQC**. Use **StrongQC** when you want a stricter confidence check.
        """,
    },
    {
        "title": "4. Expression thresholds",
        "sidebar": "Set how detectable a gene must be.",
        "body": """
        These sliders define what counts as detectable enough.

        - **Min Detection Rate (%)**: percent of cells in which the gene has nonzero counts.
        - **Min Aggregated CPM**: total raw counts for the gene divided by total sample counts, scaled per million. This captures overall abundance across the sample.
        - **Min Mean Expr Detected**: normalized expression among only cells where the gene is detected. This helps distinguish weak background from stronger signal in expressing cells.
        """,
    },
    {
        "title": "5. Cell Type And Condition",
        "sidebar": "Pick the biological contexts where detectability should be evaluated.",
        "body": """
        **Cell Type** chooses the broad expression source, such as Cardiac, Neuron, or Liver. **Condition** then chooses the specific cell line, stage, or study context within those sources.

        Use **OR** if a gene only needs to pass in any selected context. Use **AND** if a gene must pass in every selected context.
        """,
    },
    {
        "title": "6. Literature, genetics, and HPO",
        "sidebar": "Prioritize genes with genetic constraint, mutation, citation, or phenotype support.",
        "body": """
        These filters act at the gene level.

        - **s_het**: intolerance to heterozygous loss-of-function variation; higher means less tolerant.
        - **Dominant Mutation Count**: known dominant mutation evidence.
        - **Citation Count**: literature support.
        - **Clinical Phenotypes (HPO)**: keep genes associated with selected phenotype systems.
        """,
    },
    {
        "title": "7. CRISPRoff high confidence",
        "sidebar": "Use the Editing Strategy filter and choose CRISPRoff high confidence.",
        "body": """
        There is no separate promoter-window dropdown now. Use **Editing Strategy -> CRISPRoff high confidence** to keep variant rows that overlap the promoter-window CRISPRoff target-site map in the -100/+200 bp TSS window.

        This behaves like the other editing strategy options, so you can combine it with base editing, indel, or other CRISPRoff categories using the OR/AND strategy logic.
        """,
    },
    {
        "title": "8. Variant filters",
        "sidebar": "Filter targetable heterozygous variants by line, editing strategy, and population frequency.",
        "body": """
        Variant filters operate on rows in the variant table.

        - **Variant Cell Line**: where the heterozygous targetable variant was observed.
        - **Editing Strategy**: targetability method. The list also includes **CRISPRoff high confidence**, which means promoter-window CRISPRoff target-site overlap.
        - **Min Population Variant Frequency**: keeps variants with population frequency at or above the threshold, while retaining missing values.
        """,
    },
    {
        "title": "9. Read the outputs",
        "sidebar": "Use the tabs to move from summary to overlap to detailed tables.",
        "body": """
        Read the app from left to right by tab.

        - **Overview**: broad counts and heatmaps for what remains after filtering.
        - **Overlap**: pairwise sharing and optional UpSet intersections.
        - **Tables**: detailed gene, variant, and expression tables.
        - **Downloads**: export the filtered results.
        """,
    },
]

if "tutorial_step" not in st.session_state:
    st.session_state.tutorial_step = 0

show_tutorial = st.checkbox(
    "Show guided tutorial",
    value=False,
    help="Turn this on when someone wants a walkthrough of the app. It is hidden by default to keep the page compact."
)

if show_tutorial:
    with st.container(border=True):
        st.markdown("### Guided tutorial")
        t_left, t_right = st.columns([1, 2])
        with t_left:
            selected_tutorial_title = st.selectbox(
                "Tutorial step",
                [step["title"] for step in TUTORIAL_STEPS],
                index=st.session_state.tutorial_step,
                help="Pick a step to learn what that part of the dashboard controls. Use Previous and Next to walk through the app in order."
            )
            st.session_state.tutorial_step = [step["title"] for step in TUTORIAL_STEPS].index(selected_tutorial_title)
            prev_col, next_col = st.columns(2)
            with prev_col:
                if st.button("Previous", disabled=st.session_state.tutorial_step == 0, help="Go to the previous tutorial step."):
                    st.session_state.tutorial_step -= 1
                    st.rerun()
            with next_col:
                if st.button("Next", disabled=st.session_state.tutorial_step == len(TUTORIAL_STEPS) - 1, help="Go to the next tutorial step."):
                    st.session_state.tutorial_step += 1
                    st.rerun()
        with t_right:
            current_step = TUTORIAL_STEPS[st.session_state.tutorial_step]
            st.markdown(f"#### {current_step['title']}")
            st.markdown(current_step["body"])

def tutorial_active(*step_indices):
    return show_tutorial and st.session_state.tutorial_step in step_indices

def sidebar_section_title(title, *step_indices):
    if tutorial_active(*step_indices):
        st.sidebar.markdown(
            f'<div class="tutorial-filter-highlight">▶ {title}</div>',
            unsafe_allow_html=True
        )
    else:
        st.sidebar.markdown(
            f'<div class="sidebar-section-title">{title}</div>',
            unsafe_allow_html=True
        )

def tutorial_label(label, *step_indices):
    if tutorial_active(*step_indices):
        return f"**▶ {label}**"
    return label

# ------------------------------------------------
# Sidebar Filters
# ------------------------------------------------




sidebar_section_title("Gene Search")
gene_search_query = st.sidebar.text_input(
    "Search",
    value="",
    help="Gene symbol search(eg. nefl or NEFL). When provided, the app focuses all outputs on that gene."
).strip()
gene_search_norm = normalize_gene_symbol(gene_search_query)
gene_search_matches = []
gene_search_active = bool(gene_search_norm)
if gene_search_active:
    gene_symbol_lookup = df[["Gene_Symbol"]].dropna().drop_duplicates().copy()
    gene_symbol_lookup["Gene_Symbol_norm"] = gene_symbol_lookup["Gene_Symbol"].map(normalize_gene_symbol)
    gene_search_matches = sorted(
        gene_symbol_lookup.loc[
            gene_symbol_lookup["Gene_Symbol_norm"].eq(gene_search_norm),
            "Gene_Symbol"
        ].astype(str).unique().tolist()
    )
    if gene_search_matches:
        st.sidebar.success(f"Focused on {', '.join(gene_search_matches)}")
        st.sidebar.caption("Gene search shows all available expression and variant records for this gene.")
    else:
        st.sidebar.warning(f"No exact gene symbol match found for '{gene_search_query}'.")

st.sidebar.title("Apply Filters")
detectability_cell_types = sorted(df["Detectability_Cell_Type"].dropna().unique().tolist())

sidebar_section_title("Expression Toggle", 1)
include_expression = st.sidebar.checkbox(
    tutorial_label("Include Expression Data & Filters", 1),
    value=True,
    help="Uncheck this to explore genes strictly based on Annotations and Variants, ignoring Single-Cell Detectability."
)


if include_expression:
    sidebar_section_title("Expression Thresholds", 2, 3)

    qc_level = st.sidebar.selectbox(
        tutorial_label("QC Level", 2),
        QC_LEVELS,
        index=1,
        help="Choose which Quality-Check-specific expression metrics to use. Liver(iPSC-Hepatocyte) maps Raw/LooseQC/StrongQC to QC_Min_Genes 0/200/500."
    )

    expression_df = df

    det_col = qc_col(qc_level, "Detection_Rate_%")
    mean_all_col = qc_col(qc_level, "Mean_Expr_All")
    mean_detected_col = qc_col(qc_level, "Mean_Expr_Detected")
    agg_col = qc_col(qc_level, "Aggregated_CPM")

    required_cols = [det_col, mean_all_col, mean_detected_col, agg_col]
    missing_cols = [c for c in required_cols if c not in expression_df.columns]
    if missing_cols:
        st.error(f"Missing expected QC-specific columns: {missing_cols}")
        st.stop()

    expression_filter_mode = st.sidebar.radio(
        tutorial_label("Expression Filter Mode", 3),
        ["Manual thresholds", "Top N ranked genes"],
        help="Manual thresholds use the sliders below. Top N ranks genes within the selected expression conditions."
    )

    min_det = 0.0
    min_cpm = 0.0
    min_mean_detected = 0.0
    top_n_genes = 50
    expression_rank_basis = "Average percentile from all"

    if expression_filter_mode == "Manual thresholds":
        min_det = st.sidebar.slider(
            tutorial_label(f"Min Detection Rate (%) [{qc_level}]", 3),
            0.0, 100.0, 0.0, step=1.0,
            help="Minimum percent of cells in the selected cell line context where this gene has nonzero expression."
        )

        max_agg = float(pd.to_numeric(expression_df[agg_col], errors="coerce").fillna(0).max())
        min_cpm = st.sidebar.slider(
            tutorial_label("Min Aggregated CPM", 3),
            0.0,
            max(0.0, max_agg),
            0.0,
            step=max(1.0, max_agg / 100 if max_agg > 100 else 1.0),
            help="Minimum aggregate gene expression across all selected cells, scaled as counts per million total counts."
        )

        max_mean_detected = float(pd.to_numeric(expression_df[mean_detected_col], errors="coerce").fillna(0).max())
        min_mean_detected = st.sidebar.slider(
            tutorial_label("Min Mean Expr Detected", 3),
            0.0,
            max(0.0, max_mean_detected),
            0.0,
            step=max(0.01, max_mean_detected / 100 if max_mean_detected > 1 else 0.01),
            help="Minimum normalized expression among only the cells where this gene is detected."
        )
    else:
        max_rank_genes = int(expression_df["Gene_Symbol"].dropna().nunique())
        top_n_genes = st.sidebar.number_input(
            tutorial_label("Top N Genes", 3),
            min_value=1,
            max_value=max(1, max_rank_genes),
            value=min(50, max(1, max_rank_genes)),
            step=10,
            help="Keep the top N genes after ranking expression in the selected conditions."
        )
        expression_rank_basis = st.sidebar.selectbox(
            tutorial_label("Rank Genes By", 3),
            [
                "Average percentile from all",
                "Detection rate",
                "Aggregated CPM",
                "Mean expression among detected cells",
            ],
            help="You can select to sort genes by one metric only or from the average percentile ranks from the three expression layers."
        )

    sidebar_section_title("Expression Conditions", 4)

    selected_cell_types = st.sidebar.multiselect(
        tutorial_label("Cell Type", 4),
        detectability_cell_types,
        default=detectability_cell_types,
        help="Choose broad expression sources before selecting specific conditions."
    )

    condition_options_df = expression_df[
        expression_df["Detectability_Cell_Type"].isin(selected_cell_types)
    ][["Detectability_Label", "Detectability_Context"]].drop_duplicates()
    condition_options = sorted(condition_options_df["Detectability_Label"].dropna().tolist())
    label_to_context = dict(
        zip(condition_options_df["Detectability_Label"], condition_options_df["Detectability_Context"])
    )
    selected_condition_labels = st.sidebar.multiselect(
        tutorial_label("Condition", 4),
        condition_options,
        default=condition_options,
        help="Choose the specific cell line, stage, or study context within the selected cell types."
    )
    selected_contexts = [label_to_context[label] for label in selected_condition_labels if label in label_to_context]

    expr_logic = st.sidebar.radio(
        tutorial_label("Match Logic for Conditions:", 4),
        ["OR (Passes thresholds in ANY selected condition)", "AND (Passes thresholds in ALL selected conditions)"],
        help="OR is permissive: a gene can pass in one selected condition. AND is strict: a gene must pass in every selected condition."
    )

sidebar_section_title("Literature & Genetics", 5)

if "s_het" in df.columns:
    max_s_het = float(df["s_het"].max())
    s_het_range = st.sidebar.slider(
        tutorial_label("s_het range", 5),
        0.0,
        max(1.0, max_s_het),
        (0.0, max(1.0, max_s_het)),
        step=0.01,
        help="Keep genes whose heterozygous constraint score falls within this range. Higher values indicate less tolerance to heterozygous loss-of-function."
    )
else:
    s_het_range = (0.0, float("inf"))

if "dominant_mutation_count" in df.columns:
    max_mut = int(df["dominant_mutation_count"].fillna(0).max())
    mutation_count_range = st.sidebar.slider(
        tutorial_label("Dominant Mutation Count range", 5),
        0,
        max_mut,
        (0, max_mut),
        step=1,
        help="Keep genes whose known dominant mutation count falls within this range."
    )
else:
    mutation_count_range = (0, float("inf"))

if "Citation_Count" in df.columns:
    max_cite = int(df["Citation_Count"].fillna(0).max())
    citation_count_range = st.sidebar.slider(
        tutorial_label("Citation Count range", 5),
        0,
        max_cite,
        (0, max_cite),
        step=1,
        help="Keep genes whose literature citation count falls within this range."
    )
else:
    citation_count_range = (0, float("inf"))

with st.sidebar.expander("Clinical Phenotypes (HPO)"):
    st.markdown("*Check to only show genes associated with:*")
    req_cardio = st.checkbox(tutorial_label("Cardiovascular System", 5), help="Keep genes annotated to cardiovascular system abnormality.")
    req_neuro = st.checkbox(tutorial_label("Nervous System", 5), help="Keep genes annotated to nervous system abnormality.")
    req_metab = st.checkbox(tutorial_label("Metabolism / Homeostasis", 5), help="Keep genes annotated to metabolism or homeostasis abnormality.")
    req_musculo = st.checkbox(tutorial_label("Musculoskeletal System", 5), help="Keep genes annotated to musculoskeletal system abnormality.")

with st.sidebar.expander("ClinGen Dosage Sensitivity"):
    hi_filter_categories = st.multiselect(
        tutorial_label("HI categories", 5),
        CLINGEN_CATEGORY_OPTIONS,
        default=CLINGEN_CATEGORY_OPTIONS,
        help="ClinGen HI categories. Strong and emerging evidence are combined. No and little evidence are combined."
    )
    ts_filter_categories = st.multiselect(
        tutorial_label("TS categories", 5),
        CLINGEN_CATEGORY_OPTIONS,
        default=CLINGEN_CATEGORY_OPTIONS,
        help="ClinGen TS categories. Strong and emerging evidence are combined. No and little evidence are combined."
    )

if variant_master_df.empty:
    sidebar_section_title("Variant Filters", 6, 7)
    st.sidebar.info("Variant table not found at the configured data paths.")
    selected_variant_cell_lines = []
    selected_variant_strategies = []
    selected_editor_categories = EDITOR_CATEGORY_OPTIONS.copy()
    min_pop_freq = 0.0
    var_logic = "OR"
    strat_logic = "OR"
    pam_scope = "Pre-PAM"
else:
    variant_cell_lines = sorted(variant_master_df["Cell_Line"].dropna().unique().tolist())
    variant_strategies = variant_strategy_options(variant_master_df)

    sidebar_section_title("Variant Thresholds", 6, 7)

    max_popfreq = float(
        pd.to_numeric(variant_master_df["Population_Variant_Frequency"], errors="coerce").dropna().max()
    ) if variant_master_df["Population_Variant_Frequency"].notna().any() else 1.0

    min_pop_freq = st.sidebar.slider(
        tutorial_label("Min Population Variant Frequency", 7),
        0.0,
        max(1.0, max_popfreq),
        0.0,
        step=0.01,
        help="Keep variants with population frequency at or above this value. Variants with missing frequency are retained."
    )

    pam_scope = st.sidebar.radio(
        tutorial_label("Targetability Scope", 7),
        ["Pre-PAM", "Post-PAM"],
        help="Pre-PAM uses the expanded variant universe before PAM filtering. Post-PAM keeps only variants that pass PAM filtering."
    )

    sidebar_section_title("Variant Conditions", 6, 7)

    selected_variant_cell_lines = st.sidebar.multiselect(
        tutorial_label("Variant Cell Line", 7),
        variant_cell_lines,
        default=variant_cell_lines,
        format_func=prettify_cell_line,
        help="Choose which cell lines must contain targetable heterozygous variants."
    )

    var_logic = st.sidebar.radio(
        tutorial_label("Match Logic for Variant Cell Lines:", 7),
        ["OR (Variants in ANY selected line)", "AND (Variants in ALL selected lines)"],
        help="OR keeps genes with variants in at least one selected line. AND requires variants in every selected line."
    )

    selected_variant_strategies = st.sidebar.multiselect(
        tutorial_label("Editing Strategy", 6, 7),
        variant_strategies,
        default=[x for x in variant_strategies if x != CRISPROFF_HIGH_CONF_OPTION],
        help="Choose targetability/editing strategies to include. CRISPRoff high confidence keeps rows whose variant overlaps the promoter-window CRISPRoff target-site map."
    )

    variant_editor_categories = editor_category_options(variant_master_df)
    selected_editor_categories = st.sidebar.multiselect(
        tutorial_label("Base Editor", 6, 7),
        variant_editor_categories,
        default=variant_editor_categories,
        help="Filter variants by editor assignment category from editor_assignment_info.csv. Default includes all categories and rows with no editor assignment info."
    )

    strat_logic = st.sidebar.radio(
        tutorial_label("Match Logic for Editing Strategies:", 6, 7),
        ["OR (Variants in ANY selected strategy)", "AND (Variants in ALL selected strategies)"],
        help="OR keeps genes targetable by at least one selected strategy. AND requires every selected strategy."
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
annot_cols = [
    "Gene_Symbol",
    "dominant_mutation_count",
    "Citation_Count",
    "s_het",
    "HPO_Cardiovascular",
    "HPO_Nervous",
    "HPO_Metabolism",
    "HPO_Musculoskeletal",
    "targetable_epi_silencing_100_200_prom",
    "ClinGen_HI_Score",
    "ClinGen_HI_Label",
    "ClinGen_HI_Category",
    "ClinGen_TS_Score",
    "ClinGen_TS_Label",
    "ClinGen_TS_Category",
]
available_annot_cols = [c for c in annot_cols if c in df.columns]
all_gene_level_df = df[available_annot_cols].drop_duplicates()
gene_level_df = all_gene_level_df.copy()

if "dominant_mutation_count" in gene_level_df.columns:
    gene_level_df = gene_level_df[
        gene_level_df["dominant_mutation_count"].between(
            mutation_count_range[0], mutation_count_range[1], inclusive="both"
        )
    ]
if "Citation_Count" in gene_level_df.columns:
    gene_level_df = gene_level_df[
        gene_level_df["Citation_Count"].between(
            citation_count_range[0], citation_count_range[1], inclusive="both"
        )
    ]
if "s_het" in gene_level_df.columns:
    gene_level_df = gene_level_df[
        gene_level_df["s_het"].between(s_het_range[0], s_het_range[1], inclusive="both")
    ]
if req_cardio and "HPO_Cardiovascular" in gene_level_df.columns:
    gene_level_df = gene_level_df[gene_level_df["HPO_Cardiovascular"] == True]
if req_neuro and "HPO_Nervous" in gene_level_df.columns:
    gene_level_df = gene_level_df[gene_level_df["HPO_Nervous"] == True]
if req_metab and "HPO_Metabolism" in gene_level_df.columns:
    gene_level_df = gene_level_df[gene_level_df["HPO_Metabolism"] == True]
if req_musculo and "HPO_Musculoskeletal" in gene_level_df.columns:
    gene_level_df = gene_level_df[gene_level_df["HPO_Musculoskeletal"] == True]
if "ClinGen_HI_Score" in gene_level_df.columns:
    gene_level_df = gene_level_df[clingen_score_filter_mask(gene_level_df, "ClinGen_HI_Score", hi_filter_categories)]
if "ClinGen_TS_Score" in gene_level_df.columns:
    gene_level_df = gene_level_df[clingen_score_filter_mask(gene_level_df, "ClinGen_TS_Score", ts_filter_categories)]

if gene_search_active:
    if gene_search_matches:
        gene_level_df = all_gene_level_df[
            all_gene_level_df["Gene_Symbol"].map(normalize_gene_symbol).isin(
                {normalize_gene_symbol(g) for g in gene_search_matches}
            )
        ].copy()
        filtered_gene_set = set(gene_search_matches)
    else:
        gene_level_df = all_gene_level_df.iloc[0:0].copy()
        filtered_gene_set = set()
else:
    filtered_gene_set = set(gene_level_df["Gene_Symbol"].dropna().unique())

# Expression Application
if include_expression:
    if gene_search_active and gene_search_matches:
        base_df = expression_df.copy()
    else:
        base_df = expression_df[
            (expression_df["Detectability_Context"].isin(selected_contexts))
        ].copy()

    for c in [det_col, mean_all_col, mean_detected_col, agg_col]:
        base_df[c] = safe_numeric(base_df[c], fill_value=0)

    if gene_search_active:
        passed_expr_df = base_df[
            base_df["Gene_Symbol"].map(normalize_gene_symbol).isin(
                {normalize_gene_symbol(g) for g in gene_search_matches}
            )
        ].copy()
        passed_expr_df["Expression_Filter_Mode"] = "Gene search"
    elif expression_filter_mode == "Manual thresholds":
        passed_expr_df = base_df[
            (base_df[det_col] >= min_det) &
            (base_df[agg_col] >= min_cpm) &
            (base_df[mean_detected_col] >= min_mean_detected)
        ].copy()
        
        if expr_logic.startswith("AND") and len(selected_contexts) > 0:
            gene_context_counts = passed_expr_df.groupby("Gene_Symbol")["Detectability_Context"].nunique()
            valid_genes = gene_context_counts[gene_context_counts == len(selected_contexts)].index
            passed_expr_df = passed_expr_df[passed_expr_df["Gene_Symbol"].isin(valid_genes)]
        passed_expr_df["Expression_Filter_Mode"] = "Manual thresholds"
    else:
        rank_df = base_df.copy()
        rank_cols = {
            "Detection rate": det_col,
            "Aggregated CPM": agg_col,
            "Mean expression among detected cells": mean_detected_col,
        }
        rank_basis_to_percentile_col = {}
        percentile_cols = []
        for label, col in rank_cols.items():
            pct_col = f"{label}_Percentile"
            rank_df[pct_col] = (
                rank_df.groupby("Detectability_Context")[col]
                .rank(method="average", pct=True)
                .fillna(0)
            )
            rank_basis_to_percentile_col[label] = pct_col
            percentile_cols.append(pct_col)

        if expression_rank_basis == "Average percentile from all":
            rank_df["Expression_Rank_Score"] = rank_df[percentile_cols].mean(axis=1)
        else:
            rank_df["Expression_Rank_Score"] = rank_df[
                rank_basis_to_percentile_col[expression_rank_basis]
            ]

        if expr_logic.startswith("AND") and len(selected_contexts) > 0:
            context_counts = rank_df.groupby("Gene_Symbol")["Detectability_Context"].nunique()
            complete_genes = context_counts[context_counts == len(selected_contexts)].index
            rank_df = rank_df[rank_df["Gene_Symbol"].isin(complete_genes)]
            gene_rank_df = (
                rank_df.groupby("Gene_Symbol", as_index=False)["Expression_Rank_Score"]
                .mean()
            )
        else:
            gene_rank_df = (
                rank_df.groupby("Gene_Symbol", as_index=False)["Expression_Rank_Score"]
                .max()
            )

        gene_rank_df = gene_rank_df.sort_values(
            by=["Expression_Rank_Score", "Gene_Symbol"],
            ascending=[False, True]
        ).head(int(top_n_genes)).copy()
        gene_rank_df["Expression_Rank_Position"] = range(1, len(gene_rank_df) + 1)
        passed_expr_df = rank_df[rank_df["Gene_Symbol"].isin(gene_rank_df["Gene_Symbol"])].merge(
            gene_rank_df,
            how="left",
            on="Gene_Symbol",
            suffixes=("", "_Gene")
        )
        passed_expr_df["Expression_Filter_Mode"] = expression_rank_basis
        
    filtered_df = passed_expr_df[passed_expr_df["Gene_Symbol"].isin(filtered_gene_set)].copy()
    filtered_gene_set = set(filtered_df["Gene_Symbol"].dropna().unique())
else:
    filtered_df = pd.DataFrame()

# Variant Application
if variant_master_df.empty:
    filtered_variant_df = pd.DataFrame()
    overlap_variant_df = pd.DataFrame()
    shared_position_variant_df = pd.DataFrame()
    gene_variant_summary = pd.DataFrame()
    pairwise_variant_summary = pd.DataFrame()
    pairwise_gene_summary = pd.DataFrame()
    variant_universe_context_summary = pd.DataFrame()
    variant_universe_line_summary = pd.DataFrame()
    variant_universe_strategy_summary = pd.DataFrame()
    shared_position_context_summary = pd.DataFrame()
    shared_position_gene_summary = pd.DataFrame()
    shared_position_line_summary = pd.DataFrame()
    shared_position_strategy_summary = pd.DataFrame()
    cell_line_uniqueness_summary = pd.DataFrame()
    strategy_overlap_summary = pd.DataFrame()
else:
    filtered_variant_df = variant_master_df.copy()

    if gene_search_active:
        if gene_search_matches:
            filtered_variant_df = filtered_variant_df[
                filtered_variant_df["Gene"].map(normalize_gene_symbol).isin(
                    {normalize_gene_symbol(g) for g in gene_search_matches}
                )
            ]
        else:
            filtered_variant_df = filtered_variant_df.iloc[0:0].copy()
    else:
        if pam_scope == "Post-PAM":
            filtered_variant_df = filtered_variant_df[
                filtered_variant_df["Is_PAM_Filtered"].fillna(False).astype(bool)
            ]

        if selected_variant_cell_lines:
            filtered_variant_df = filtered_variant_df[filtered_variant_df["Cell_Line"].isin(selected_variant_cell_lines)]
        if selected_variant_strategies:
            filtered_variant_df = apply_variant_strategy_filter(filtered_variant_df, selected_variant_strategies)
        if selected_editor_categories:
            filtered_variant_df = apply_editor_category_filter(filtered_variant_df, selected_editor_categories)

        if filtered_gene_set:
            filtered_variant_df = filtered_variant_df[filtered_variant_df["Gene"].astype(str).isin(filtered_gene_set)]
        else:
            filtered_variant_df = filtered_variant_df.iloc[0:0].copy()

        filtered_variant_df = filtered_variant_df[
            filtered_variant_df["Population_Variant_Frequency"].isna() | 
            (filtered_variant_df["Population_Variant_Frequency"] >= min_pop_freq)
        ]

    if not gene_search_active and var_logic.startswith("AND") and len(selected_variant_cell_lines) > 0:
        var_counts = filtered_variant_df.groupby("Gene")["Cell_Line"].nunique()
        valid_var_genes = var_counts[var_counts == len(selected_variant_cell_lines)].index
        filtered_variant_df = filtered_variant_df[filtered_variant_df["Gene"].isin(valid_var_genes)]

    if not gene_search_active and strat_logic.startswith("AND") and len(selected_variant_strategies) > 0:
        valid_strat_genes = genes_matching_all_variant_strategy_options(
            filtered_variant_df, selected_variant_strategies
        )
        filtered_variant_df = filtered_variant_df[filtered_variant_df["Gene"].astype(str).isin(valid_strat_genes)]

    overlap_variant_df = expand_variant_rows_for_selected_strategy_options(
        filtered_variant_df, selected_variant_strategies
    )
    shared_position_variant_df = build_shared_position_only_df(overlap_variant_df)
    gene_variant_summary = build_variant_gene_summary(filtered_variant_df)
    pairwise_variant_summary = build_pairwise_summary(overlap_variant_df)
    pairwise_gene_summary = build_pairwise_gene_summary(overlap_variant_df)
    variant_universe_context_summary = build_variant_universe_context_summary(overlap_variant_df)
    variant_universe_line_summary = build_variant_universe_line_summary(overlap_variant_df)
    variant_universe_strategy_summary = build_variant_universe_strategy_summary(overlap_variant_df)
    shared_position_context_summary = build_variant_universe_context_summary(shared_position_variant_df)
    shared_position_gene_summary = build_variant_gene_summary(shared_position_variant_df)
    shared_position_line_summary = build_variant_universe_line_summary(shared_position_variant_df)
    shared_position_strategy_summary = build_variant_universe_strategy_summary(shared_position_variant_df)
    cell_line_uniqueness_summary = build_cell_line_uniqueness_summary(filtered_variant_df)
    strategy_overlap_summary = build_strategy_overlap_summary(overlap_variant_df)

# ------------------------------------------------
# UI Dashboards
# ------------------------------------------------
gene_targetable_count = int(filtered_variant_df["Gene"].nunique()) if not filtered_variant_df.empty else 0
variant_targetable_count = (
    int(add_variant_site_key(filtered_variant_df)["Variant_Site_Key"].nunique())
    if not filtered_variant_df.empty else 0
)
condition_position_summary = build_condition_position_summary(overlap_variant_df)

tabs = st.tabs(["Input Summary", "Gene-level Results", "Variant-level Results", "Cell-line Intersections", "Downloads"])

with tabs[0]:
    st.subheader("Input Summary")
    st.caption("These are the fixed starting dataset counts before filtering.")

    input_cols = st.columns(3)
    input_cols[0].metric("Input genes", INPUT_SUMMARY_GENE_COUNT)
    input_cols[1].metric("Input cell lines", INPUT_SUMMARY_CELL_LINE_COUNT)
    input_cols[2].metric("Input editing strategies", INPUT_SUMMARY_EDITING_STRATEGY_COUNT)

with tabs[1]:
    st.subheader("Gene-level Results")
    st.metric("Genes targetable after all filters", gene_targetable_count)

    if gene_search_active:
        if gene_search_matches:
            st.info(f"Searched {', '.join(gene_search_matches)}.")
        else:
            st.warning(f"No exact gene symbol match was found for '{gene_search_query}'.")

    if gene_search_active:
        st.subheader("Gene Annotation")
        if gene_level_df.empty:
            st.info("No gene annotation row is available for this search.")
        else:
            render_bold_dataframe(gene_level_df.set_index("Gene_Symbol"))

        if include_expression:
            st.subheader("Expression Records")
            if filtered_df.empty:
                st.info("No expression rows are available for this gene.")
            else:
                preferred_expr_cols = [
                    "Gene_Symbol",
                    "Detectability_Cell_Type",
                    "Detectability_Condition",
                    det_col,
                    agg_col,
                    mean_all_col,
                    mean_detected_col,
                    "Expression_Filter_Mode",
                ]
                expr_cols = [c for c in preferred_expr_cols if c in filtered_df.columns]
                render_bold_dataframe(filtered_df[expr_cols].set_index("Gene_Symbol"))

    if (
        include_expression
        and not gene_search_active
        and expression_filter_mode == "Top N ranked genes"
        and not filtered_df.empty
    ):
        st.subheader("Top Expression-Ranked Genes")
        rank_score_col = (
            "Expression_Rank_Score_Gene"
            if "Expression_Rank_Score_Gene" in filtered_df.columns
            else "Expression_Rank_Score"
        )
        rank_cols = [
            "Gene_Symbol",
            "Expression_Rank_Position",
            rank_score_col,
            "Expression_Filter_Mode",
        ]
        rank_cols = [c for c in rank_cols if c in filtered_df.columns]
        rank_summary_df = (
            filtered_df[rank_cols]
            .drop_duplicates(subset=["Gene_Symbol"])
            .sort_values("Expression_Rank_Position")
        )
        render_bold_dataframe(rank_summary_df.set_index("Gene_Symbol"))

    if variant_universe_context_summary.empty:
        st.info("No gene-level variant contexts remain after the current filters.")
    else:
        render_summary_heatmap(
            build_context_matrix(variant_universe_context_summary, "Genes_with_Filtered_Variants"),
            "Genes with Targetable Heterozygous Variants by Context",
            "Genes",
            "Greens"
        )

    if gene_variant_summary.empty:
        st.info("No per-gene variant counts are available for the current filters.")
    else:
        top_gene_variant_counts = gene_variant_summary.sort_values(
            by=["Unique_Variant_Sites", "Gene_Symbol"],
            ascending=[False, True]
        ).head(10)
        fig_top_genes = px.bar(
            top_gene_variant_counts,
            x="Gene_Symbol",
            y="Unique_Variant_Sites",
            title="Top 10 Genes by Filtered Variant Count",
            labels={
                "Gene_Symbol": "Gene",
                "Unique_Variant_Sites": "Variant positions",
            }
        )
        fig_top_genes.update_xaxes(tickangle=-30)
        st.plotly_chart(fig_top_genes, width="stretch")

    if include_expression and not base_df.empty:
        fig_hist = px.histogram(
            base_df,
            x=det_col,
            color="Detectability_Condition",
            facet_row="Detectability_Cell_Type",
            nbins=50,
            barmode="overlay",
            opacity=0.65,
            title=f"Distribution within Selected Conditions ({qc_level})",
            labels={
                det_col: f"{qc_level} Detection Rate (%)",
                "Detectability_Cell_Type": "Cell type",
                "Detectability_Condition": "Condition",
            }
        )
        fig_hist.for_each_annotation(
            lambda annotation: annotation.update(text=annotation.text.split("=")[-1])
        )
        if expression_filter_mode == "Manual thresholds":
            fig_hist.add_vline(
                x=min_det,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Cutoff: {min_det}%",
                annotation_position="top right"
            )
        st.plotly_chart(fig_hist, width="stretch")
    elif not include_expression:
        st.info("Expression filtering is turned off, so the detectability histogram is hidden.")
    else:
        st.info("No expression records are available for the current settings.")

with tabs[2]:
    st.subheader("Variant-level Results")
    st.metric("Variants targetable after filtering", variant_targetable_count)

    if gene_search_active:
        st.subheader("Variant Records")
        if filtered_variant_df.empty:
            st.info("No variant rows are available for this gene.")
        else:
            preferred_variant_cols = [
                "Gene",
                "Cell_Line",
                "Editing_Strategy",
                "Chromosome",
                "Position",
                "Ref_Allele",
                "Alt_Allele",
                "Population_Variant_Frequency",
                "PAM_Filter_Status",
                "Variant_Source",
            ]
            variant_cols = [c for c in preferred_variant_cols if c in filtered_variant_df.columns]
            render_bold_dataframe(filtered_variant_df[variant_cols].set_index("Gene"))

    if variant_universe_context_summary.empty:
        st.info("No variant-level contexts remain after the current filters.")
    else:
        render_summary_heatmap(
            build_context_matrix(variant_universe_context_summary, "Unique_Variant_Sites"),
            "Targetable Heterozygous Variant Positions by Context",
            "Variant positions",
            "Blues"
        )

    if condition_position_summary.empty:
        st.info("No condition-level position summaries are available for the current filters.")
    else:
        condition_position_summary = condition_position_summary.copy()
        condition_position_summary["Shared_Unique_Variant_Sites"] = (
            condition_position_summary["Total_Unique_Variant_Sites"] -
            condition_position_summary["Private_Unique_Variant_Sites"]
        )
        total_condition_positions = condition_position_summary.sort_values(
            by=["Total_Unique_Variant_Sites", "Condition"],
            ascending=[False, True]
        )

        fig_total_condition = px.bar(
            total_condition_positions,
            x="Condition",
            y=["Shared_Unique_Variant_Sites", "Private_Unique_Variant_Sites"],
            title="Total Variant Sites per Context with Context-Unique Sites Highlighted",
            labels={
                "Condition": "Cell line + editing strategy",
                "value": "Variant sites",
                "variable": "Site class",
            }
        )
        fig_total_condition.update_layout(
            barmode="stack",
            legend_title_text="Site class"
        )
        fig_total_condition.for_each_trace(
            lambda trace: trace.update(
                name=(
                    "Unique to this context"
                    if trace.name == "Private_Unique_Variant_Sites"
                    else "Also seen in other contexts"
                )
            )
        )
        fig_total_condition.update_xaxes(tickangle=-30)
        st.plotly_chart(fig_total_condition, width="stretch")

    st.subheader("Variant Position Intersections")
    st.caption(
        "The UpSet plot is ordered from high to low intersection size. "
        "Each count is the number of exact variant positions shared by at least two of the shown contexts."
    )

    if overlap_variant_df.empty:
        st.info("No variants are available for the UpSet plot.")
    else:
        tmp_upset = add_variant_site_key(overlap_variant_df)
        tmp_upset["Condition"] = tmp_upset.apply(
            lambda row: condition_label(row["Cell_Line"], row["Editing_Strategy"]),
            axis=1
        )

        site_conditions = tmp_upset.groupby("Variant_Site_Key")["Condition"].apply(
            lambda x: tuple(sorted(set(x)))
        )
        combination_counts = site_conditions.value_counts().sort_values(ascending=False)
        combination_counts = combination_counts[combination_counts.index.map(len) >= 2]

        if combination_counts.empty:
            st.info("No position intersections are available for the current filters.")
        else:
            top_n_intersections = st.slider(
                "Show top intersections",
                min_value=5,
                max_value=40,
                value=20,
                key="upset_top_n_intersections"
            )
            top_combinations = combination_counts.head(top_n_intersections)

            upset_data = from_memberships(top_combinations.index, data=top_combinations.values)
            fig_upset = plt.figure(figsize=(10, 5.5))
            plot(
                upset_data,
                fig=fig_upset,
                show_counts="%d",
                element_size=28,
                sort_by="cardinality",
                totals_plot_elements=0
            )
            if fig_upset.axes:
                fig_upset.axes[0].set_ylabel("Number of variant positions shared by contexts")
            st.pyplot(fig_upset)
            plt.close(fig_upset)

with tabs[3]:
    st.subheader("Cell-line Intersections")
    st.caption(
        "Venn diagrams show exact intersections between selected variant cell lines after the current sidebar filters."
    )

    if filtered_variant_df.empty:
        st.info("No variant rows are available for cell-line intersections after the current filters.")
    else:
        venn_item_type = st.radio(
            "Intersection item",
            ["Variant positions", "Genes"],
            horizontal=True,
            key="cell_line_venn_item_type",
        )
        venn_df = add_variant_site_key(filtered_variant_df)
        set_value_col = "Variant_Site_Key" if venn_item_type == "Variant positions" else "Gene"
        venn_sets = {}
        for cell_line in sorted(venn_df["Cell_Line"].dropna().astype(str).unique().tolist()):
            values = set(venn_df.loc[venn_df["Cell_Line"].astype(str) == cell_line, set_value_col].dropna().astype(str))
            if values:
                venn_sets[prettify_cell_line(cell_line)] = values

        if len(venn_sets) < 2:
            st.info("At least two cell lines are needed for a Venn diagram.")
        elif len(venn_sets) > 6:
            st.info("The Venn plot supports up to six sets; reduce the selected cell lines.")
        else:
            fig_venn, ax_venn = plt.subplots(figsize=(8, 8))
            draw_venn_diagram(
                venn_sets,
                fmt="{size}",
                cmap="Set2",
                alpha=0.45,
                fontsize=10,
                legend_loc=None,
                ax=ax_venn,
            )
            ax_venn.set_title(f"Cell-line intersections: {venn_item_type}")
            label_positions_by_n = {
                2: [(0.15, 0.83), (0.78, 0.83)],
                3: [(0.12, 0.82), (0.78, 0.82), (0.48, 0.08)],
                4: [(0.0, 0.75), (0.28, 0.95), (0.65, 0.95), (0.88, 0.75)],
                5: [(0.0, 0.55), (0.37, 0.93), (0.88, 0.71), (0.78, 0.08), (0.18, 0.05)],
                #6: [(0.05, 0.75), (0.30, 0.97), (0.68, 0.97), (0.92, 0.75), (0.70, 0.08), (0.22, 0.08)],
            }
            for label, (x, y) in zip(venn_sets.keys(), label_positions_by_n.get(len(venn_sets), [])):
                ax_venn.text(x, y, label, transform=ax_venn.transAxes, fontsize=12)
            st.pyplot(fig_venn)
            plt.close(fig_venn)

        if venn_sets:
            st.subheader("Cell-line set sizes")
            set_size_df = pd.DataFrame({
                "Cell line": list(venn_sets.keys()),
                venn_item_type: [len(values) for values in venn_sets.values()],
            }).sort_values(venn_item_type, ascending=False)
            render_bold_dataframe(set_size_df.set_index("Cell line"))

with tabs[4]:
    st.subheader("Downloads")
    st.caption("These files reflect the current filters applied in the sidebar.")

    download_col1, download_col2 = st.columns(2)

    with download_col1:
        st.download_button(
            label="Download gene-level results",
            data=gene_variant_summary.to_csv(index=False).encode("utf-8") if not gene_variant_summary.empty else b"",
            file_name="dnd_gene_level_filtered.csv",
            mime="text/csv",
            disabled=gene_variant_summary.empty,
        )
        if gene_variant_summary.empty:
            st.info("No gene-level rows are available for download.")

    with download_col2:
        st.download_button(
            label="Download variant-level results",
            data=filtered_variant_df.to_csv(index=False).encode("utf-8") if not filtered_variant_df.empty else b"",
            file_name="dnd_variant_level_filtered.csv",
            mime="text/csv",
            disabled=filtered_variant_df.empty,
        )
        if filtered_variant_df.empty:
            st.info("No variant-level rows are available for download.")




