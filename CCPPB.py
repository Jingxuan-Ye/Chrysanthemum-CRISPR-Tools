import streamlit as st
import pandas as pd
import re
from io import BytesIO
from typing import List, Tuple

# ==========================================
# Core Biological & Sequence Configurations
# ==========================================
P1F_PREFIX = "atatatGGTCTCT"
PF_PREFIX = "atatatGGTCTCA"
PR_PREFIX = "attattGGTCTCA"

CHRY_STU2_CONFIG = {
    "start_link": "gcag",
    "tail": "ctgcctatacggcagtg",
    "scaffold": "gttttagagctagaaatagcaa",
    "last_r_overhang": "aaac"
}

SINGLE_F_OVERHANG = "GCAG"
SINGLE_R_OVERHANG = "AAAC"

DNA_RE = re.compile(r"^[ACGT]+$", re.I)
COMP = str.maketrans("ACGTacgt", "TGCAtgca")

# Search priority for dynamic overhang shifting (index represents start of 4bp overhang)
# 10 means standard 14/10 split (overhang at pos 11-14)
# Then it expands outwards to find alternative unique overhangs.
CANDIDATE_INDICES = [10, 11, 9, 12, 8, 13, 7, 14, 6, 15]


def rc(seq: str) -> str:
    return seq.translate(COMP)[::-1]


def clean_seq(seq: str) -> str:
    return re.sub(r"[^ACGTacgt]", "", seq).upper()


def normalize_protospacer(s: str) -> Tuple[str, int]:
    cleaned = clean_seq(s)
    L = len(cleaned)
    if L < 20:
        raise ValueError(f"Spacer length {L} < 20. Full 20bp sequence is required.")
    return cleaned[:20], L


def build_primers(spacers: List[str]) -> List[dict]:
    N = len(spacers)
    if N < 1:
        raise ValueError("At least 1 spacer is required.")
    rows = []

    # ---------------------------------------------------------
    # Scenario 1: Single sgRNA (Annealing)
    # ---------------------------------------------------------
    if N == 1:
        t1 = spacers[0]
        rows.append({"primer_name": "sgRNA-F", "sequence": f"{SINGLE_F_OVERHANG}{t1}",
                     "note": "Single sgRNA annealing (Top strand)"})
        rows.append({"primer_name": "sgRNA-R", "sequence": f"{SINGLE_R_OVERHANG}{rc(t1)}",
                     "note": "Single sgRNA annealing (Bottom strand)"})
        return rows

    # ---------------------------------------------------------
    # Scenario 2: Multiplex sgRNA (Golden Gate with Dynamic Shifting)
    # ---------------------------------------------------------
    start_link = CHRY_STU2_CONFIG["start_link"].upper()
    tail = clean_seq(CHRY_STU2_CONFIG["tail"])
    scaffold = clean_seq(CHRY_STU2_CONFIG["scaffold"])
    last_r_overhang = CHRY_STU2_CONFIG["last_r_overhang"].upper()

    # Track used overhangs to prevent cross-talk in assembly
    used_overhangs = {start_link, last_r_overhang}
    resolved_splits = {}  # Stores the optimal split index for each junction

    # Pre-calculate unique overhangs for all internal junctions
    for k in range(2, N):
        tk = spacers[k - 1]
        resolved = False
        for idx in CANDIDATE_INDICES:
            candidate_oh = tk[idx:idx + 4].upper()
            if candidate_oh not in used_overhangs:
                used_overhangs.add(candidate_oh)
                resolved_splits[k] = (idx, candidate_oh)
                resolved = True
                break

        if not resolved:
            raise ValueError(
                f"Conflict resolution failed for Protospacer {k}. Unable to find a unique 4bp overhang by shifting. Please change the sgRNA sequence.")

    # Generate P1-F
    t1 = spacers[0]
    rows.append({"primer_name": "P1-F", "sequence": f"{P1F_PREFIX}{start_link}{t1}{scaffold}",
                 "note": f"T1 full 20bp (Link: {start_link})"})

    # Generate Internal Primers using the dynamically resolved split points
    for k in range(2, N):
        tk = spacers[k - 1]
        idx, oh = resolved_splits[k]

        # Calculate lengths based on shifted index
        left_len = idx + 4
        right_len = 20 - idx

        first_part = tk[:left_len]
        last_part = tk[idx:]

        rows.append({"primer_name": f"P{k - 1}-R", "sequence": f"{PR_PREFIX}{rc(first_part)}{tail}",
                     "note": f"T{k} internal R (Split {left_len}/{right_len}, OH: {oh})"})
        rows.append({"primer_name": f"P{k}-F", "sequence": f"{PF_PREFIX}{last_part}{scaffold}",
                     "note": f"T{k} internal F (Split {left_len}/{right_len}, OH: {oh})"})

    # Generate Last R
    tn = spacers[-1]
    rows.append({"primer_name": f"P{N - 1}-R", "sequence": f"{PR_PREFIX}{last_r_overhang}{rc(tn)}{tail}",
                 "note": f"Last R: T{N} (Overhang: {last_r_overhang})"})

    return rows


# ==========================================
# Web Frontend UI
# ==========================================

st.set_page_config(page_title="CCPPB - Primer Builder", layout="wide")

st.markdown("""
    <style>
    .custom-header {
        border-bottom: 2px solid #367c39;
        padding-bottom: 10px;
        margin-bottom: 30px;
        font-size: 24px;
        font-weight: 700;
        color: #333333;
    }
    div.stButton > button:first-child {
        background-color: #367c39;
        color: white;
        border: none;
        padding: 8px 32px;
        border-radius: 4px;
    }
    div.stButton > button:first-child:hover {
        background-color: #2b612d;
        color: white;
    }
    .st-emotion-cache-16idsys p {
        font-size: 16px;
        font-weight: 600;
        margin-top: 5px;
    }
    .field-label {
        font-size: 16px;
        font-weight: 600;
        color: #333333;
        white-space: nowrap;
        margin-top: 6px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="custom-header"><em>Chrysanthemum</em> CRISPR Plasmid Primer Builder (CCPPB)</div>',
            unsafe_allow_html=True)

col1, col2 = st.columns([1.6, 5.4], vertical_alignment="center")
with col1:
    st.markdown('<div class="field-label">System :</div>', unsafe_allow_html=True)
with col2:
    st.radio("System_hidden", options=["Chry-STU2.0 (Csy4/SpCas9)"], label_visibility="collapsed", horizontal=True)

st.write("")

col3, col4 = st.columns([1.6, 5.4], vertical_alignment="top")
with col3:
    st.markdown('<div class="field-label">Input Protospacers :</div>', unsafe_allow_html=True)
with col4:
    raw_spacers = st.text_area(
        "Protospacers_hidden",
        height=180,
        placeholder="Enter protospacer sequences, one per line or comma-separated...",
        label_visibility="collapsed"
    )
    st.markdown("<span style='color: #367c39; font-size: 14px;'>e.g., <i>ATCGATCGATCGATCGATCG</i></span>",
                unsafe_allow_html=True)

    st.write("")
    submit_btn = st.button("Run")

st.markdown("---")

# ==========================================
# Execution & Results
# ==========================================
if submit_btn:
    if not raw_spacers.strip():
        st.warning("⚠️ Run failed: Please enter at least one Protospacer sequence.")
    else:
        try:
            raw_list = [x.strip() for x in raw_spacers.replace("\n", ",").split(",") if x.strip()]
            spacers = []
            truncated_info = []

            for s in raw_list:
                sp20, L = normalize_protospacer(s)
                if L > 20:
                    truncated_info.append((s, sp20, L))
                spacers.append(sp20)

            # The new build_primers function automatically resolves overhang conflicts
            primers = build_primers(spacers)

            if truncated_info:
                for orig, used, L in truncated_info:
                    st.toast(f"Input length {L}bp, automatically truncated to first 20bp: {used}", icon="✂️")

            mode_text = "Single sgRNA Annealing" if len(spacers) == 1 else "Multiplex Golden Gate Assembly"
            st.success(
                f"✅ Run successful! Generation mode: {mode_text}. Overhang conflicts automatically resolved by dynamic shifting.")

            df = pd.DataFrame(primers)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                height=min(35 + len(df) * 42, 500),
                column_config={
                    "primer_name": st.column_config.TextColumn("primer_name", width="medium"),
                    "sequence": st.column_config.TextColumn("sequence", width="large"),
                    "note": st.column_config.TextColumn("note", width="large"),
                },
            )

            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="primers")
            output.seek(0)
            st.download_button(
                label="📥 Download XLSX",
                data=output.getvalue(),
                file_name="CCPPB_primers.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error(f"❌ Design failed: {str(e)}")
