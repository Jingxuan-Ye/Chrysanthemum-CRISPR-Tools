import json
import re
from datetime import datetime
from io import BytesIO
from typing import List, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

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
# Visitor Counter
# ==========================================
# CounterAPI stores the counters outside Streamlit. The fixed namespace/key
# keep the totals stable after redeployment or an app URL change.
COUNTER_NAMESPACE = "chrysanthemum-crispr-tools"
COUNTER_KEY = "ccppb"
COUNTER_START_NUMBER = 25
COUNTER_LIKE_KEY = "ccppb-likes"
APP_TIMEZONE = ZoneInfo("Asia/Shanghai")
COUNTER_TIMEOUT_SECONDS = 3


def counter_request(
    action: str,
    key: str,
    *,
    read_only: bool = False,
    start_number: int | None = None,
) -> int | None:
    """Read or increment one CounterAPI counter."""
    namespace = quote(COUNTER_NAMESPACE, safe="")
    encoded_key = quote(key, safe="")
    params = {}
    if read_only:
        params["readOnly"] = "true"
    if start_number is not None:
        params["startNumber"] = str(start_number)
    query = f"?{urlencode(params)}" if params else ""
    endpoint = f"https://counterapi.com/api/{namespace}/{action}/{encoded_key}{query}"
    request = Request(endpoint, headers={"User-Agent": "CCPPB visitor counter"})

    try:
        with urlopen(request, timeout=COUNTER_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
        count = int(payload["value"])
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError):
        return None

    return count


def count_visitor_once() -> int | None:
    if st.session_state.get("visitor_counter_counted", False):
        return st.session_state.get("visitor_counter_value")

    count = counter_request("view", COUNTER_KEY, start_number=COUNTER_START_NUMBER)
    if count is None:
        return None

    st.session_state["visitor_counter_counted"] = True
    st.session_state["visitor_counter_value"] = count
    return count


def count_monthly_visit_once() -> int | None:
    month_key = f"{COUNTER_KEY}-{datetime.now(APP_TIMEZONE):%Y-%m}"
    if st.session_state.get("monthly_counter_key") == month_key:
        return st.session_state.get("monthly_counter_value")

    count = counter_request("view", month_key)
    if count is None:
        return None

    st.session_state["monthly_counter_key"] = month_key
    st.session_state["monthly_counter_value"] = count
    return count


def get_like_count() -> int | None:
    if "like_counter_value" in st.session_state:
        return st.session_state["like_counter_value"]

    count = counter_request("vote", COUNTER_LIKE_KEY, read_only=True)
    if count is not None:
        st.session_state["like_counter_value"] = count
    return count


def add_like_once() -> int | None:
    if st.session_state.get("like_clicked", False):
        return st.session_state.get("like_counter_value")

    count = counter_request("vote", COUNTER_LIKE_KEY)
    if count is None:
        return None

    st.session_state["like_clicked"] = True
    st.session_state["like_counter_value"] = count
    return count


def format_count(value: int | None) -> str:
    return f"{value:,}" if value is not None else "—"


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
    .stats-card {
        background: #f8fbf8;
        border: 1px solid #dbe7dc;
        border-top: 3px solid #367c39;
        border-radius: 6px;
        padding: 14px 16px 13px;
        min-height: 82px;
    }
    .stats-label {
        color: #5b675c;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 5px;
    }
    .stats-value {
        color: #2b612d;
        font-size: 26px;
        font-weight: 700;
        line-height: 1.1;
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

            # The build_primers function automatically resolves overhang conflicts
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


# ==========================================
# Footer Statistics
# ==========================================
visitor_count = count_visitor_once()
monthly_visit_count = count_monthly_visit_once()
like_count = get_like_count()

st.markdown("---")
visitor_col, likes_col, monthly_col = st.columns(3, gap="medium")

with visitor_col:
    st.markdown(
        f"""
        <div class="stats-card">
            <div class="stats-label">Visitors</div>
            <div class="stats-value">{format_count(visitor_count)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with likes_col:
    st.markdown(
        f"""
        <div class="stats-card">
            <div class="stats-label">Likes</div>
            <div class="stats-value">{format_count(like_count)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button(
        "👍 Liked" if st.session_state.get("like_clicked", False) else "👍 Like",
        key="like_button",
        disabled=st.session_state.get("like_clicked", False),
    ):
        if add_like_once() is not None:
            st.rerun()
        st.toast("Please try again later.", icon="⚠️")

with monthly_col:
    st.markdown(
        f"""
        <div class="stats-card">
            <div class="stats-label">New visits this month</div>
            <div class="stats-value">{format_count(monthly_visit_count)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
