import os
import io
import json
import uuid
import mimetypes
from typing import Any, Dict, Optional, List

import requests
import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import boto3

# =============================================================
# Streamlit App: Visual Search with "same-type" gating (Marqo + R2)
# =============================================================
# Key upgrades vs your version:
# 1) Robust object-type inference from TOP-K image hits using rich keyword map
# 2) Hard filter by OBJECT_TYPE field when present in the index; otherwise
#    post-filter by inferred type (title/description/spec_text keywords)
# 3) Optional fusion of image and text queries (image + user text) with
#    weighted score merge
# 4) Searchable attributes expanded to include name/description/spec_text
# 5) UI chip showing detected/selected type with manual override
# =============================================================

# -----------------------------
# Config (from secrets or ENV)
# -----------------------------

def _cfg(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        return st.secrets[key]  # type: ignore[attr-defined]
    except Exception:
        return os.getenv(key, default)

MARQO_URL: str = (_cfg("MARQO_URL", "https://marqo.logicafutura.com") or "").rstrip("/")
INDEX_NAME: str = _cfg("INDEX_NAME", "furniture-index") or "furniture-index"

TITLE_FIELD: str = _cfg("TITLE_FIELD", "name") or "name"
IMAGE_FIELD: str = _cfg("IMAGE_FIELD", "image") or "image"
ALT_IMAGE_FIELD: str = _cfg("ALT_IMAGE_FIELD", "image_url") or "image_url"
DESCRIPTION_FIELD: str = _cfg("DESCRIPTION_FIELD", "description") or "description"
SPEC_TEXT_FIELD: str = _cfg("SPEC_TEXT_FIELD", "spec_text") or "spec_text"
SEARCH_BLOB_FIELD: str = _cfg("SEARCH_BLOB_FIELD", "search_blob") or "search_blob"
DOM_COLOR_FIELD: str = _cfg("DOMINANT_COLOR_FIELD", "dominant_color") or "dominant_color"
OBJECT_TYPE_FIELD: str = _cfg("OBJECT_TYPE_FIELD", "object_type") or "object_type"
SKU_FIELD: str = _cfg("SKU_FIELD", "product_id") or "product_id"
CLICK_URL_FIELD: str = _cfg("CLICK_URL_FIELD", "product_url") or "product_url"

# Cloudflare Access headers (if your Marqo is behind Cloudflare Access)
CF_ACCESS_CLIENT_ID = _cfg("CF_ACCESS_CLIENT_ID")
CF_ACCESS_CLIENT_SECRET = _cfg("CF_ACCESS_CLIENT_SECRET")

# R2 (for uploading the query image and making it web-accessible to Marqo)
R2_ENDPOINT_URL: str = _cfg("R2_ENDPOINT_URL") or ""
R2_ACCESS_KEY_ID: Optional[str] = _cfg("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY: Optional[str] = _cfg("R2_SECRET_ACCESS_KEY")
R2_BUCKET: str = _cfg("R2_BUCKET", "streamlit098") or "streamlit098"
PUBLIC_BASE_URL: str = _cfg("PUBLIC_BASE_URL", "") or ""

HEADERS: Dict[str, str] = {"Content-Type": "application/json"}
if CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET:
    HEADERS["CF-Access-Client-Id"] = CF_ACCESS_CLIENT_ID
    HEADERS["CF-Access-Client-Secret"] = CF_ACCESS_CLIENT_SECRET

# Known, normalized object types we support for gating
KNOWN_TYPES = [
    "lamp", "light", "wall lamp", "floor lamp", "table lamp", "pendant", "chandelier", "sconce",
    "table", "coffee table", "side table", "console table", "dining table",
    "sofa", "couch", "loveseat", "sectional",
    "chair", "armchair", "stool", "bench",
    "rack", "shelf", "shelving",
    "bed", "cabinet", "desk", "trolley", "mirror", "vase", "tray", "carpet", "rug"
]

# Rich keyword map to catch synonyms/variants for inference & filtering
TYPE_KEYWORDS = {
    "lamp": ["lamp", "light", "wall lamp", "floor lamp", "table lamp", "pendant", "chandelier", "sconce", "wall light"],
    "table": ["table", "coffee table", "side table", "console table", "dining table"],
    "sofa": ["sofa", "couch", "loveseat", "sectional"],
    "chair": ["chair", "armchair", "stool"],
    "bench": ["bench"],
    "rack": ["rack", "shelf", "shelving"],
    "bed": ["bed"],
    "cabinet": ["cabinet"],
    "desk": ["desk"],
    "trolley": ["trolley", "bar cart"],
    "mirror": ["mirror"],
    "vase": ["vase"],
    "tray": ["tray"],
    "rug": ["rug", "carpet"]
}

# Search attribute sets for dual-search fusion
TEXT_ATTRS = [TITLE_FIELD, DESCRIPTION_FIELD, SPEC_TEXT_FIELD, SEARCH_BLOB_FIELD]
VISUAL_ATTRS = [IMAGE_FIELD]

# Tensor fields available in the current index (keep in sync with your indexer)
KNOWN_TENSOR_FIELDS = {"name", "description", "image", "spec_text", "search_blob"}


def sanitize_attrs(attrs: List[str]) -> List[str]:
    """Map aliases (e.g., 'title' -> 'name'), drop unknowns, ensure non-empty.
    This prevents Marqo from rejecting the request when an invalid tensor field is sent.
    """
    out: List[str] = []
    for a in attrs:
        a = (a or "").strip()
        if not a:
            continue
        # Map common alias
        if a == "title":
            a = "name"
        if a in KNOWN_TENSOR_FIELDS and a not in out:
            out.append(a)
    if not out:
        # last resort: use image only
        out = ["image"] if "image" in KNOWN_TENSOR_FIELDS else list(KNOWN_TENSOR_FIELDS)
    return out

# =============================================================
# R2 helpers (upload the query image)
# =============================================================

def _r2_client():
    if not (R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY and R2_ENDPOINT_URL):
        raise RuntimeError("R2 credentials or endpoint missing. Set R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT_URL in secrets.")
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    )


def upload_query_image_to_r2(img_bytes: bytes, filename: str) -> str:
    if not PUBLIC_BASE_URL:
        raise RuntimeError("PUBLIC_BASE_URL is not configured.")
    ext = os.path.splitext(filename)[1].lower() or ".jpg"
    key = f"queries/{uuid.uuid4()}{ext}"
    content_type = mimetypes.guess_type(filename)[0] or "image/jpeg"
    r2 = _r2_client()
    r2.put_object(Bucket=R2_BUCKET, Key=key, Body=img_bytes, ContentType=content_type)
    return f"{PUBLIC_BASE_URL.rstrip('/')}/{key}"

# =============================================================
# Color helpers (optional color gating)
# =============================================================

def get_dominant_color(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((50, 50))
        data = np.array(img).reshape(-1, 3)

        def is_background(px):
            return np.all(px > 240) or np.all(px < 15)
        filtered = np.array([px for px in data if not is_background(px)])
        if filtered.size == 0:
            filtered = data
        kmeans = KMeans(n_clusters=1, n_init=10)
        kmeans.fit(filtered)
        return kmeans.cluster_centers_[0].astype(int)
    except Exception as e:
        st.warning(f"Nepavyko nustatyti spalvos: {e}")
        return np.array([0, 0, 0])


def hex_to_rgb(hex_color: str) -> np.ndarray:
    hex_color = (hex_color or "").lstrip('#')
    if len(hex_color) != 6:
        return np.array([0, 0, 0])
    return np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)])


def color_distance(c1: np.ndarray, c2: np.ndarray) -> float:
    return float(np.linalg.norm(c1 - c2))

# =============================================================
# Type inference & gating helpers
# =============================================================

def normalize_text(s: str) -> str:
    return (s or "").strip().lower()


def infer_type_from_text(text: str) -> Optional[str]:
    t = normalize_text(text)
    best_type, best_hits = None, 0
    for tname, kws in TYPE_KEYWORDS.items():
        hits = sum(1 for kw in kws if kw in t)
        if hits > best_hits:
            best_type, best_hits = tname, hits
    return best_type


def canonicalize_type(t: Optional[str]) -> Optional[str]:
    if not t:
        return None
    t_low = t.strip().lower()
    if t_low in TYPE_KEYWORDS.keys():
        return t_low
    for canon, kws in TYPE_KEYWORDS.items():
        if t_low in (kw.lower() for kw in kws):
            return canon
    return None


def infer_type_from_hit(hit: Dict[str, Any]) -> Optional[str]:

    # Prefer explicit field if present
    val = hit.get(OBJECT_TYPE_FIELD)
    if isinstance(val, str) and val:
        return canonicalize_type(val)
    # Otherwise infer from title/description/spec
    title = hit.get(TITLE_FIELD, hit.get("title", ""))
    desc = hit.get(DESCRIPTION_FIELD, "")
    spec = hit.get(SPEC_TEXT_FIELD, "")
    for candidate in (title, desc, spec):
        t = infer_type_from_text(candidate)
        if t:
            return t
    return None


def infer_type_from_hits(hits: List[Dict[str, Any]], top_k: int = 20) -> Optional[str]:
    votes: Dict[str, float] = {}
    for h in hits[:top_k]:
        t = infer_type_from_hit(h)
        if not t:
            continue
        # score-weighted voting
        score = float(h.get('_score', 1.0))
        votes[t] = votes.get(t, 0.0) + max(0.5, score)
    if not votes:
        return None
    # return the type with the highest cumulative score
    return max(votes.items(), key=lambda kv: kv[1])[0]


def postfilter_hits_by_type(hits: List[Dict[str, Any]], target_type: str) -> List[Dict[str, Any]]:
    """If OBJECT_TYPE_FIELD is missing, filter by keywords in title/description/spec."""
    filtered = []
    kws = TYPE_KEYWORDS.get(target_type, [target_type])
    kws = [k.lower() for k in kws]
    for h in hits:
        blob = " ".join([
            normalize_text(h.get(TITLE_FIELD, h.get("title", ""))),
            normalize_text(h.get(DESCRIPTION_FIELD, "")),
            normalize_text(h.get(SPEC_TEXT_FIELD, "")),
        ])
        if any(kw in blob for kw in kws):
            filtered.append(h)
    return filtered

# =============================================================
# Marqo search (HTTP API)
# =============================================================

def marqo_search(q: str, limit: int = 200, filter_string: Optional[str] = None, attrs: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """
    Performs a tensor search on Marqo.
    q can be a text phrase or an image URL accessible to Marqo.
    """
    searchable_attributes = sanitize_attrs(attrs or [
        IMAGE_FIELD,
        TITLE_FIELD,
        DESCRIPTION_FIELD,
        SPEC_TEXT_FIELD,
        SEARCH_BLOB_FIELD,
    ])

    payload: Dict[str, Any] = {
        "limit": limit,
        "q": q,
        "searchMethod": "TENSOR",
        "searchableAttributes": searchable_attributes,
        "attributesToRetrieve": [
            "_id", TITLE_FIELD, IMAGE_FIELD, ALT_IMAGE_FIELD, DOM_COLOR_FIELD,
            SKU_FIELD, OBJECT_TYPE_FIELD, DESCRIPTION_FIELD, SPEC_TEXT_FIELD,
            SEARCH_BLOB_FIELD, CLICK_URL_FIELD
        ]
    }

    if filter_string:
        payload["filter"] = filter_string

    url = f"{MARQO_URL}/indexes/{INDEX_NAME}/search"
    try:
        resp = requests.post(url, json=payload, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API paieškos klaida: {e}")
        if getattr(e, 'response', None) is not None and e.response.text:
            with st.expander("📄 Serverio atsakymas"):
                st.code(e.response.text, language='json')
        return None


def fuse_hits(img_hits: List[Dict[str, Any]], txt_hits: List[Dict[str, Any]], alpha: float = 0.7) -> List[Dict[str, Any]]:
    """Weighted late-fusion of two hit lists by _id. alpha weighs image scores."""
    def to_map(hits):
        return {h.get('_id'): float(h.get('_score', 0.0)) for h in hits}
    imap, tmap = to_map(img_hits), to_map(txt_hits)
    ids = set(imap) | set(tmap)
    fused = []
    for _id in ids:
        s = alpha * imap.get(_id, 0.0) + (1 - alpha) * tmap.get(_id, 0.0)
        # pick a representative hit (prefer image list metadata)
        base = next((h for h in img_hits if h.get('_id') == _id), None) or next((h for h in txt_hits if h.get('_id') == _id), None)
        if not base:
            continue
        h = dict(base)
        h['_fused_score'] = s
        fused.append(h)
    fused.sort(key=lambda x: x.get('_fused_score', x.get('_score', 0.0)), reverse=True)
    return fused

# =============================================================
# UI
# =============================================================

st.set_page_config(page_title="Baldų paieška", layout="wide")
st.title("🛋️ Baldų ir interjero elementų paieška")
st.caption("Įkelkite produkto nuotrauką arba įveskite raktažodžius. Sistema ieškos tik tarp to paties tipo objektų (pvz., įkėlę lempos nuotrauką – ieškos lempų).")

# Session state
for key, default in (
    ('last_upload_hash', None), ('search_results', None), ('query_color', None),
    ('page', 0), ('detected_object_type', None), ('selected_object_type', None),
):
    if key not in st.session_state:
        st.session_state[key] = default

# Sidebar controls
st.sidebar.header("Paieškos nustatymai")
uploaded_file = st.sidebar.file_uploader(
    "Pasirinkite paveikslėlį",
    type=["jpg", "jpeg", "png", "gif", "bmp", "webp"],
    key="uploader"
)
search_query = st.sidebar.text_input("🔍 Ieškoti pagal tekstą:", "")
color_threshold = st.sidebar.slider("Spalvos panašumo riba", 0, 200, 50, 10)
use_color_filter = st.sidebar.checkbox("Įjungti spalvos filtravimą", value=True)

# Boosting sliders
visual_weight = 0.75  # fixed visual weight (removed UI slider)
text_weight = None
if search_query.strip():
    text_weight = st.sidebar.slider(
        "🔤 Teksto svoris (β)", 0.0, 1.0, 0.35, 0.05,
        help="Kiek pridėti vartotojo teksto paiešką prie rezultato (sujungiama su vaizdo paieška)."
    )


# -----------------------------
# Main logic
# -----------------------------

results_payload: Optional[Dict[str, Any]] = None

if uploaded_file:
    st.sidebar.image(uploaded_file, caption="Įkeltas paveikslėlis", width=180)
    img_bytes = uploaded_file.getvalue()
    current_hash = hash(img_bytes)

    if current_hash != st.session_state.last_upload_hash:
        st.session_state.last_upload_hash = current_hash
        st.session_state.page = 0

        # Upload query image to R2 for a stable, public URL
        try:
            query_url = upload_query_image_to_r2(img_bytes, uploaded_file.name)
        except Exception as e:
            st.error(f"Nepavyko įkelti į R2: {e}")
            st.stop()

        # Dominant color (for optional refinement)
        st.session_state.query_color = get_dominant_color(img_bytes)

        with st.spinner("Analizuojamas paveikslėlis ir nustatomas tipas..."):
            # Step 1: broad image search (no filter) to infer type from TOP-K
            probe = marqo_search(query_url, limit=200)
            hits = probe.get("hits", []) if probe else []
            inferred_raw = infer_type_from_hits(hits, top_k=30) or ""
            inferred = canonicalize_type(inferred_raw) or ""
            st.session_state.detected_object_type = inferred
            # Only set selected type if it is a valid canonical option
            if inferred in TYPE_KEYWORDS.keys() and (st.session_state.get('selected_object_type') in (None, "")):
                st.session_state.selected_object_type = inferred

            # Build a filter if the index stores OBJECT_TYPE_FIELD
            filter_query = None
            if inferred:
                # Try hard filter first (requires OBJECT_TYPE_FIELD in index)
                filter_query = f'{OBJECT_TYPE_FIELD}:"{inferred}"'

            # Step 2: run two searches for controllable boosting: visual-only and semantic (text fields)
            def run_dual_search(q_url: str, flt: Optional[str]):
                a = marqo_search(q_url, limit=200, filter_string=flt, attrs=VISUAL_ATTRS)
                b = marqo_search(q_url, limit=200, filter_string=flt, attrs=TEXT_ATTRS)
                return (a.get("hits", []) if a else [], b.get("hits", []) if b else [])

            img_vis_hits, img_sem_hits = run_dual_search(query_url, filter_query)

            # Fallback: if filtered search produced no hits, retry WITHOUT filter and post-filter
            if not img_vis_hits and not img_sem_hits:
                img_vis_hits, img_sem_hits = run_dual_search(query_url, None)
                if inferred:
                    img_vis_hits = postfilter_hits_by_type(img_vis_hits, inferred)
                    img_sem_hits = postfilter_hits_by_type(img_sem_hits, inferred)
            
            # If OBJECT_TYPE is missing from index, post-filter by keywords but don't over-filter
            if inferred:
                def safe_postfilter(lst):
                    fl = postfilter_hits_by_type(lst, inferred)
                    return fl if fl else lst  # keep originals if filtering removes everything
                if not img_vis_hits or (img_vis_hits and OBJECT_TYPE_FIELD not in img_vis_hits[0]):
                    img_vis_hits = safe_postfilter(img_vis_hits)
                if not img_sem_hits or (img_sem_hits and OBJECT_TYPE_FIELD not in img_sem_hits[0]):
                    img_sem_hits = safe_postfilter(img_sem_hits)
            if inferred:
                if not img_vis_hits or (img_vis_hits and OBJECT_TYPE_FIELD not in img_vis_hits[0]):
                    img_vis_hits = postfilter_hits_by_type(img_vis_hits, inferred)
                if not img_sem_hits or (img_sem_hits and OBJECT_TYPE_FIELD not in img_sem_hits[0]):
                    img_sem_hits = postfilter_hits_by_type(img_sem_hits, inferred)

            # Fuse the two lists with adjustable visual weight
            fused_img = fuse_hits(img_vis_hits, img_sem_hits, alpha=visual_weight)

            # Optional: if the user also typed text, do a text search (same filter) and fuse again
            if search_query.strip():
                txt_res = marqo_search(search_query.strip(), limit=200, filter_string=filter_query, attrs=TEXT_ATTRS)
                txt_hits = txt_res.get("hits", []) if txt_res else []
                if inferred and (not txt_hits or (txt_hits and OBJECT_TYPE_FIELD not in txt_hits[0])):
                    txt_hits = postfilter_hits_by_type(txt_hits, inferred)
                gamma = text_weight if text_weight is not None else 0.35
                results_payload = {"hits": fuse_hits(fused_img, txt_hits, alpha=1.0 - gamma)}
            else:
                results_payload = {"hits": fused_img}

elif search_query.strip():
    with st.spinner("Ieškoma pagal tekstą..."):
        # Text-only search (no image). Use manual type from sidebar if present.
        manual_t_raw = st.session_state.selected_object_type
        manual_t = canonicalize_type(manual_t_raw)
        filter_query = f'{OBJECT_TYPE_FIELD}:"{manual_t}"' if manual_t else None

        txt = marqo_search(search_query.strip(), limit=200, filter_string=filter_query, attrs=TEXT_ATTRS)
        hits = txt.get("hits", []) if txt else []

        # If filtered search empty, retry without filter and post-filter by manual type
        if (not hits) and manual_t:
            txt = marqo_search(search_query.strip(), limit=200, filter_string=None, attrs=TEXT_ATTRS)
            hits = txt.get("hits", []) if txt else []
            hits = postfilter_hits_by_type(hits, manual_t)

        results_payload = {"hits": hits}
else:
    results_payload = None
    st.session_state.page = 0

# =============================================================
# Render results
# =============================================================

if results_payload and results_payload.get("hits"):
    hits = list(results_payload["hits"])  # copy

    # Optional color filter (applies mainly to image queries)
    if uploaded_file and use_color_filter and st.session_state.query_color is not None:
        qcol = st.session_state.query_color
        original_hits = list(hits)
        filtered = []
        for h in hits:
            hcol_hex = h.get(DOM_COLOR_FIELD, "#000000")
            h_rgb = hex_to_rgb(hcol_hex)
            dist = color_distance(qcol, h_rgb)
            if dist <= color_threshold + 5:
                h['_adj_score'] = h.get('_fused_score', h.get('_score', 0.0)) - (dist / 441.0)
                filtered.append(h)
        if filtered:
            hits = filtered
        else:
            st.info("Spalvos filtras pašalino visus rezultatus — rodau be spalvų filtro.")
            hits = original_hits

    # Final sort by fused/score
    hits.sort(key=lambda h: h.get('_adj_score', h.get('_fused_score', h.get('_score', 0.0))), reverse=True)

    # Pagination
    page_size = 9
    total_pages = (len(hits) - 1) // page_size + 1 if hits else 0
    current_page = st.session_state.page

    st.subheader(f"Rasta rezultatų: {len(hits)}")

    # Show detected/selected type
    if st.session_state.selected_object_type:
        st.info(f"🔒 Paieška apribota tipui: **{st.session_state.selected_object_type}**")

    col_prev, col_pg, col_next = st.columns([2, 8, 2])
    with col_prev:
        if st.button("⬅ Ankstesnis", disabled=(current_page == 0)):
            st.session_state.page -= 1
            st.rerun()
    with col_pg:
        st.markdown(f"<div style='text-align:center;'>Puslapis {current_page + 1} iš {total_pages}</div>", unsafe_allow_html=True)
    with col_next:
        if st.button("Kitas ➡", disabled=(current_page >= total_pages - 1)):
            st.session_state.page += 1
            st.rerun()

    start = current_page * page_size
    end = start + page_size
    page_hits = hits[start:end]

    cols = st.columns(3)
    for i, h in enumerate(page_hits):
        with cols[i % 3]:
            img_url = h.get(IMAGE_FIELD) or h.get(ALT_IMAGE_FIELD) or h.get("image")
            title = h.get(TITLE_FIELD, h.get('title', 'Be pavadinimo'))
            _id = h.get('_id', 'Nėra')
            obj_t = h.get(OBJECT_TYPE_FIELD) or infer_type_from_hit(h) or "—"
            score = h.get('_fused_score', h.get('_score', None))
            click_url = h.get(CLICK_URL_FIELD)

            if img_url:
                st.image(img_url, use_container_width=True)
            st.write(f"**{title}**")
            st.caption(f"ID: {_id} · Tipas: {obj_t}")
            if isinstance(score, (int, float)):
                st.caption(f"Panašumas: {score:.3f}")
            if isinstance(click_url, str) and click_url:
                st.markdown(f"[🔗 Atidaryti produktą]({click_url})")
            st.markdown("---")

elif results_payload is not None:
    st.info("Rezultatų nerasta. Pabandykite kitą nuotrauką, pakeiskite tipą arba atlaisvinkite filtrus.")
else:
    st.info("Įkelkite paveikslėlį arba įveskite paieškos frazę.")
