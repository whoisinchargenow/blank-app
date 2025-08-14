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
    # Fireplace & accessories
    "andiron": ["andiron", "fire dog", "firedog"],
    "fire tools": ["fire tools", "fireplace tools", "fire poker", "fireplace set"],

    # Botanicals
    "artificial flowers & plants": ["artificial flowers", "faux flowers", "faux plants", "artificial plant", "silk flowers", "artificial greenery", "artificial palm"],

    # Small objects & decor
    "ashtray": ["ashtray"],
    "basket": ["basket", "wicker basket", "storage basket"],
    "bowl": ["bowl", "decorative bowl"],
    "box": ["box", "storage box", "decorative box"],
    "bust": ["bust", "sculpture bust", "head sculpture"],
    "candle holder": ["candle holder", "candlestick", "candelabra", "tealight holder", "tea light holder"],
    "decanter": ["decanter", "carafe", "wine decanter"],
    "desk accessory": ["desk accessory", "desk organizer", "pen holder", "paper tray"],
    "globe": ["globe", "world globe"],
    "hurricane": ["hurricane", "hurricane lamp", "hurricane candle holder"],
    "jar": ["jar", "storage jar", "ginger jar"],
    "lighter holder": ["lighter holder", "lighter case"],
    "object": ["object", "decorative object", "decor object"],
    "picture frame": ["picture frame", "photo frame"],
    "plaid": ["plaid", "throw", "blanket", "tartan throw"],
    "planter": ["planter", "flower pot", "plant pot", "cachepot"],
    "pouf": ["pouf", "pouffe"],
    "print": ["print", "art print", "poster"],
    "sculpture": ["sculpture", "statue", "figurine"],
    "stand": ["stand", "display stand", "pedestal stand"],
    "tray": ["tray"],
    "umbrella stand": ["umbrella stand"],
    "vase": ["vase"],
    "wall art": ["wall art", "wall print", "wall picture"],
    "wall decoration": ["wall decoration", "wall decor", "wall ornament"],
    "wine cooler": ["wine cooler", "wine chiller"],
    "wine rack": ["wine rack", "bottle rack"],

    # Furniture
    "bar": ["bar", "bar table", "bar furniture", "home bar", "bar cabinet", "bar counter"],
    "bed": ["bed", "bedframe", "bed frame"],
    "bench": ["bench", "entryway bench", "bedroom bench"],
    "cabinet": ["cabinet", "storage cabinet", "display cabinet"],
    "cake standard": ["cake standard", "cake stand"],
    "chair": ["chair", "armchair", "dining chair", "side chair"],
    "coatrack": ["coat rack", "coatrack", "coat stand", "hall stand"],
    "column": ["column", "pedestal"],
    "cushion": ["cushion", "pillow", "throw pillow"],
    "daybed": ["daybed", "chaise lounge", "chaise longue"],
    "desk": ["desk", "writing desk"],
    "dresser": ["dresser", "chest of drawers", "drawer chest"],
    "headboard": ["headboard", "bed headboard"],
    "lamp": ["lamp", "light", "wall lamp", "floor lamp", "table lamp", "pendant", "chandelier", "sconce", "wall light"],
    "lantern": ["lantern", "candle lantern"],
    "mirror": ["mirror", "wall mirror"],
    "nightstand": ["nightstand", "bedside table", "bedside cabinet"],
    "ottoman": ["ottoman", "footstool", "foot stool"],
    "rug": ["rug", "carpet"],
    "sofa": ["sofa", "couch", "loveseat", "sectional"],
    "stool": ["stool", "bar stool", "counter stool"],
    "table": ["table", "coffee table", "side table", "console table", "dining table"],
    "trolley": ["trolley", "bar cart", "serving cart"],
    "trunk": ["trunk", "storage trunk"],
    "tv cabinet": ["tv cabinet", "tv stand", "media console", "media unit"],
    "wall rack": ["wall rack", "wall shelf", "wall-mounted rack"],
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
# Marqo search (HTTP API)
# =============================================================
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
            SKU_FIELD, DESCRIPTION_FIELD, SPEC_TEXT_FIELD,
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
st.caption("Sistema pirmiausia ieško vizualiai panašių produktų. Tekstas – papildomas signalas.")

# Session state
for key, default in (
    ('last_upload_hash', None), ('search_results', None), ('query_color', None), ('base_hits', None), ('last_color_threshold', None),
    ('page', 0),
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

# Boosting weights
visual_weight = 0.85  # vizualinis signalas svarbiausias
text_weight = 0.35 if search_query.strip() else 0.0

results_payload: Optional[Dict[str, Any]] = None

if uploaded_file:
    st.sidebar.image(uploaded_file, caption="Įkeltas paveikslėlis", width=180)
    img_bytes = uploaded_file.getvalue()
    try:
        query_url = upload_query_image_to_r2(img_bytes, uploaded_file.name)
    except Exception as e:
        st.error(f"Nepavyko įkelti į R2: {e}")
        st.stop()

    st.session_state.query_color = get_dominant_color(img_bytes)

    with st.spinner("Ieškoma vizualiai panašių elementų..."):
        vis_res = marqo_search(query_url, limit=200, attrs=VISUAL_ATTRS)
        sem_res = marqo_search(query_url, limit=200, attrs=TEXT_ATTRS)
        vis_hits = vis_res.get("hits", []) if vis_res else []
        sem_hits = sem_res.get("hits", []) if sem_res else []
        fused_img = fuse_hits(vis_hits, sem_hits, alpha=visual_weight)

        if search_query.strip():
            txt_res = marqo_search(search_query.strip(), limit=200, attrs=TEXT_ATTRS)
            txt_hits = txt_res.get("hits", []) if txt_res else []
            final_hits = fuse_hits(fused_img, txt_hits, alpha=1.0 - text_weight) if text_weight > 0 else fused_img
        else:
            final_hits = fused_img

        st.session_state.base_hits = final_hits
        results_payload = {"hits": final_hits}

elif search_query.strip():
    with st.spinner("Ieškoma pagal tekstą..."):
        txt_res = marqo_search(search_query.strip(), limit=200, attrs=TEXT_ATTRS)
        txt_hits = txt_res.get("hits", []) if txt_res else []
        st.session_state.base_hits = txt_hits
        results_payload = {"hits": txt_hits}
else:
    results_payload = None
    st.session_state.base_hits = None
    st.session_state.query_color = None
    st.session_state.page = 0

# =============================================================
# Render results
# =============================================================

base_list = results_payload.get("hits") if results_payload else st.session_state.get("base_hits")
if base_list:
    hits = list(base_list)

    # Optional color filter (applies mainly to image queries)
    if uploaded_file and use_color_filter and st.session_state.query_color is not None:
        qcol = st.session_state.query_color
        original_hits = list(hits)
        matches, unknowns = [], []

        def _is_valid_hex(hx: str) -> bool:
            if not isinstance(hx, str):
                return False
            s = hx.lstrip('#')
            return len(s) == 6 and all(c in '0123456789abcdefABCDEF' for c in s)

        for h in hits:
            hx = h.get(DOM_COLOR_FIELD)
            if not _is_valid_hex(hx):
                unknowns.append(h)
                continue
            h_rgb = hex_to_rgb(hx)
            dist = color_distance(qcol, h_rgb)
            if dist <= color_threshold + 5:
                h['_adj_score'] = h.get('_fused_score', h.get('_score', 0.0)) - (dist / 441.0)
                matches.append(h)

        if matches:
            hits = matches + (unknowns if len(matches) < 5 else [])
        else:
            if unknowns:
                st.info("Dauguma įrašų neturi spalvos indekse — rodau be spalvų filtro.")
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
            score = h.get('_fused_score', h.get('_score', None))
            click_url = h.get(CLICK_URL_FIELD)

            if img_url:
                st.image(img_url, use_container_width=True)
            st.write(f"**{title}**")
            st.caption(f"ID: {_id}")
            if isinstance(score, (int, float)):
                st.caption(f"Panašumas: {score:.3f}")
            if isinstance(click_url, str) and click_url:
                st.markdown(f"[🔗 Atidaryti produktą]({click_url})")
            st.markdown('---')

elif results_payload is not None:
    st.info("Rezultatų nerasta. Pabandykite kitą paveikslėlį arba įvesti paieškos frazę.")
else:
    st.info("Įkelkite paveikslėlį arba įveskite paieškos frazę.")
