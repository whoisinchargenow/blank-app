import os
import io
import json
import uuid
import mimetypes
from typing import Any, Dict, Optional, List, Tuple

import requests
import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import boto3

# =============================================================
# Streamlit App — Visual-first Similarity Search (Marqo + R2)
# =============================================================
# - Prioritises visual similarity (image embeddings)
# - Optional text refinement (lexical)
# - Client-side colour filtering using per-product dominant colour (hex)
# - Robust query image colour estimation (object-focused)
# - No type-gating
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

TITLE_FIELD: str = _cfg("TITLE_FIELD", "name") or "name"  # must be a tensor field in index
IMAGE_FIELD: str = _cfg("IMAGE_FIELD", "image") or "image"
ALT_IMAGE_FIELD: str = _cfg("ALT_IMAGE_FIELD", "image_url") or "image_url"
DESCRIPTION_FIELD: str = _cfg("DESCRIPTION_FIELD", "description") or "description"
SEARCH_BLOB_FIELD: str = _cfg("SEARCH_BLOB_FIELD", "search_blob") or "search_blob"
DOM_COLOR_FIELD: str = _cfg("DOMINANT_COLOR_FIELD", "dominant_color") or "dominant_color"
SKU_FIELD: str = _cfg("SKU_FIELD", "product_id") or "product_id"
CLICK_URL_FIELD: str = _cfg("CLICK_URL_FIELD", "product_url") or "product_url"

# Cloudflare Access headers (if protected)
CF_ACCESS_CLIENT_ID = _cfg("CF_ACCESS_CLIENT_ID")
CF_ACCESS_CLIENT_SECRET = _cfg("CF_ACCESS_CLIENT_SECRET")
HEADERS: Dict[str, str] = {"Content-Type": "application/json"}
if CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET:
    HEADERS["CF-Access-Client-Id"] = CF_ACCESS_CLIENT_ID
    HEADERS["CF-Access-Client-Secret"] = CF_ACCESS_CLIENT_SECRET

# R2 Config
R2_ENDPOINT_URL: str = _cfg("R2_ENDPOINT_URL") or ""
R2_ACCESS_KEY_ID: Optional[str] = _cfg("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY: Optional[str] = _cfg("R2_SECRET_ACCESS_KEY")
R2_BUCKET: str = _cfg("R2_BUCKET", "streamlit098") or "streamlit098"
PUBLIC_BASE_URL: str = _cfg("PUBLIC_BASE_URL", "") or ""

# =============================================================
# Helpers
# =============================================================

# Known tensor fields present in your index (adjust if you add more)
KNOWN_TENSOR_FIELDS = {"image", "name", "description", "spec_text", "search_blob"}


# ---- Product type support (to narrow results) ----
TYPE_FIELDS = ["product_type", "type", "object_type", "category"]
TYPES_ALL = [
    "Visi",
    "Andiron",
    "Artificial Flowers & Plants",
    "Ashtray",
    "Bar",
    "Basket",
    "Bed",
    "Bench",
    "Bookend",
    "Bowl",
    "Box",
    "Bust",
    "Cabinet",
    "Cake Standard",
    "Candle Holder",
    "Chair",
    "Coatrack",
    "Column",
    "Cushion",
    "Daybed",
    "Decanter",
    "Desk",
    "Desk Accessory",
    "Dresser",
    "Fire Tools",
    "Folding Screen",
    "Globe",
    "Headboard",
    "Hurricane",
    "Jar",
    "Lamp",
    "Lantern",
    "Lighter holder",
    "Mirror",
    "Nightstand",
    "Object",
    "Ottoman",
    "Picture Frame",
    "Plaid",
    "Planter",
    "Pouf",
    "Print",
    "Rug",
    "Sculpture",
    "Sofa",
    "Stand",
    "Stool",
    "Table",
    "Tray",
    "Trolley",
    "Trunk",
    "TV Cabinet",
    "Umbrella Stand",
    "Vase",
    "Wall Art",
    "Wall Decoration",
    "Wall Rack",
    "Wine Cooler",
    "Wine Rack",
]

def filter_by_type(hits: List[Dict[str, Any]], selected_type: Optional[str]) -> List[Dict[str, Any]]:
    if not hits or not selected_type or selected_type == "Visi":
        return hits
    tsel = selected_type.lower()
    out: List[Dict[str, Any]] = []
    for h in hits:
        ty = get_hit_field(h, *TYPE_FIELDS) if 'get_hit_field' in globals() else (h.get('product_type') or h.get('type') or h.get('object_type') or h.get('category'))
        if isinstance(ty, str) and tsel in ty.lower():
            out.append(h)
            continue
        # Fallback to check common text fields
        name = (h.get('name') or h.get('title') or "").lower()
        desc = (h.get('description') or "").lower()
        blob = (h.get('search_blob') or "").lower()
        if tsel in name or tsel in desc or tsel in blob:
            out.append(h)
    return out

def get_hit_field(hit: Dict[str, Any], *names: str) -> Optional[Any]:
    """Safely get a field from different possible locations in Marqo hits.
    Checks top-level, then 'fields', then 'document'. Returns first non-empty value.
    """
    for n in names:
        if n in hit and hit.get(n) not in (None, ""):
            return hit.get(n)
    f = hit.get("fields") or {}
    for n in names:
        if n in f and f.get(n) not in (None, ""):
            return f.get(n)
    d = hit.get("document") or {}
    for n in names:
        if n in d and d.get(n) not in (None, ""):
            return d.get(n)
    return None


def sanitize_attrs(attrs: Optional[List[str]], *, for_method: str) -> List[str]:
    """Map aliases and drop unknown fields to avoid 400s from Marqo.
    - maps 'title' -> 'name'
    - removes attributes not in KNOWN_TENSOR_FIELDS
    - for LEXICAL, defaults to textual fields
    - for TENSOR image queries, you can pass [IMAGE_FIELD]
    """
    if attrs is None:
        if for_method.upper() == "LEXICAL":
            attrs = [TITLE_FIELD, DESCRIPTION_FIELD, "spec_text", SEARCH_BLOB_FIELD]
        else:
            attrs = [IMAGE_FIELD, TITLE_FIELD, DESCRIPTION_FIELD, "spec_text", SEARCH_BLOB_FIELD]
    clean: List[str] = []
    for a in attrs:
        if not a:
            continue
        a = a.strip()
        if a == "title":
            a = "name"
        if a in KNOWN_TENSOR_FIELDS and a not in clean:
            clean.append(a)
    if not clean:
        clean = [IMAGE_FIELD] if for_method.upper() == "TENSOR" else [TITLE_FIELD, DESCRIPTION_FIELD, "spec_text", SEARCH_BLOB_FIELD]
    return clean

# Known tensor fields present in your index (adjust if you add more)
KNOWN_TENSOR_FIELDS = {"image", "name", "description", "spec_text", "search_blob"}


def sanitize_attrs(attrs: Optional[List[str]], *, for_method: str) -> List[str]:
    """Map aliases and drop unknown fields to avoid 400s from Marqo.
    - maps 'title' -> 'name'
    - removes attributes not in KNOWN_TENSOR_FIELDS
    - for LEXICAL, defaults to textual fields
    - for TENSOR image queries, you can pass [IMAGE_FIELD]
    """
    if attrs is None:
        if for_method.upper() == "LEXICAL":
            attrs = [TITLE_FIELD, DESCRIPTION_FIELD, "spec_text", SEARCH_BLOB_FIELD]
        else:
            attrs = [IMAGE_FIELD, TITLE_FIELD, DESCRIPTION_FIELD, "spec_text", SEARCH_BLOB_FIELD]
    clean: List[str] = []
    for a in attrs:
        if not a:
            continue
        a = a.strip()
        if a == "title":
            a = "name"
        if a in KNOWN_TENSOR_FIELDS and a not in clean:
            clean.append(a)
    if not clean:
        clean = [IMAGE_FIELD] if for_method.upper() == "TENSOR" else [TITLE_FIELD, DESCRIPTION_FIELD, "spec_text", SEARCH_BLOB_FIELD]
    return clean

def to_hex(rgb: np.ndarray | Tuple[int, int, int]) -> str:
    r, g, b = [int(max(0, min(255, v))) for v in (rgb if isinstance(rgb, (list, tuple, np.ndarray)) else (0, 0, 0))]
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_rgb(hex_color: str) -> Optional[np.ndarray]:
    if not isinstance(hex_color, str):
        return None
    s = hex_color.lstrip('#')
    if len(s) != 6:
        return None
    try:
        return np.array([int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)], dtype=int)
    except ValueError:
        return None


def color_distance(c1: np.ndarray, c2: np.ndarray) -> float:
    return float(np.linalg.norm(c1.astype(float) - c2.astype(float)))

# -----------------------------
# Color-name → RGB mapping & parser (LT + EN)
# -----------------------------
COLOR_NAME_RGB: Dict[str, Tuple[int, int, int]] = {
    # Reds
    "raudona": (200, 30, 30), "raudonas": (200, 30, 30), "red": (200, 30, 30), "bordo": (128, 0, 32), "burgundy": (128, 0, 32),
    # Blues
    "mėlyna": (30, 60, 200), "melyna": (30, 60, 200), "blue": (30, 60, 200), "turkio": (64, 224, 208), "turquoise": (64, 224, 208),
    # Greens
    "žalia": (40, 160, 60), "zalia": (40, 160, 60), "green": (40, 160, 60), "mėtinė": (170, 240, 200), "metine": (170, 240, 200), "mint": (170, 240, 200),
    # Neutrals
    "juoda": (20, 20, 20), "black": (20, 20, 20), "balta": (245, 245, 245), "white": (245, 245, 245),
    "pilka": (128, 128, 128), "pilkas": (128, 128, 128), "grey": (128, 128, 128), "gray": (128, 128, 128),
    # Warm tones
    "geltona": (240, 220, 80), "yellow": (240, 220, 80), "oranžinė": (245, 150, 50), "oranzine": (245, 150, 50), "orange": (245, 150, 50),
    "ruda": (120, 80, 50), "brown": (120, 80, 50), "auksinė": (212, 175, 55), "auksine": (212, 175, 55), "gold": (212, 175, 55),
    "sidabrinė": (192, 192, 192), "sidabrine": (192, 192, 192), "silver": (192, 192, 192),
    # Others
    "rožinė": (255, 160, 180), "rozine": (255, 160, 180), "pink": (255, 160, 180),
    "violetinė": (150, 90, 200), "violetine": (150, 90, 200), "purple": (150, 90, 200),
    "smėlio": (222, 203, 178), "smelio": (222, 203, 178), "beige": (222, 203, 178), "kremas": (243, 229, 171), "cream": (243, 229, 171),
}


def parse_color_from_text(text: str) -> Tuple[Optional[np.ndarray], str]:
    """Extract the first color word (LT/EN) from the text and return (rgb, remaining_text)."""
    if not isinstance(text, str):
        return None, ""
    words = [w.strip().lower() for w in text.replace("/", " ").replace("-", " ").split() if w.strip()]
    color_rgb: Optional[np.ndarray] = None
    rest_words: List[str] = []
    for w in words:
        if color_rgb is None and w in COLOR_NAME_RGB:
            color_rgb = np.array(COLOR_NAME_RGB[w], dtype=int)
        else:
            rest_words.append(w)
    remaining = " ".join(rest_words)
    return color_rgb, remaining


# Robust, object-focused dominant colour for the **query image**
# (center-weight + saturation/value mask + relaxed fallbacks)

def get_dominant_color(image_bytes: bytes) -> Optional[np.ndarray]:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception:
        return None

    img = img.resize((160, 160))
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    hsv = np.asarray(img.convert('HSV'), dtype=np.float32) / 255.0
    s = hsv[..., 1]
    v = hsv[..., 2]

    H, W = s.shape
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    center_w = np.exp(-(((xx - cx) / (0.45 * W)) ** 2 + ((yy - cy) / (0.45 * H)) ** 2))

    def _kmeans_pick(samples: np.ndarray) -> Optional[np.ndarray]:
        if samples is None or samples.size == 0:
            return None
        k = 3 if samples.shape[0] >= 600 else 2
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=0)
            km.fit(samples)
            labels = km.labels_
            centers = km.cluster_centers_
            scores = []
            for ki in range(centers.shape[0]):
                idx = (labels == ki)
                if not np.any(idx):
                    scores.append(-1)
                    continue
                mean_rgb = samples[idx].mean(axis=0)
                mx, mn = float(np.max(mean_rgb)), float(np.min(mean_rgb))
                sat = 0.0 if mx <= 1e-6 else (mx - mn) / mx
                scores.append(idx.mean() * (sat + 0.05))
            best = int(np.argmax(scores))
            c = np.clip((centers[best] * 255.0).round().astype(int), 0, 255)
            return c
        except Exception:
            return None

    flat_rgb = rgb.reshape(-1, 3)

    # Pass 1: strict mask
    weights = center_w * (0.3 + 0.7 * s)
    mask = (s >= 0.22) & (v >= 0.10) & (v <= 0.95)
    w_flat = (weights * mask).reshape(-1)
    if np.any(w_flat > 0):
        thr = np.quantile(w_flat[w_flat > 0], 0.75)
        sel = flat_rgb[w_flat >= max(thr, 1e-6)]
        col = _kmeans_pick(sel)
        if col is not None:
            return col

    # Pass 2: relaxed mask
    mask2 = (s >= 0.05) & (v >= 0.08) & (v <= 0.98)
    w_flat2 = (center_w * (0.2 + 0.8 * s) * mask2).reshape(-1)
    if np.any(w_flat2 > 0):
        thr2 = np.quantile(w_flat2[w_flat2 > 0], 0.6)
        sel2 = flat_rgb[w_flat2 >= max(thr2, 1e-6)]
        col = _kmeans_pick(sel2)
        if col is not None:
            return col

    # Pass 3: edge mask on V channel
    gx = np.zeros_like(v)
    gy = np.zeros_like(v)
    gx[:, 1:-1] = np.abs(v[:, 2:] - v[:, :-2])
    gy[1:-1, :] = np.abs(v[2:, :] - v[:-2, :])
    edge = gx + gy
    if float(edge.max()) > 0:
        e = edge / (edge.max() + 1e-6)
        sel3 = flat_rgb[(e >= np.quantile(e, 0.6)).reshape(-1)]
        col = _kmeans_pick(sel3)
        if col is not None:
            return col

    # Pass 4: palette quantization
    try:
        pal = img.convert('P', palette=Image.ADAPTIVE, colors=6)
        palette = pal.getpalette()
        counts = pal.getcolors()
        if counts:
            counts.sort(reverse=True)
            for cnt, idx in counts:
                r, g, b = palette[idx*3: idx*3+3]
                if not ((r > 245 and g > 245 and b > 245) or (r < 10 and g < 10 and b < 10)):
                    return np.array([int(r), int(g), int(b)])
            r, g, b = palette[counts[0][1]*3: counts[0][1]*3+3]
            return np.array([int(r), int(g), int(b)])
    except Exception:
        pass

    # Pass 5: global mean
    mean = (rgb.mean(axis=(0, 1)) * 255.0).round().astype(int)
    return np.array([int(mean[0]), int(mean[1]), int(mean[2])])


# Marqo search (HTTP API)

def marqo_search(q: str, limit: int = 200, attrs: Optional[List[str]] = None, method: str = "TENSOR") -> Optional[Dict[str, Any]]:
    method = (method or "TENSOR").upper()
    searchable_attrs = sanitize_attrs(attrs, for_method=method)

    payload: Dict[str, Any] = {
        "limit": limit,
        "q": q,
        "searchMethod": method,
        "searchableAttributes": searchable_attrs,
        "attributesToRetrieve": [
            "_id", TITLE_FIELD, IMAGE_FIELD, ALT_IMAGE_FIELD,
            DOM_COLOR_FIELD, SKU_FIELD, "product_id", "sku", "SKU", CLICK_URL_FIELD,
            DESCRIPTION_FIELD, SEARCH_BLOB_FIELD,
            # include possible type fields for client-side narrowing
            "product_type", "type", "object_type", "category",
        ],
    }
    url = f"{MARQO_URL}/indexes/{INDEX_NAME}/search"
    try:
        resp = requests.post(url, json=payload, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        msg = e.response.text if getattr(e, "response", None) is not None else str(e)
        st.error("API paieškos klaida")
        with st.expander("📄 Serverio atsakymas"):
            st.code(msg, language="json")
        return None


# Weighted late fusion (image-first)

def fuse_hits(img_hits: List[Dict[str, Any]], txt_hits: List[Dict[str, Any]], alpha: float = 0.88) -> List[Dict[str, Any]]:
    def to_map(hits):
        return {h.get('_id'): float(h.get('_score', 0.0)) for h in hits}
    imap, tmap = to_map(img_hits), to_map(txt_hits)
    ids = set(imap) | set(tmap)
    fused: List[Dict[str, Any]] = []
    for _id in ids:
        s = alpha * imap.get(_id, 0.0) + (1 - alpha) * tmap.get(_id, 0.0)
        base = next((h for h in img_hits if h.get('_id') == _id), None) or next((h for h in txt_hits if h.get('_id') == _id), None)
        if not base:
            continue
        h = dict(base)
        h['_fused_score'] = s
        fused.append(h)
    fused.sort(key=lambda x: x.get('_fused_score', x.get('_score', 0.0)), reverse=True)
    return fused


# R2 upload

def upload_query_image_to_r2(img_bytes: bytes, filename: str) -> str:
    if not PUBLIC_BASE_URL:
        raise RuntimeError("PUBLIC_BASE_URL is not configured.")
    r2 = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    )
    ext = os.path.splitext(filename)[1].lower() or ".jpg"
    key = f"queries/{uuid.uuid4()}{ext}"
    content_type = mimetypes.guess_type(filename)[0] or "image/jpeg"
    r2.put_object(Bucket=R2_BUCKET, Key=key, Body=img_bytes, ContentType=content_type)
    return f"{PUBLIC_BASE_URL.rstrip('/')}/{key}"


# Pagination UI

def render_pagination(total_pages: int, current_page_zerobased: int):
    if total_pages <= 1:
        return
    max_pages_to_show = 10
    nav = st.container()
    cols = nav.columns([1, 1, *(2 for _ in range(max_pages_to_show)), 1, 1])

    def set_page(page_num):
        st.session_state.page = page_num

    is_first_page = current_page_zerobased == 0
    cols[0].button("«", on_click=set_page, args=[0], disabled=is_first_page, use_container_width=True)
    cols[1].button("‹", on_click=set_page, args=[current_page_zerobased - 1], disabled=is_first_page, use_container_width=True)

    half = max_pages_to_show // 2
    start_page = max(0, current_page_zerobased - half)
    end_page = min(total_pages, start_page + max_pages_to_show)
    if end_page - start_page < max_pages_to_show:
        start_page = max(0, end_page - max_pages_to_show)

    for i, page_num in enumerate(range(start_page, end_page)):
        is_current = page_num == current_page_zerobased
        cols[i + 2].button(
            f"{page_num + 1}",
            on_click=set_page,
            args=[page_num],
            disabled=is_current,
            type="primary" if is_current else "secondary",
            use_container_width=True,
        )

    is_last_page = current_page_zerobased == total_pages - 1
    cols[-2].button("›", on_click=set_page, args=[current_page_zerobased + 1], disabled=is_last_page, use_container_width=True)
    cols[-1].button("»", on_click=set_page, args=[total_pages - 1], disabled=is_last_page, use_container_width=True)


# =============================================================
# UI and Application Flow
# =============================================================

st.set_page_config(page_title="Baldų paieška", layout="wide")
st.title("🛋️ Baldų ir interjero elementų paieška")
st.caption("Sistema pirmiausia ieško vizualiai panašių produktų. Tekstas – papildomas signalas.")

# Session state defaults
for k, v in (
    ('page', 0),
    ('last_upload_hash', None),
    ('color_filter_hidden', False),
    ('color_controls_rerolled', False),
):
    if k not in st.session_state:
        st.session_state[k] = v

# Sidebar
st.sidebar.header("Paieškos nustatymai")
st.sidebar.markdown("**📤 Įkelkite paveikslėlį**")
st.sidebar.caption("Vilkite ir numeskite arba pasirinkite failą iš kompiuterio.")
uploaded_file = st.sidebar.file_uploader(
    "Įkelkite paveikslėlį",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
    help=(
        "Nuotrauka naudojama vizualinei paieškai — surandami vaizdu panašūs produktai. "
        "Jei įvesite tekstą, jis dar labiau susiaurins rezultatus (pvz., modelis ar medžiaga). "
        "Įjungus spalvų filtrą, rezultatai papildomai filtruojami pagal objekto spalvą jūsų nuotraukoje. "
        "Leidžiami formatai: JPG, JPEG, PNG, WEBP."
    ),
)
# Išvalyti teksto paieškos lauką kiekvieno naujo paveikslėlio įkėlimo metu (prieš kuriant valdiklį)
if uploaded_file:
    try:
        _hash_for_text = hash(uploaded_file.getvalue())
        if st.session_state.get('last_upload_hash_textclear') != _hash_for_text:
            st.session_state['last_upload_hash_textclear'] = _hash_for_text
            st.session_state['search_text'] = ""
    except Exception:
        pass

search_query = st.sidebar.text_input(
    "🔍 Ieškoti pagal tekstą",
    help=(
        "Įveskite produkto pavadinimą, modelį ar frazę (pvz., ‘raudona sofa’). "
        "Paieška vertina visos frazės prasmę (semantiškai), todėl ‘raudona sofa’ ieškos būtent raudonų sofų. "
        "Jei įkelta nuotrauka, tekstas susiaurina vizualiai rastus rezultatus; jei nuotraukos nėra – ieško tik pagal tekstą."
    ),
key="search_text",
)

# Optional: narrow by product type
selected_type = st.sidebar.selectbox(
    "Produkto tipas (pasirinktinai)", TYPES_ALL, index=0,
    help="Apribokite rezultatus iki pasirinktos kategorijos.",
)

# Tekstinės paieškos atveju rodyti spalvų parinkiklį tik kai įvestas tekstas
text_use_color_picker = False
text_picker_hex: Optional[str] = None
if (not uploaded_file) and search_query.strip():
    text_use_color_picker = st.sidebar.checkbox(
        "Filtruoti pagal pasirinktą spalvą",
        value=False,
        help="Jei įjungta, rezultatai bus filtruojami pagal žemiau pasirinktą spalvą.",
    )
    text_picker_hex = st.sidebar.color_picker(
        "🎨 Pasirinkite spalvą",
        value="#d9bc92",
        help="Taikoma tik tekstinei paieškai. Parinkta spalva naudojama atrinkti vizualiai panašius atspalvius.",
    )
    # Spalvos panašumo slankiklis (tik tekstinei paieškai)
    st.sidebar.slider(
        "Spalvos panašumo riba",
        0, 150,
        st.session_state.get('text_color_threshold', 60), 10,
        key="text_color_threshold",
        help=(
            "Kuo mažesnė reikšmė, tuo griežčiau parenkamos tik labai panašios spalvos. "
            "Didesnė reikšmė leidžia platesnį atspalvių diapazoną."
        ),
    )

# --- Early reset: if a NEW image is uploaded, unhide colour controls BEFORE rendering them
is_new_upload = False
if uploaded_file:
    try:
        _img_preview_bytes = uploaded_file.getvalue()
        _img_hash = hash(_img_preview_bytes)
        prev_hash = st.session_state.get('last_upload_hash')
        is_new_upload = (_img_hash != prev_hash)
        if is_new_upload:
            st.session_state['last_upload_hash'] = _img_hash
            st.session_state['color_filter_hidden'] = False
            st.session_state['color_controls_rerolled'] = False
    except Exception:
        is_new_upload = False
else:
    is_new_upload = False

# Colour controls: visible only when an image is uploaded
if uploaded_file:
    if st.session_state.get('color_filter_hidden', False):
        use_color_filter = False
        color_threshold = 50
    else:
        # Default ON on a brand new upload
        if is_new_upload:
            st.session_state['color_filter_checked'] = True
        use_color_filter = st.sidebar.checkbox(
            "Įjungti spalvų filtravimą",
            value=st.session_state.get('color_filter_checked', True),
            key="color_filter_checked",
        )
        color_threshold = st.sidebar.slider(
            "Spalvos panašumo riba", 0, 150,
            st.session_state.get('color_threshold', 50), 10,
            key="color_threshold",
            help=(
                "Nustato, kiek artima turi būti katalogo prekės spalva jūsų nuotraukai. "
                "Mažesnė reikšmė – griežtesnis atitikimas; didesnė – leidžia daugiau atspalvių."
            ),
        )
else:
    # No image uploaded: hide and turn off colour filter
    use_color_filter = False
    color_threshold = 50

final_hits: List[Dict[str, Any]] = []

# --- Main Logic Branch: Image Search (+ optional text) ---
if uploaded_file:
    st.sidebar.image(uploaded_file, caption="Įkeltas paveikslėlis", width=180)
    img_bytes = uploaded_file.getvalue()
    current_hash = hash(img_bytes)

    # New image resets colour-control hiding
    if st.session_state.last_upload_hash != current_hash:
        st.session_state.last_upload_hash = current_hash
        st.session_state.color_filter_hidden = False
        st.session_state.color_controls_rerolled = False

    query_rgb = get_dominant_color(img_bytes) if use_color_filter else None

    with st.spinner("Ieškoma vizualiai panašių elementų..."):
        try:
            query_url = upload_query_image_to_r2(img_bytes, uploaded_file.name)
        except Exception as e:
            st.error(f"Nepavyko įkelti paveikslėlio: {e}")
            st.stop()

        # Visual-only and text-fields tensor searches (no server-side colour filter)
        vis_res = marqo_search(query_url, attrs=[IMAGE_FIELD], method="TENSOR")
        sem_res = marqo_search(query_url, attrs=[TITLE_FIELD, DESCRIPTION_FIELD, "spec_text", SEARCH_BLOB_FIELD], method="TENSOR")

        vis_hits = vis_res.get("hits", []) if vis_res else []
        sem_hits = sem_res.get("hits", []) if sem_res else []
        image_search_results = fuse_hits(vis_hits, sem_hits, alpha=0.88)

        if search_query.strip():
            # Parse color from text; if provided, use it to override image color filter
            color_rgb_text, rest_text = parse_color_from_text(search_query)
            if color_rgb_text is not None:
                query_rgb = color_rgb_text
            qtext = rest_text.strip() if rest_text else search_query.strip()
            txt_res = marqo_search(q=qtext, limit=1000, attrs=[TITLE_FIELD, DESCRIPTION_FIELD, "spec_text", SEARCH_BLOB_FIELD], method="TENSOR")
            text_hits = txt_res.get("hits", []) if txt_res else []
            text_ids = {h.get('_id') for h in text_hits}
            final_hits = [h for h in image_search_results if h.get('_id') in text_ids]
        else:
            final_hits = image_search_results

    # Client-side colour filtering using per-item hex colour
    if use_color_filter and query_rgb is not None and final_hits:
        kept, unknowns = [], []
        for h in final_hits:
            hx = h.get(DOM_COLOR_FIELD)
            rgb = hex_to_rgb(hx)
            if rgb is None:
                unknowns.append(h)
                continue
            if color_distance(query_rgb, rgb) <= float(color_threshold):
                kept.append(h)

        if kept:
            # Keep unknowns if very few matches to avoid over-pruning
            final_hits = kept + (unknowns if len(kept) < 5 else [])
        else:
            # No colour matches: don't hide controls; just show info and fall back
            st.info("Dauguma įrašų neturi spalvos indekse — rodau be spalvų filtro.")
            final_hits = image_search_results

# --- Main Logic Branch: Text-Only Search ---
elif search_query.strip():
    # Text-only search: keep colour controls hidden & off
    st.session_state.color_filter_hidden = True
    st.session_state.color_controls_rerolled = False

    with st.spinner("Ieškoma pagal tekstą..."):
        color_rgb_text, rest_text = parse_color_from_text(search_query)
        # Jei naudotojas pasirinko spalvą parinkiklyje – ji turi pirmenybę
        if text_use_color_picker and isinstance(text_picker_hex, str):
            _picked = hex_to_rgb(text_picker_hex)
            if _picked is not None:
                color_rgb_text = _picked

        qtext = (rest_text.strip() if rest_text else "baldai")  # neutral fallback to fetch a broad set
        txt_res = marqo_search(q=qtext, limit=1000, attrs=[TITLE_FIELD, DESCRIPTION_FIELD, "spec_text", SEARCH_BLOB_FIELD], method="TENSOR")
        final_hits = txt_res.get("hits", []) if txt_res else []

        # Apply colour filter (from text or picker), if any
        if color_rgb_text is not None and final_hits:
            threshold = float(st.session_state.get('text_color_threshold', 60))
            kept, unknowns = [], []
            for h in final_hits:
                hx = get_hit_field(h, DOM_COLOR_FIELD, 'dominant_color')
                rgb = hex_to_rgb(hx) if isinstance(hx, str) else None
                if rgb is None:
                    unknowns.append(h)
                    continue
                if color_distance(color_rgb_text, rgb) <= threshold:
                    kept.append(h)
            final_hits = kept if kept else final_hits

# --- Initial State ---
else:
    # Initial state: no image — hide colour controls
    st.session_state.color_filter_hidden = True
    st.info("Įkelkite paveikslėlį arba įveskite paieškos frazę.")

# Before rendering, apply optional type narrowing
final_hits = filter_by_type(final_hits, selected_type)

# =============================================================
# Render Results
# =============================================================

if final_hits:
    # Reset pagination when result size changes
    if "last_hit_count" not in st.session_state or st.session_state.last_hit_count != len(final_hits):
        st.session_state.page = 0
    st.session_state.last_hit_count = len(final_hits)

    st.subheader(f"Rasta rezultatų: {len(final_hits)}")
    page_size = 50  # 5 per row × 10 rows per page
    total_pages = (len(final_hits) - 1) // page_size + 1
    current_page = st.session_state.page

    render_pagination(total_pages, current_page)

    start_idx = current_page * page_size
    end_idx = start_idx + page_size
    page_hits = final_hits[start_idx:end_idx]

    cols = st.columns(5)
    for i, h in enumerate(page_hits):
        with cols[i % 5]:
            img_url = get_hit_field(h, IMAGE_FIELD, ALT_IMAGE_FIELD, "image")
            title = get_hit_field(h, TITLE_FIELD, 'title') or 'Be pavadinimo'
            sku = get_hit_field(h, SKU_FIELD, 'product_id', 'sku', 'SKU') or h.get('_id')
            click_url = get_hit_field(h, CLICK_URL_FIELD, 'product_url')
            dom_color_hex = get_hit_field(h, DOM_COLOR_FIELD, 'dominant_color')
            score = h.get('_fused_score', h.get('_score', None))

            if img_url and click_url:
                alt_txt = f"{title} · SKU {sku}" if sku else title
                st.markdown(
                    f'<a href="{click_url}" target="_blank" rel="noopener noreferrer" title="Atidaryti produktą">'
                    f'<img src="{img_url}" alt="{alt_txt}" style="width:100%;border-radius:12px;display:block;"/></a>',
                    unsafe_allow_html=True,
                )
            elif img_url:
                st.image(img_url, use_container_width=True)
            if sku:
                st.write(f"**{title}** · `{sku}`")
            else:
                st.write(f"**{title}**")
            if isinstance(score, (int, float)):
                st.caption(f"Panašumas: {score:.3f}")
            if isinstance(dom_color_hex, str) and len(dom_color_hex) >= 4:
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px">'
                    f'<div style="width:20px;height:20px;background:{dom_color_hex};border:1px solid #ccc;border-radius:4px"></div>'
                    f'<span>{dom_color_hex}</span></div>',
                    unsafe_allow_html=True,
                )
            st.markdown('---')

elif uploaded_file or search_query:
    st.warning("Rezultatų nerasta. Pabandykite pakoreguoti užklausą ar filtrus.")
import os
import io
import json
import uuid
import mimetypes
from typing import Any, Dict, Optional, List, Tuple

import requests
import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import boto3

# =============================================================
# Streamlit App — Visual-first Similarity Search (Marqo + R2)
# =============================================================
# - Prioritises visual similarity (image embeddings)
# - Optional text refinement (lexical)
# - Client-side colour filtering using per-product dominant colour (hex)
# - Robust query image colour estimation (object-focused)
# - No type-gating
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

TITLE_FIELD: str = _cfg("TITLE_FIELD", "name") or "name"  # must be a tensor field in index
IMAGE_FIELD: str = _cfg("IMAGE_FIELD", "image") or "image"
ALT_IMAGE_FIELD: str = _cfg("ALT_IMAGE_FIELD", "image_url") or "image_url"
DESCRIPTION_FIELD: str = _cfg("DESCRIPTION_FIELD", "description") or "description"
SEARCH_BLOB_FIELD: str = _cfg("SEARCH_BLOB_FIELD", "search_blob") or "search_blob"
DOM_COLOR_FIELD: str = _cfg("DOMINANT_COLOR_FIELD", "dominant_color") or "dominant_color"
SKU_FIELD: str = _cfg("SKU_FIELD", "product_id") or "product_id"
CLICK_URL_FIELD: str = _cfg("CLICK_URL_FIELD", "product_url") or "product_url"

# Cloudflare Access headers (if protected)
CF_ACCESS_CLIENT_ID = _cfg("CF_ACCESS_CLIENT_ID")
CF_ACCESS_CLIENT_SECRET = _cfg("CF_ACCESS_CLIENT_SECRET")
HEADERS: Dict[str, str] = {"Content-Type": "application/json"}
if CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET:
    HEADERS["CF-Access-Client-Id"] = CF_ACCESS_CLIENT_ID
    HEADERS["CF-Access-Client-Secret"] = CF_ACCESS_CLIENT_SECRET

# R2 Config
R2_ENDPOINT_URL: str = _cfg("R2_ENDPOINT_URL") or ""
R2_ACCESS_KEY_ID: Optional[str] = _cfg("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY: Optional[str] = _cfg("R2_SECRET_ACCESS_KEY")
R2_BUCKET: str = _cfg("R2_BUCKET", "streamlit098") or "streamlit098"
PUBLIC_BASE_URL: str = _cfg("PUBLIC_BASE_URL", "") or ""

# =============================================================
# Helpers
# =============================================================

# Known tensor fields present in your index (adjust if you add more)
KNOWN_TENSOR_FIELDS = {"image", "name", "description", "spec_text", "search_blob"}


# ---- Product type support (to narrow results) ----
TYPE_FIELDS = ["product_type", "type", "object_type", "category"]
TYPES_ALL = [
    "Visi",
    "Andiron",
    "Artificial Flowers & Plants",
    "Ashtray",
    "Bar",
    "Basket",
    "Bed",
    "Bench",
    "Bookend",
    "Bowl",
    "Box",
    "Bust",
    "Cabinet",
    "Cake Standard",
    "Candle Holder",
    "Chair",
    "Coatrack",
    "Column",
    "Cushion",
    "Daybed",
    "Decanter",
    "Desk",
    "Desk Accessory",
    "Dresser",
    "Fire Tools",
    "Folding Screen",
    "Globe",
    "Headboard",
    "Hurricane",
    "Jar",
    "Lamp",
    "Lantern",
    "Lighter holder",
    "Mirror",
    "Nightstand",
    "Object",
    "Ottoman",
    "Picture Frame",
    "Plaid",
    "Planter",
    "Pouf",
    "Print",
    "Rug",
    "Sculpture",
    "Sofa",
    "Stand",
    "Stool",
    "Table",
    "Tray",
    "Trolley",
    "Trunk",
    "TV Cabinet",
    "Umbrella Stand",
    "Vase",
    "Wall Art",
    "Wall Decoration",
    "Wall Rack",
    "Wine Cooler",
    "Wine Rack",
]

def filter_by_type(hits: List[Dict[str, Any]], selected_type: Optional[str]) -> List[Dict[str, Any]]:
    if not hits or not selected_type or selected_type == "Visi":
        return hits
    tsel = selected_type.lower()
    out: List[Dict[str, Any]] = []
    for h in hits:
        ty = get_hit_field(h, *TYPE_FIELDS) if 'get_hit_field' in globals() else (h.get('product_type') or h.get('type') or h.get('object_type') or h.get('category'))
        if isinstance(ty, str) and tsel in ty.lower():
            out.append(h)
            continue
        # Fallback to check common text fields
        name = (h.get('name') or h.get('title') or "").lower()
        desc = (h.get('description') or "").lower()
        blob = (h.get('search_blob') or "").lower()
        if tsel in name or tsel in desc or tsel in blob:
            out.append(h)
    return out

def get_hit_field(hit: Dict[str, Any], *names: str) -> Optional[Any]:
    """Safely get a field from different possible locations in Marqo hits.
    Checks top-level, then 'fields', then 'document'. Returns first non-empty value.
    """
    for n in names:
        if n in hit and hit.get(n) not in (None, ""):
            return hit.get(n)
    f = hit.get("fields") or {}
    for n in names:
        if n in f and f.get(n) not in (None, ""):
            return f.get(n)
    d = hit.get("document") or {}
    for n in names:
        if n in d and d.get(n) not in (None, ""):
            return d.get(n)
    return None


def sanitize_attrs(attrs: Optional[List[str]], *, for_method: str) -> List[str]:
    """Map aliases and drop unknown fields to avoid 400s from Marqo.
    - maps 'title' -> 'name'
    - removes attributes not in KNOWN_TENSOR_FIELDS
    - for LEXICAL, defaults to textual fields
    - for TENSOR image queries, you can pass [IMAGE_FIELD]
    """
    if attrs is None:
        if for_method.upper() == "LEXICAL":
            attrs = [TITLE_FIELD, DESCRIPTION_FIELD, "spec_text", SEARCH_BLOB_FIELD]
        else:
            attrs = [IMAGE_FIELD, TITLE_FIELD, DESCRIPTION_FIELD, "spec_text", SEARCH_BLOB_FIELD]
    clean: List[str] = []
    for a in attrs:
        if not a:
            continue
        a = a.strip()
        if a == "title":
            a = "name"
        if a in KNOWN_TENSOR_FIELDS and a not in clean:
            clean.append(a)
    if not clean:
        clean = [IMAGE_FIELD] if for_method.upper() == "TENSOR" else [TITLE_FIELD, DESCRIPTION_FIELD, "spec_text", SEARCH_BLOB_FIELD]
    return clean

# Known tensor fields present in your index (adjust if you add more)
KNOWN_TENSOR_FIELDS = {"image", "name", "description", "spec_text", "search_blob"}


def sanitize_attrs(attrs: Optional[List[str]], *, for_method: str) -> List[str]:
    """Map aliases and drop unknown fields to avoid 400s from Marqo.
    - maps 'title' -> 'name'
    - removes attributes not in KNOWN_TENSOR_FIELDS
    - for LEXICAL, defaults to textual fields
    - for TENSOR image queries, you can pass [IMAGE_FIELD]
    """
    if attrs is None:
        if for_method.upper() == "LEXICAL":
            attrs = [TITLE_FIELD, DESCRIPTION_FIELD, "spec_text", SEARCH_BLOB_FIELD]
        else:
            attrs = [IMAGE_FIELD, TITLE_FIELD, DESCRIPTION_FIELD, "spec_text", SEARCH_BLOB_FIELD]
    clean: List[str] = []
    for a in attrs:
        if not a:
            continue
        a = a.strip()
        if a == "title":
            a = "name"
        if a in KNOWN_TENSOR_FIELDS and a not in clean:
            clean.append(a)
    if not clean:
        clean = [IMAGE_FIELD] if for_method.upper() == "TENSOR" else [TITLE_FIELD, DESCRIPTION_FIELD, "spec_text", SEARCH_BLOB_FIELD]
    return clean

def to_hex(rgb: np.ndarray | Tuple[int, int, int]) -> str:
    r, g, b = [int(max(0, min(255, v))) for v in (rgb if isinstance(rgb, (list, tuple, np.ndarray)) else (0, 0, 0))]
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_rgb(hex_color: str) -> Optional[np.ndarray]:
    if not isinstance(hex_color, str):
        return None
    s = hex_color.lstrip('#')
    if len(s) != 6:
        return None
    try:
        return np.array([int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)], dtype=int)
    except ValueError:
        return None


def color_distance(c1: np.ndarray, c2: np.ndarray) -> float:
    return float(np.linalg.norm(c1.astype(float) - c2.astype(float)))

# -----------------------------
# Color-name → RGB mapping & parser (LT + EN)
# -----------------------------
COLOR_NAME_RGB: Dict[str, Tuple[int, int, int]] = {
    # Reds
    "raudona": (200, 30, 30), "raudonas": (200, 30, 30), "red": (200, 30, 30), "bordo": (128, 0, 32), "burgundy": (128, 0, 32),
    # Blues
    "mėlyna": (30, 60, 200), "melyna": (30, 60, 200), "blue": (30, 60, 200), "turkio": (64, 224, 208), "turquoise": (64, 224, 208),
    # Greens
    "žalia": (40, 160, 60), "zalia": (40, 160, 60), "green": (40, 160, 60), "mėtinė": (170, 240, 200), "metine": (170, 240, 200), "mint": (170, 240, 200),
    # Neutrals
    "juoda": (20, 20, 20), "black": (20, 20, 20), "balta": (245, 245, 245), "white": (245, 245, 245),
    "pilka": (128, 128, 128), "pilkas": (128, 128, 128), "grey": (128, 128, 128), "gray": (128, 128, 128),
    # Warm tones
    "geltona": (240, 220, 80), "yellow": (240, 220, 80), "oranžinė": (245, 150, 50), "oranzine": (245, 150, 50), "orange": (245, 150, 50),
    "ruda": (120, 80, 50), "brown": (120, 80, 50), "auksinė": (212, 175, 55), "auksine": (212, 175, 55), "gold": (212, 175, 55),
    "sidabrinė": (192, 192, 192), "sidabrine": (192, 192, 192), "silver": (192, 192, 192),
    # Others
    "rožinė": (255, 160, 180), "rozine": (255, 160, 180), "pink": (255, 160, 180),
    "violetinė": (150, 90, 200), "violetine": (150, 90, 200), "purple": (150, 90, 200),
    "smėlio": (222, 203, 178), "smelio": (222, 203, 178), "beige": (222, 203, 178), "kremas": (243, 229, 171), "cream": (243, 229, 171),
}


def parse_color_from_text(text: str) -> Tuple[Optional[np.ndarray], str]:
    """Extract the first color word (LT/EN) from the text and return (rgb, remaining_text)."""
    if not isinstance(text, str):
        return None, ""
    words = [w.strip().lower() for w in text.replace("/", " ").replace("-", " ").split() if w.strip()]
    color_rgb: Optional[np.ndarray] = None
    rest_words: List[str] = []
    for w in words:
        if color_rgb is None and w in COLOR_NAME_RGB:
            color_rgb = np.array(COLOR_NAME_RGB[w], dtype=int)
        else:
            rest_words.append(w)
    remaining = " ".join(rest_words)
    return color_rgb, remaining


# Robust, object-focused dominant colour for the **query image**
# (center-weight + saturation/value mask + relaxed fallbacks)

def get_dominant_color(image_bytes: bytes) -> Optional[np.ndarray]:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception:
        return None

    img = img.resize((160, 160))
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    hsv = np.asarray(img.convert('HSV'), dtype=np.float32) / 255.0
    s = hsv[..., 1]
    v = hsv[..., 2]

    H, W = s.shape
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    center_w = np.exp(-(((xx - cx) / (0.45 * W)) ** 2 + ((yy - cy) / (0.45 * H)) ** 2))

    def _kmeans_pick(samples: np.ndarray) -> Optional[np.ndarray]:
        if samples is None or samples.size == 0:
            return None
        k = 3 if samples.shape[0] >= 600 else 2
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=0)
            km.fit(samples)
            labels = km.labels_
            centers = km.cluster_centers_
            scores = []
            for ki in range(centers.shape[0]):
                idx = (labels == ki)
                if not np.any(idx):
                    scores.append(-1)
                    continue
                mean_rgb = samples[idx].mean(axis=0)
                mx, mn = float(np.max(mean_rgb)), float(np.min(mean_rgb))
                sat = 0.0 if mx <= 1e-6 else (mx - mn) / mx
                scores.append(idx.mean() * (sat + 0.05))
            best = int(np.argmax(scores))
            c = np.clip((centers[best] * 255.0).round().astype(int), 0, 255)
            return c
        except Exception:
            return None

    flat_rgb = rgb.reshape(-1, 3)

    # Pass 1: strict mask
    weights = center_w * (0.3 + 0.7 * s)
    mask = (s >= 0.22) & (v >= 0.10) & (v <= 0.95)
    w_flat = (weights * mask).reshape(-1)
    if np.any(w_flat > 0):
        thr = np.quantile(w_flat[w_flat > 0], 0.75)
        sel = flat_rgb[w_flat >= max(thr, 1e-6)]
        col = _kmeans_pick(sel)
        if col is not None:
            return col

    # Pass 2: relaxed mask
    mask2 = (s >= 0.05) & (v >= 0.08) & (v <= 0.98)
    w_flat2 = (center_w * (0.2 + 0.8 * s) * mask2).reshape(-1)
    if np.any(w_flat2 > 0):
        thr2 = np.quantile(w_flat2[w_flat2 > 0], 0.6)
        sel2 = flat_rgb[w_flat2 >= max(thr2, 1e-6)]
        col = _kmeans_pick(sel2)
        if col is not None:
            return col

    # Pass 3: edge mask on V channel
    gx = np.zeros_like(v)
    gy = np.zeros_like(v)
    gx[:, 1:-1] = np.abs(v[:, 2:] - v[:, :-2])
    gy[1:-1, :] = np.abs(v[2:, :] - v[:-2, :])
    edge = gx + gy
    if float(edge.max()) > 0:
        e = edge / (edge.max() + 1e-6)
        sel3 = flat_rgb[(e >= np.quantile(e, 0.6)).reshape(-1)]
        col = _kmeans_pick(sel3)
        if col is not None:
            return col

    # Pass 4: palette quantization
    try:
        pal = img.convert('P', palette=Image.ADAPTIVE, colors=6)
        palette = pal.getpalette()
        counts = pal.getcolors()
        if counts:
            counts.sort(reverse=True)
            for cnt, idx in counts:
                r, g, b = palette[idx*3: idx*3+3]
                if not ((r > 245 and g > 245 and b > 245) or (r < 10 and g < 10 and b < 10)):
                    return np.array([int(r), int(g), int(b)])
            r, g, b = palette[counts[0][1]*3: counts[0][1]*3+3]
            return np.array([int(r), int(g), int(b)])
    except Exception:
        pass

    # Pass 5: global mean
    mean = (rgb.mean(axis=(0, 1)) * 255.0).round().astype(int)
    return np.array([int(mean[0]), int(mean[1]), int(mean[2])])


# Marqo search (HTTP API)

def marqo_search(q: str, limit: int = 200, attrs: Optional[List[str]] = None, method: str = "TENSOR") -> Optional[Dict[str, Any]]:
    method = (method or "TENSOR").upper()
    searchable_attrs = sanitize_attrs(attrs, for_method=method)

    payload: Dict[str, Any] = {
        "limit": limit,
        "q": q,
        "searchMethod": method,
        "searchableAttributes": searchable_attrs,
        "attributesToRetrieve": [
            "_id", TITLE_FIELD, IMAGE_FIELD, ALT_IMAGE_FIELD,
            DOM_COLOR_FIELD, SKU_FIELD, "product_id", "sku", "SKU", CLICK_URL_FIELD,
            DESCRIPTION_FIELD, SEARCH_BLOB_FIELD,
            # include possible type fields for client-side narrowing
            "product_type", "type", "object_type", "category",
        ],
    }
    url = f"{MARQO_URL}/indexes/{INDEX_NAME}/search"
    try:
        resp = requests.post(url, json=payload, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        msg = e.response.text if getattr(e, "response", None) is not None else str(e)
        st.error("API paieškos klaida")
        with st.expander("📄 Serverio atsakymas"):
            st.code(msg, language="json")
        return None


# Weighted late fusion (image-first)

def fuse_hits(img_hits: List[Dict[str, Any]], txt_hits: List[Dict[str, Any]], alpha: float = 0.88) -> List[Dict[str, Any]]:
    def to_map(hits):
        return {h.get('_id'): float(h.get('_score', 0.0)) for h in hits}
    imap, tmap = to_map(img_hits), to_map(txt_hits)
    ids = set(imap) | set(tmap)
    fused: List[Dict[str, Any]] = []
    for _id in ids:
        s = alpha * imap.get(_id, 0.0) + (1 - alpha) * tmap.get(_id, 0.0)
        base = next((h for h in img_hits if h.get('_id') == _id), None) or next((h for h in txt_hits if h.get('_id') == _id), None)
        if not base:
            continue
        h = dict(base)
        h['_fused_score'] = s
        fused.append(h)
    fused.sort(key=lambda x: x.get('_fused_score', x.get('_score', 0.0)), reverse=True)
    return fused


# R2 upload

def upload_query_image_to_r2(img_bytes: bytes, filename: str) -> str:
    if not PUBLIC_BASE_URL:
        raise RuntimeError("PUBLIC_BASE_URL is not configured.")
    r2 = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    )
    ext = os.path.splitext(filename)[1].lower() or ".jpg"
    key = f"queries/{uuid.uuid4()}{ext}"
    content_type = mimetypes.guess_type(filename)[0] or "image/jpeg"
    r2.put_object(Bucket=R2_BUCKET, Key=key, Body=img_bytes, ContentType=content_type)
    return f"{PUBLIC_BASE_URL.rstrip('/')}/{key}"


# Pagination UI

def render_pagination(total_pages: int, current_page_zerobased: int):
    if total_pages <= 1:
        return
    max_pages_to_show = 10
    nav = st.container()
    cols = nav.columns([1, 1, *(2 for _ in range(max_pages_to_show)), 1, 1])

    def set_page(page_num):
        st.session_state.page = page_num

    is_first_page = current_page_zerobased == 0
    cols[0].button("«", on_click=set_page, args=[0], disabled=is_first_page, use_container_width=True)
    cols[1].button("‹", on_click=set_page, args=[current_page_zerobased - 1], disabled=is_first_page, use_container_width=True)

    half = max_pages_to_show // 2
    start_page = max(0, current_page_zerobased - half)
    end_page = min(total_pages, start_page + max_pages_to_show)
    if end_page - start_page < max_pages_to_show:
        start_page = max(0, end_page - max_pages_to_show)

    for i, page_num in enumerate(range(start_page, end_page)):
        is_current = page_num == current_page_zerobased
        cols[i + 2].button(
            f"{page_num + 1}",
            on_click=set_page,
            args=[page_num],
            disabled=is_current,
            type="primary" if is_current else "secondary",
            use_container_width=True,
        )

    is_last_page = current_page_zerobased == total_pages - 1
    cols[-2].button("›", on_click=set_page, args=[current_page_zerobased + 1], disabled=is_last_page, use_container_width=True)
    cols[-1].button("»", on_click=set_page, args=[total_pages - 1], disabled=is_last_page, use_container_width=True)


# =============================================================
# UI and Application Flow
# =============================================================

st.set_page_config(page_title="Baldų paieška", layout="wide")
st.title("🛋️ Baldų ir interjero elementų paieška")
st.caption("Sistema pirmiausia ieško vizualiai panašių produktų. Tekstas – papildomas signalas.")

# Session state defaults
for k, v in (
    ('page', 0),
    ('last_upload_hash', None),
    ('color_filter_hidden', False),
    ('color_controls_rerolled', False),
):
    if k not in st.session_state:
        st.session_state[k] = v

# Sidebar
st.sidebar.header("Paieškos nustatymai")
st.sidebar.markdown("**📤 Įkelkite paveikslėlį**")
st.sidebar.caption("Vilkite ir numeskite arba pasirinkite failą iš kompiuterio.")
uploaded_file = st.sidebar.file_uploader(
    "Įkelkite paveikslėlį",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
    help=(
        "Nuotrauka naudojama vizualinei paieškai — surandami vaizdu panašūs produktai. "
        "Jei įvesite tekstą, jis dar labiau susiaurins rezultatus (pvz., modelis ar medžiaga). "
        "Įjungus spalvų filtrą, rezultatai papildomai filtruojami pagal objekto spalvą jūsų nuotraukoje. "
        "Leidžiami formatai: JPG, JPEG, PNG, WEBP."
    ),
)
# Išvalyti teksto paieškos lauką kiekvieno naujo paveikslėlio įkėlimo metu (prieš kuriant valdiklį)
if uploaded_file:
    try:
        _hash_for_text = hash(uploaded_file.getvalue())
        if st.session_state.get('last_upload_hash_textclear') != _hash_for_text:
            st.session_state['last_upload_hash_textclear'] = _hash_for_text
            st.session_state['search_text'] = ""
    except Exception:
        pass

search_query = st.sidebar.text_input(
    "🔍 Ieškoti pagal tekstą",
    help=(
        "Įveskite produkto pavadinimą, modelį ar frazę (pvz., ‘raudona sofa’). "
        "Paieška vertina visos frazės prasmę (semantiškai), todėl ‘raudona sofa’ ieškos būtent raudonų sofų. "
        "Jei įkelta nuotrauka, tekstas susiaurina vizualiai rastus rezultatus; jei nuotraukos nėra – ieško tik pagal tekstą."
    ),
key="search_text",
)

# Optional: narrow by product type
selected_type = st.sidebar.selectbox(
    "Produkto tipas (pasirinktinai)", TYPES_ALL, index=0,
    help="Apribokite rezultatus iki pasirinktos kategorijos.",
)

# Tekstinės paieškos atveju rodyti spalvų parinkiklį
text_use_color_picker = False
text_picker_hex: Optional[str] = None
if not uploaded_file:
    text_use_color_picker = st.sidebar.checkbox(
        "Filtruoti pagal pasirinktą spalvą",
        value=False,
        help="Jei įjungta, rezultatai bus filtruojami pagal žemiau pasirinktą spalvą.",
    )
    text_picker_hex = st.sidebar.color_picker(
        "🎨 Pasirinkite spalvą",
        value="#d9bc92",
        help="Taikoma tik tekstinei paieškai. Parinkta spalva naudojama atrinkti vizualiai panašius atspalvius.",
    )
    # Spalvos panašumo slankiklis (tik tekstinei paieškai)
    st.sidebar.slider(
        "Spalvos panašumo riba",
        0, 150,
        st.session_state.get('text_color_threshold', 60), 10,
        key="text_color_threshold",
        help=(
            "Kuo mažesnė reikšmė, tuo griežčiau parenkamos tik labai panašios spalvos. "
            "Didesnė reikšmė leidžia platesnį atspalvių diapazoną."
        ),
    )

# --- Early reset: if a NEW image is uploaded, unhide colour controls BEFORE rendering them
is_new_upload = False
if uploaded_file:
    try:
        _img_preview_bytes = uploaded_file.getvalue()
        _img_hash = hash(_img_preview_bytes)
        prev_hash = st.session_state.get('last_upload_hash')
        is_new_upload = (_img_hash != prev_hash)
        if is_new_upload:
            st.session_state['last_upload_hash'] = _img_hash
            st.session_state['color_filter_hidden'] = False
            st.session_state['color_controls_rerolled'] = False
    except Exception:
        is_new_upload = False
else:
    is_new_upload = False

# Colour controls: visible only when an image is uploaded
if uploaded_file:
    if st.session_state.get('color_filter_hidden', False):
        use_color_filter = False
        color_threshold = 50
    else:
        # Default ON on a brand new upload
        if is_new_upload:
            st.session_state['color_filter_checked'] = True
        use_color_filter = st.sidebar.checkbox(
            "Įjungti spalvų filtravimą",
            value=st.session_state.get('color_filter_checked', True),
            key="color_filter_checked",
        )
        color_threshold = st.sidebar.slider(
            "Spalvos panašumo riba", 0, 150,
            st.session_state.get('color_threshold', 50), 10,
            key="color_threshold",
            help=(
                "Nustato, kiek artima turi būti katalogo prekės spalva jūsų nuotraukai. "
                "Mažesnė reikšmė – griežtesnis atitikimas; didesnė – leidžia daugiau atspalvių."
            ),
        )
else:
    # No image uploaded: hide and turn off colour filter
    use_color_filter = False
    color_threshold = 50

final_hits: List[Dict[str, Any]] = []

# --- Main Logic Branch: Image Search (+ optional text) ---
if uploaded_file:
    st.sidebar.image(uploaded_file, caption="Įkeltas paveikslėlis", width=180)
    img_bytes = uploaded_file.getvalue()
    current_hash = hash(img_bytes)

    # New image resets colour-control hiding
    if st.session_state.last_upload_hash != current_hash:
        st.session_state.last_upload_hash = current_hash
        st.session_state.color_filter_hidden = False
        st.session_state.color_controls_rerolled = False

    query_rgb = get_dominant_color(img_bytes) if use_color_filter else None

    with st.spinner("Ieškoma vizualiai panašių elementų..."):
        try:
            query_url = upload_query_image_to_r2(img_bytes, uploaded_file.name)
        except Exception as e:
            st.error(f"Nepavyko įkelti paveikslėlio: {e}")
            st.stop()

        # Visual-only and text-fields tensor searches (no server-side colour filter)
        vis_res = marqo_search(query_url, attrs=[IMAGE_FIELD], method="TENSOR")
        sem_res = marqo_search(query_url, attrs=[TITLE_FIELD, DESCRIPTION_FIELD, "spec_text", SEARCH_BLOB_FIELD], method="TENSOR")

        vis_hits = vis_res.get("hits", []) if vis_res else []
        sem_hits = sem_res.get("hits", []) if sem_res else []
        image_search_results = fuse_hits(vis_hits, sem_hits, alpha=0.88)

        if search_query.strip():
            # Parse color from text; if provided, use it to override image color filter
            color_rgb_text, rest_text = parse_color_from_text(search_query)
            if color_rgb_text is not None:
                query_rgb = color_rgb_text
            qtext = rest_text.strip() if rest_text else search_query.strip()
            txt_res = marqo_search(q=qtext, limit=1000, attrs=[TITLE_FIELD, DESCRIPTION_FIELD, "spec_text", SEARCH_BLOB_FIELD], method="TENSOR")
            text_hits = txt_res.get("hits", []) if txt_res else []
            text_ids = {h.get('_id') for h in text_hits}
            final_hits = [h for h in image_search_results if h.get('_id') in text_ids]
        else:
            final_hits = image_search_results

    # Client-side colour filtering using per-item hex colour
    if use_color_filter and query_rgb is not None and final_hits:
        kept, unknowns = [], []
        for h in final_hits:
            hx = h.get(DOM_COLOR_FIELD)
            rgb = hex_to_rgb(hx)
            if rgb is None:
                unknowns.append(h)
                continue
            if color_distance(query_rgb, rgb) <= float(color_threshold):
                kept.append(h)

        if kept:
            # Keep unknowns if very few matches to avoid over-pruning
            final_hits = kept + (unknowns if len(kept) < 5 else [])
        else:
            # No colour matches: don't hide controls; just show info and fall back
            st.info("Dauguma įrašų neturi spalvos indekse — rodau be spalvų filtro.")
            final_hits = image_search_results

# --- Main Logic Branch: Text-Only Search ---
elif search_query.strip():
    # Text-only search: keep colour controls hidden & off
    st.session_state.color_filter_hidden = True
    st.session_state.color_controls_rerolled = False

    with st.spinner("Ieškoma pagal tekstą..."):
        color_rgb_text, rest_text = parse_color_from_text(search_query)
        # Jei naudotojas pasirinko spalvą parinkiklyje – ji turi pirmenybę
        if text_use_color_picker and isinstance(text_picker_hex, str):
            _picked = hex_to_rgb(text_picker_hex)
            if _picked is not None:
                color_rgb_text = _picked

        qtext = (rest_text.strip() if rest_text else "baldai")  # neutral fallback to fetch a broad set
        txt_res = marqo_search(q=qtext, limit=1000, attrs=[TITLE_FIELD, DESCRIPTION_FIELD, "spec_text", SEARCH_BLOB_FIELD], method="TENSOR")
        final_hits = txt_res.get("hits", []) if txt_res else []

        # Apply colour filter (from text or picker), if any
        if color_rgb_text is not None and final_hits:
            threshold = float(st.session_state.get('text_color_threshold', 60))
            kept, unknowns = [], []
            for h in final_hits:
                hx = get_hit_field(h, DOM_COLOR_FIELD, 'dominant_color')
                rgb = hex_to_rgb(hx) if isinstance(hx, str) else None
                if rgb is None:
                    unknowns.append(h)
                    continue
                if color_distance(color_rgb_text, rgb) <= threshold:
                    kept.append(h)
            final_hits = kept if kept else final_hits

# --- Initial State ---
else:
    # Initial state: no image — hide colour controls
    st.session_state.color_filter_hidden = True
    st.info("Įkelkite paveikslėlį arba įveskite paieškos frazę.")

# Before rendering, apply optional type narrowing
final_hits = filter_by_type(final_hits, selected_type)

# =============================================================
# Render Results
# =============================================================

if final_hits:
    # Reset pagination when result size changes
    if "last_hit_count" not in st.session_state or st.session_state.last_hit_count != len(final_hits):
        st.session_state.page = 0
    st.session_state.last_hit_count = len(final_hits)

    st.subheader(f"Rasta rezultatų: {len(final_hits)}")
    page_size = 50  # 5 per row × 10 rows per page
    total_pages = (len(final_hits) - 1) // page_size + 1
    current_page = st.session_state.page

    render_pagination(total_pages, current_page)

    start_idx = current_page * page_size
    end_idx = start_idx + page_size
    page_hits = final_hits[start_idx:end_idx]

    cols = st.columns(5)
    for i, h in enumerate(page_hits):
        with cols[i % 5]:
            img_url = get_hit_field(h, IMAGE_FIELD, ALT_IMAGE_FIELD, "image")
            title = get_hit_field(h, TITLE_FIELD, 'title') or 'Be pavadinimo'
            sku = get_hit_field(h, SKU_FIELD, 'product_id', 'sku', 'SKU') or h.get('_id')
            click_url = get_hit_field(h, CLICK_URL_FIELD, 'product_url')
            dom_color_hex = get_hit_field(h, DOM_COLOR_FIELD, 'dominant_color')
            score = h.get('_fused_score', h.get('_score', None))

            if img_url and click_url:
                alt_txt = f"{title} · SKU {sku}" if sku else title
                st.markdown(
                    f'<a href="{click_url}" target="_blank" rel="noopener noreferrer" title="Atidaryti produktą">'
                    f'<img src="{img_url}" alt="{alt_txt}" style="width:100%;border-radius:12px;display:block;"/></a>',
                    unsafe_allow_html=True,
                )
            elif img_url:
                st.image(img_url, use_container_width=True)
            if sku:
                st.write(f"**{title}** · `{sku}`")
            else:
                st.write(f"**{title}**")
            if isinstance(score, (int, float)):
                st.caption(f"Panašumas: {score:.3f}")
            if isinstance(dom_color_hex, str) and len(dom_color_hex) >= 4:
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px">'
                    f'<div style="width:20px;height:20px;background:{dom_color_hex};border:1px solid #ccc;border-radius:4px"></div>'
                    f'<span>{dom_color_hex}</span></div>',
                    unsafe_allow_html=True,
                )
            st.markdown('---')

elif uploaded_file or search_query:
    st.warning("Rezultatų nerasta. Pabandykite pakoreguoti užklausą ar filtrus.")
