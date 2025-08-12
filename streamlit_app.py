import os
import io
import json
import uuid
import mimetypes
from typing import Any, Dict, Optional

import requests
import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import boto3

# Streamlit App for Visual Search using Marqo, Cloudflare R2, and Cloudflare Access.

# -----------------------------
# Konfigūracija (iš secrets arba ENV)
# -----------------------------

def _cfg(key: str, default: Optional[str] = None) -> Optional[str]:
    """Retrieves a configuration value from Streamlit secrets or environment variables."""
    try:
        # Try to get from Streamlit secrets
        return st.secrets[key]  # type: ignore[attr-defined]
    except Exception:
        # Fallback to environment variables
        return os.getenv(key, default)

MARQO_URL: str = (_cfg("MARQO_URL", "https://marqo.logicafutura.com") or "").rstrip("/")
INDEX_NAME: str = _cfg("INDEX_NAME", "furniture-index") or "furniture-index"

TITLE_FIELD: str = _cfg("TITLE_FIELD", "title") or "title"
IMAGE_FIELD: str = _cfg("IMAGE_FIELD", "image") or "image"
ALT_IMAGE_FIELD: str = _cfg("ALT_IMAGE_FIELD", "image_vendor_url") or "image_vendor_url"
DOM_COLOR_FIELD: str = _cfg("DOMINANT_COLOR_FIELD", "dominant_color") or "dominant_color"
SKU_FIELD: str = _cfg("SKU_FIELD", "sku") or "sku"

# Cloudflare Access Credentials
CF_ACCESS_CLIENT_ID = _cfg("CF_ACCESS_CLIENT_ID")
CF_ACCESS_CLIENT_SECRET = _cfg("CF_ACCESS_CLIENT_SECRET")

# R2 Configuration
R2_ENDPOINT_URL: str = _cfg("R2_ENDPOINT_URL") or ""
R2_ACCESS_KEY_ID: Optional[str] = _cfg("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY: Optional[str] = _cfg("R2_SECRET_ACCESS_KEY")
R2_BUCKET: str = _cfg("R2_BUCKET", "streamlit098") or "streamlit098"
PUBLIC_BASE_URL: str = _cfg("PUBLIC_BASE_URL", "https://pub-3ec323b4b4864664846453a4dda0930e.r2.dev") or ""


HEADERS: Dict[str, str] = {"Content-Type": "application/json"}
if CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET:
    HEADERS["CF-Access-Client-Id"] = CF_ACCESS_CLIENT_ID
    HEADERS["CF-Access-Client-Secret"] = CF_ACCESS_CLIENT_SECRET

KNOWN_TYPES = ['table', 'lamp', 'rack', 'chair', 'sofa', 'bench', 'bed', 'cabinet', 'desk']

# -----------------------------
# R2 pagalbinės (įkėlimas)
# -----------------------------

def _r2_client():
    if not (R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY and R2_ENDPOINT_URL):
        raise RuntimeError("R2 prieigos raktai arba endpoint URL nenurodyti. Įdėkite R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY ir R2_ENDPOINT_URL į secrets.")
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    )


def upload_query_image_to_r2(img_bytes: bytes, filename: str) -> str:
    """Uploads an image to R2 and returns its full public URL."""
    if not PUBLIC_BASE_URL:
        raise RuntimeError("PUBLIC_BASE_URL is not configured in secrets.")
        
    ext = os.path.splitext(filename)[1].lower() or ".jpg"
    # Create a unique key for the object in a 'queries' folder
    key = f"queries/{uuid.uuid4()}{ext}"
    content_type = mimetypes.guess_type(filename)[0] or "image/jpeg"
    
    r2 = _r2_client()
    r2.put_object(Bucket=R2_BUCKET, Key=key, Body=img_bytes, ContentType=content_type)
    
    # Construct the full public URL
    base = PUBLIC_BASE_URL.rstrip('/')
    return f"{base}/{key}"

# -----------------------------
# Spalvos/objekto tipas pagalbinės
# -----------------------------

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


def detect_object_type(title: str) -> str:
    t = (title or "").lower()
    for obj in KNOWN_TYPES:
        if obj in t:
            return obj
    return "other"

# -----------------------------
# Marqo paieška (su Access antraštėmis)
# -----------------------------

def marqo_search(query: str) -> Optional[Dict[str, Any]]:
    """
    Performs a search on Marqo, handling errors and displaying full server responses in an expander.
    """
    payload = {
        "limit": 1000,
        "q": query,
        "searchMethod": "TENSOR",
        "searchableAttributes": [IMAGE_FIELD, TITLE_FIELD],
        "attributesToRetrieve": ["_id", TITLE_FIELD, IMAGE_FIELD, ALT_IMAGE_FIELD, DOM_COLOR_FIELD, SKU_FIELD]
    }
    url = f"{MARQO_URL}/indexes/{INDEX_NAME}/search"
    
    try:
        resp = requests.post(url, json=payload, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        try:
            return resp.json()
        except json.JSONDecodeError:
            st.error("Gautas sėkmingas atsakymas iš serverio, bet jame nebuvo JSON duomenų.")
            with st.expander("📄 Rodyti visą serverio atsakymą"):
                st.code(resp.text, language='html')
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"API paieškos klaida: {e}")
        if getattr(e, 'response', None) is not None and e.response.text:
            with st.expander("📄 Rodyti visą serverio atsakymą (neapkarpyta)"):
                st.code(e.response.text, language='html')
        else:
            st.warning("Serveris negrąžino jokio atsakymo turinio.")
        return None

# -----------------------------
# Streamlit UI
# -----------------------------

# Set page layout to "centered" for a narrower view and update page title
st.set_page_config(page_title="Baldų paieška", layout="centered")

# UPDATED: New title and icon
st.title("🛋️ Baldų ir interjero elementų paieška")
st.markdown("Įkelkite produkto nuotrauką arba įveskite raktažodį, kad rastumėte panašius baldus ir interjero elementus.")


# Initialize session state variables
for key, default in (
    ('last_upload_hash', None), ('search_results', None), ('query_color', None),
    ('page', 0), ('detected_object_type', None)
):
    if key not in st.session_state:
        st.session_state[key] = default

# --- Sidebar Controls ---
st.sidebar.header("Paieškos nustatymai")
uploaded_file = st.sidebar.file_uploader(
    "Pasirinkite paveikslėlį", 
    type=["jpg", "jpeg", "png", "gif", "bmp", "webp"], 
    key="uploader"
)
search_query = st.sidebar.text_input("🔍 Ieškoti pagal tekstą:", "")
color_threshold = st.sidebar.slider(
    "Spalvos panašumo riba", 0, 200, 50, 10,
    format="%d", help="Kairėje – labiau panaši, dešinėje – mažiau panaši"
)
use_color_filter = st.sidebar.checkbox("Įjungti spalvos filtravimą", value=True)


# --- Main Logic ---
if uploaded_file:
    st.sidebar.image(uploaded_file, caption="Įkeltas paveikslėlis", width=150)
    img_bytes = uploaded_file.getvalue()
    current_hash = hash(img_bytes)

    # Trigger search only if the image or color threshold changes
    if current_hash != st.session_state.last_upload_hash or color_threshold != st.session_state.get('last_color_threshold', -1):
        st.session_state.last_upload_hash = current_hash
        st.session_state.last_color_threshold = color_threshold
        st.session_state.page = 0  # Reset to first page on new search

        try:
            query_url = upload_query_image_to_r2(img_bytes, uploaded_file.name)
        except Exception as e:
            st.error(f"Nepavyko įkelti į R2: {e}")
            st.stop()

        query_color = get_dominant_color(img_bytes)
        st.session_state.query_color = query_color

        with st.spinner("Ieškoma panašių vaizdų..."):
            results = marqo_search(query_url)
            if results and results.get("hits"):
                raw_hits = results["hits"]
                for hit in raw_hits:
                    title = hit.get(TITLE_FIELD, hit.get("title", ""))
                    hit["object_type"] = detect_object_type(title)
                st.session_state.detected_object_type = raw_hits[0]["object_type"] if raw_hits else None
                st.session_state.search_results = {"hits": raw_hits}
            else:
                st.session_state.search_results = {"hits": []}

elif search_query.strip():
    # Trigger text search
    with st.spinner("Ieškoma pagal raktažodį..."):
        results = marqo_search(search_query)
        if results and results.get("hits"):
            for hit in results["hits"]:
                title = hit.get(TITLE_FIELD, hit.get("title", ""))
                hit["object_type"] = detect_object_type(title)
            st.session_state.search_results = {"hits": results["hits"]}
        else:
            st.session_state.search_results = {"hits": []}
else:
    # Clear results if no input
    st.session_state.search_results = None
    st.session_state.last_upload_hash = None
    st.session_state.query_color = None
    st.session_state.page = 0


# --- Results Rendering ---
results_data = st.session_state.search_results
if results_data and results_data.get("hits"):
    hits = results_data["hits"]

    # Post-processing filters
    if uploaded_file:
        # Filter by object type detected from the uploaded image
        if st.session_state.detected_object_type:
            hits = [h for h in hits if h["object_type"] == st.session_state.detected_object_type]
        
        # Filter by color similarity
        if use_color_filter and st.session_state.query_color is not None:
            query_color = st.session_state.query_color
            filtered_by_color = []
            for hit in hits:
                hit_color_hex = hit.get(DOM_COLOR_FIELD, "#000000")
                hit_rgb = hex_to_rgb(hit_color_hex)
                dist = color_distance(query_color, hit_rgb)
                if dist <= color_threshold + 5:
                    hit["_adj_score"] = hit.get('_score', 0) - (dist / 441.0)
                    filtered_by_color.append(hit)
            # Sort by the adjusted score
            hits = sorted(filtered_by_color, key=lambda x: x.get("_adj_score", x.get("_score", 0)), reverse=True)

    if uploaded_file and search_query.strip():
        # Additional keyword filter on results
        keyword = search_query.lower()
        hits = [h for h in hits if keyword in (h.get(TITLE_FIELD, h.get("title", "")).lower())]

    # --- Pagination (from local app) ---
    page_size = 9
    total_pages = (len(hits) - 1) // page_size + 1 if hits else 0
    current_page = st.session_state.page

    if not hits:
        st.info("Rezultatų nerasta. Pabandykite įkelti kitą paveikslėlį arba pakeisti filtrus.")
    else:
        st.subheader(f"Rasta rezultatų: {len(hits)}")
        
        col_prev, col_pg, col_next = st.columns([2, 8, 2])
        with col_prev:
            if st.button("⬅ Ankstesnis", disabled=(current_page == 0)):
                st.session_state.page -= 1
                st.rerun()
        with col_pg:
             st.markdown(f"<div style='text-align: center;'>Puslapis {current_page + 1} iš {total_pages}</div>", unsafe_allow_html=True)
        with col_next:
            if st.button("Kitas ➡", disabled=(current_page >= total_pages - 1)):
                st.session_state.page += 1
                st.rerun()

        # Display results for the current page
        start = current_page * page_size
        end = start + page_size
        page_hits = hits[start:end]

        cols = st.columns(3)
        for i, hit in enumerate(page_hits):
            with cols[i % 3]:
                img_url = hit.get(IMAGE_FIELD) or hit.get(ALT_IMAGE_FIELD) or hit.get("image")
                title = hit.get(TITLE_FIELD, hit.get('title', 'Be pavadinimo'))
                _id = hit.get('_id', 'Nėra')
                score = hit.get('_score')

                if img_url:
                    st.image(img_url, use_container_width=True)
                st.write(f"**{title}**")
                st.caption(f"ID: {_id}")
                if isinstance(score, (int, float)):
                    st.write(f"Panašumas: {score:.2f}")
                st.markdown("---")

elif results_data is not None:
    st.info("Rezultatų nerasta. Pabandykite įkelti paveikslėlį arba įvesti raktažodį.")
else:
    st.info("Įkelkite paveikslėlį arba įveskite paieškos frazę šoninėje juostoje.")
