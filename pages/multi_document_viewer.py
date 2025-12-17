# pages/multi_document_viewer.py
"""
Multi-Document Bundle Viewer & Classifier
=========================================
Streamlit app for visually browsing multi-page PDF bundles and classifying 
the documents within them.

Features:
- Visual page-by-page navigation
- Automatic document boundary detection
- Classification of each detected document
- Export results to JSON
"""

import io
import json
import os
import sys
import base64
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from PIL import Image
import streamlit as st
from pdf2image import convert_from_bytes
import imagehash
import oci

# Add parent directory to path for imports
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import (
    COMPARTMENT_ID,
    DEFAULT_MODEL_ID,
    OUTPUT_DIR,
)
from oci_utils import (
    init_generative_ai_client,
    init_document_client,
    load_categories,
    create_chat_request,
    create_chat_details,
)

st.set_page_config(page_title="Multi-Document Viewer", layout="wide")
st.title("📚 Multi-Document Bundle Viewer")
st.caption("Upload a multi-page PDF to detect and classify documents within it")


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PageInfo:
    """Information about a single page"""
    page_num: int
    image: Image.Image
    is_blank: bool
    is_boundary: bool
    document_idx: Optional[int] = None
    category: Optional[str] = None
    confidence: Optional[str] = None
    confidence_score: Optional[float] = None


@dataclass
class DocumentSegment:
    """A detected document within the bundle"""
    doc_idx: int
    start_page: int
    end_page: int
    category: Optional[str] = None
    confidence: Optional[str] = None
    confidence_score: Optional[float] = None
    reasoning: Optional[str] = None
    ocr_text: Optional[str] = None


# Sensitive categories for flagging
SENSITIVE_CATEGORIES = {
    "Bank Details", "Passport", "Driving License", "ID", "Birth Certificate",
    "Marriage Certificate", "BPSS", "Security check responses", "Screening report",
    "Health Declaration", "Employee medical reports from Occupational Health",
    "Fit Notes", "Disclosure Statements", "Compromise Agreements",
}


# ============================================================================
# Helper Functions
# ============================================================================

def image_to_jpeg_bytes(image: Image.Image, quality: int = 85) -> bytes:
    """Convert PIL Image to JPEG bytes"""
    if image.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[-1])
        image = bg
    elif image.mode != "RGB":
        image = image.convert("RGB")
    
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def is_blank_page(image: Image.Image, threshold: float = 0.95) -> bool:
    """Detect if a page is blank"""
    gray = np.array(image.convert('L'))
    white_ratio = np.sum(gray > 240) / gray.size
    return white_ratio > threshold


def compute_hash(image: Image.Image) -> imagehash.ImageHash:
    """Compute perceptual hash"""
    return imagehash.phash(image.convert('L').resize((64, 64)))


@st.cache_data
def convert_pdf_to_images(pdf_bytes: bytes, dpi: int = 100) -> List[bytes]:
    """Convert PDF to list of JPEG bytes (cached)"""
    images = convert_from_bytes(pdf_bytes, dpi=dpi)
    return [image_to_jpeg_bytes(img) for img in images]


def detect_boundaries(images: List[Image.Image], hash_threshold: int = 15) -> List[PageInfo]:
    """
    Detect document boundaries and return page info.
    """
    pages = []
    boundaries = [0]
    hashes = []
    blank_indices = set()
    
    # First pass: compute hashes and detect blanks
    for i, img in enumerate(images):
        is_blank = is_blank_page(img)
        if is_blank:
            blank_indices.add(i)
            hashes.append(None)
        else:
            hashes.append(compute_hash(img))
    
    # Second pass: find boundaries
    for i in range(1, len(images)):
        is_boundary = False
        
        # After blank separator
        if i - 1 in blank_indices and i not in blank_indices:
            is_boundary = True
        # Large visual change
        elif hashes[i] is not None:
            prev_idx = i - 1
            while prev_idx >= 0 and hashes[prev_idx] is None:
                prev_idx -= 1
            if prev_idx >= 0 and hashes[prev_idx] is not None:
                if hashes[i] - hashes[prev_idx] > hash_threshold:
                    is_boundary = True
        
        if is_boundary:
            boundaries.append(i)
    
    # Create PageInfo objects
    current_doc = 0
    for i, img in enumerate(images):
        if i in boundaries[1:]:  # New document starts
            current_doc += 1
        
        pages.append(PageInfo(
            page_num=i + 1,
            image=img,
            is_blank=i in blank_indices,
            is_boundary=i in boundaries,
            document_idx=current_doc if i not in blank_indices else None, 
        ))
    
    return pages


def create_document_segments(pages: List[PageInfo]) -> List[DocumentSegment]:
    """Create document segments from page info"""
    segments = {}
    
    for page in pages:
        if page.document_idx is not None:
            if page.document_idx not in segments:
                segments[page.document_idx] = DocumentSegment(
                    doc_idx=page.document_idx,
                    start_page=page.page_num,
                    end_page=page.page_num
                )
            else:
                segments[page.document_idx].end_page = page.page_num
    
    return list(segments.values())


def extract_text_from_ocr(du_dict: dict, max_chars: int = 2000) -> str:
    """Extract plain text from OCR result"""
    text_parts = []
    total_chars = 0
    
    for page in du_dict.get("pages", []):
        for line in page.get("lines", []):
            line_text = line.get("text", "").strip()
            if line_text:
                text_parts.append(line_text)
                total_chars += len(line_text) + 1
                if total_chars >= max_chars:
                    break
        if total_chars >= max_chars:
            break
    
    return "\n".join(text_parts)


def ocr_page(doc_client, image: Image.Image) -> str:
    """Run OCR on a single page"""
    from oci.ai_document.models import (
        AnalyzeDocumentDetails,
        InlineDocumentDetails,
        DocumentTextExtractionFeature,
    )
    
    img_bytes = image_to_jpeg_bytes(image)
    encoded = base64.b64encode(img_bytes).decode("utf-8")
    
    inline_doc = InlineDocumentDetails(data=encoded, source="INLINE")
    analyze_details = AnalyzeDocumentDetails(
        compartment_id=COMPARTMENT_ID,
        features=[DocumentTextExtractionFeature()],
        document=inline_doc,
        language="en",
    )
    
    response = doc_client.analyze_document(analyze_details)
    du_dict = oci.util.to_dict(response.data)
    return extract_text_from_ocr(du_dict)


def classify_documents_batch(
    gen_client, 
    compartment_id: str, 
    segments: List[DocumentSegment],
    categories: List[str]
) -> List[DocumentSegment]:
    """Classify all documents in a single LLM call"""
    
    categories_list = "\n".join([f"- {cat}" for cat in categories])
    
    docs_text = ""
    for seg in segments:
        text_preview = seg.ocr_text[:500] if seg.ocr_text else "No text"
        docs_text += f"\n--- DOCUMENT {seg.doc_idx + 1} (Pages {seg.start_page}-{seg.end_page}) ---\n{text_preview}\n"
    
    prompt = f"""Classify each document into ONE category from the list.

CATEGORIES:
{categories_list}

DOCUMENTS:
{docs_text}

Return a JSON array:
[
  {{"document": 1, "category": "Category", "confidence": "high|medium|low", "confidence_score": 0.95, "reasoning": "Why"}},
  ...
]
"""
    
    chat_request = create_chat_request(prompt=prompt, max_tokens=4000, temperature=0.0)
    chat_detail = create_chat_details(chat_request, model_id=DEFAULT_MODEL_ID, compartment_id=compartment_id)
    
    response = gen_client.chat(chat_detail)
    response_text = (
        response.data.chat_response.choices[0]
        .message.content[0]
        .text.strip()
    )
    
    # Clean and parse
    if response_text.startswith("```"):
        response_text = response_text.strip("`").strip()
        if response_text.lower().startswith("json"):
            response_text = response_text[4:].strip()
    
    import re
    json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
    if json_match:
        response_text = json_match.group()
    
    classifications = json.loads(response_text)
    
    # Apply to segments
    for cls in classifications:
        doc_idx = cls.get("document", 0) - 1
        if 0 <= doc_idx < len(segments):
            segments[doc_idx].category = cls.get("category")
            segments[doc_idx].confidence = cls.get("confidence")
            segments[doc_idx].confidence_score = cls.get("confidence_score")
            segments[doc_idx].reasoning = cls.get("reasoning")
    
    return segments


# ============================================================================
# Main App
# ============================================================================

# Initialize session state
if "pages" not in st.session_state:
    st.session_state.pages = []
if "segments" not in st.session_state:
    st.session_state.segments = []
if "current_page" not in st.session_state:
    st.session_state.current_page = 0
if "classified" not in st.session_state:
    st.session_state.classified = False


# Sidebar
with st.sidebar:
    st.header("📁 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file:
        if st.button("🔍 Analyze Document", type="primary"):
            with st.spinner("Converting PDF to images..."):
                pdf_bytes = uploaded_file.read()
                images = convert_from_bytes(pdf_bytes, dpi=100)
                
            with st.spinner("Detecting document boundaries..."):
                st.session_state.pages = detect_boundaries(images)
                st.session_state.segments = create_document_segments(st.session_state.pages)
                st.session_state.current_page = 0
                st.session_state.classified = False
            
            st.success(f"Found {len(st.session_state.segments)} documents in {len(images)} pages")
        
        if st.session_state.pages and not st.session_state.classified:
            if st.button("🤖 Classify All Documents"):
                with st.spinner("Running OCR on first pages..."):
                    doc_client = init_document_client()
                    for seg in st.session_state.segments:
                        first_page_idx = seg.start_page - 1
                        if first_page_idx < len(st.session_state.pages):
                            img = st.session_state.pages[first_page_idx].image
                            seg.ocr_text = ocr_page(doc_client, img)
                
                with st.spinner("Classifying with Llama 3.3..."):
                    gen_client, compartment_id = init_generative_ai_client()
                    categories = load_categories()
                    st.session_state.segments = classify_documents_batch(
                        gen_client, compartment_id, 
                        st.session_state.segments, categories
                    )
                    
                    # Apply classifications to pages
                    for seg in st.session_state.segments:
                        for page in st.session_state.pages:
                            if page.document_idx == seg.doc_idx:
                                page.category = seg.category
                                page.confidence = seg.confidence
                                page.confidence_score = seg.confidence_score
                    
                    st.session_state.classified = True
                
                st.success("Classification complete!")
                st.rerun()
    
    # Document summary
    if st.session_state.segments:
        st.markdown("---")
        st.subheader("📋 Documents Found")
        for seg in st.session_state.segments:
            is_sensitive = seg.category in SENSITIVE_CATEGORIES
            icon = "🔴" if is_sensitive else "📄"
            category = seg.category or "Not classified"
            confidence = f"{seg.confidence_score:.0%}" if seg.confidence_score else ""
            
            with st.expander(f"{icon} Doc {seg.doc_idx + 1}: Pages {seg.start_page}-{seg.end_page}"):
                st.write(f"**Category:** {category}")
                if seg.confidence:
                    st.write(f"**Confidence:** {seg.confidence} {confidence}")
                if seg.reasoning:
                    st.write(f"**Reasoning:** {seg.reasoning}")
                if is_sensitive:
                    st.warning("⚠️ Potentially sensitive document")


# Main content area
if st.session_state.pages:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Page navigation
        n_pages = len(st.session_state.pages)
        
        nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
        with nav_col1:
            if st.button("⬅️ Previous", disabled=st.session_state.current_page == 0):
                st.session_state.current_page -= 1
                st.rerun()
        
        with nav_col2:
            new_page = st.slider(
                "Page", 1, n_pages, 
                st.session_state.current_page + 1,
                label_visibility="collapsed"
            )
            if new_page - 1 != st.session_state.current_page:
                st.session_state.current_page = new_page - 1
                st.rerun()
        
        with nav_col3:
            if st.button("Next ➡️", disabled=st.session_state.current_page >= n_pages - 1):
                st.session_state.current_page += 1
                st.rerun()
        
        # Display current page
        page_info = st.session_state.pages[st.session_state.current_page]
        
        # Page status badges
        badges = []
        if page_info.is_boundary:
            badges.append("🆕 Document Start")
        if page_info.is_blank:
            badges.append("⬜ Blank Page")
        if page_info.category:
            badges.append(f"📁 {page_info.category}")
        
        st.markdown(f"### Page {page_info.page_num} of {n_pages}")
        if badges:
            st.markdown(" • ".join(badges))
        
        # Display image
        img_bytes = image_to_jpeg_bytes(page_info.image)
        st.image(img_bytes, use_container_width=True)
    
    with col2:
        st.subheader("Page Details")
        
        page_info = st.session_state.pages[st.session_state.current_page]
        
        # Page info
        st.metric("Page Number", f"{page_info.page_num} / {n_pages}")
        
        if page_info.is_blank:
            st.info("⬜ This is a blank/separator page")
        
        if page_info.is_boundary:
            st.success("🆕 New document starts here")
        
        # Document info
        if page_info.document_idx is not None:
            seg = st.session_state.segments[page_info.document_idx]
            
            st.markdown("---")
            st.subheader(f"Document {seg.doc_idx + 1}")
            st.write(f"**Pages:** {seg.start_page} - {seg.end_page}")
            
            if seg.category:
                is_sensitive = seg.category in SENSITIVE_CATEGORIES
                
                if is_sensitive:
                    st.error(f"🔴 **{seg.category}**")
                    st.warning("⚠️ Potentially sensitive document")
                else:
                    st.success(f"📁 **{seg.category}**")
                
                conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(seg.confidence, "⚪")
                st.write(f"**Confidence:** {conf_emoji} {seg.confidence} ({seg.confidence_score:.0%})")
                
                if seg.reasoning:
                    st.write(f"**Reasoning:** {seg.reasoning}")
            else:
                st.info("Not classified yet")
        
        # Export button
        if st.session_state.classified:
            st.markdown("---")
            if st.button("💾 Export Results"):
                result = {
                    "total_pages": len(st.session_state.pages),
                    "documents_found": len(st.session_state.segments),
                    "documents": [
                        {
                            "pages": f"{seg.start_page}-{seg.end_page}",
                            "category": seg.category,
                            "confidence": seg.confidence,
                            "confidence_score": seg.confidence_score,
                            "reasoning": seg.reasoning,
                            "is_sensitive": seg.category in SENSITIVE_CATEGORIES
                        }
                        for seg in st.session_state.segments
                    ]
                }
                
                st.download_button(
                    "📥 Download JSON",
                    json.dumps(result, indent=2),
                    file_name="bundle_analysis.json",
                    mime="application/json"
                )

else:
    # No file uploaded - show instructions
    st.info("👈 Upload a PDF file in the sidebar to begin")
    
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        This tool analyzes multi-page PDF bundles containing multiple scanned documents:
        
        1. **Upload** a PDF containing multiple documents (e.g., employee files)
        2. **Analyze** to detect document boundaries (blank pages, layout changes)
        3. **Classify** each detected document using AI
        4. **Browse** through pages visually with classification info
        5. **Export** results to JSON
        
        **Smart Detection:**
        - Detects blank separator pages
        - Uses visual similarity to find document boundaries
        - Only OCRs the first page of each document (cost efficient)
        
        **Sensitive Document Flagging:**
        - Automatically flags potentially sensitive documents
        - Passports, Bank Details, ID documents, Health records, etc.
        """)
