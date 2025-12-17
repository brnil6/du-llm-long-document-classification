"""
Multi-Document Bundle Classifier
================================
Processes large PDF files (e.g., 70 pages) containing multiple scanned documents.
Uses smart boundary detection to minimize OCR/LLM costs.

Strategy:
1. Convert PDF to low-res images for boundary detection
2. Detect document boundaries (blank pages, visual breaks)
3. OCR only first page of each detected sub-document
4. Batch classify all documents with LLM
5. Generate report with page ranges and categories
"""

import os
import sys
import json
import base64
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image
from pdf2image import convert_from_path, convert_from_bytes
import imagehash
import oci

# Import from centralized modules
from config import (
    COMPARTMENT_ID,
    DEFAULT_MODEL_ID,
    OUTPUT_DIR,
    CATEGORIES_FILE,
)
from oci_utils import (
    init_generative_ai_client,
    init_document_client,
    load_categories,
    create_chat_request,
    create_chat_details,
)


@dataclass
class DocumentSegment:
    """Represents a detected document within the bundle"""
    start_page: int
    end_page: int
    category: Optional[str] = None
    confidence: Optional[str] = None
    confidence_score: Optional[float] = None
    reasoning: Optional[str] = None
    is_sensitive: bool = False
    first_page_text: Optional[str] = None


@dataclass
class BundleAnalysis:
    """Complete analysis of a document bundle"""
    filename: str
    total_pages: int
    documents_found: int
    segments: List[DocumentSegment]
    processing_time_seconds: float
    pages_ocrd: int
    model_used: str


# Sensitive document categories (for flagging)
SENSITIVE_CATEGORIES = {
    "Bank Details",
    "Passport",
    "Driving License",
    "ID",
    "Birth Certificate",
    "Marriage Certificate",
    "BPSS",
    "Security check responses",
    "Screening report",
    "Health Declaration",
    "Employee medical reports from Occupational Health",
    "Fit Notes",
    "Disclosure Statements",
    "Compromise Agreements",
}


def pdf_to_images(pdf_path: str, dpi: int = 72) -> List[Image.Image]:
    """
    Convert PDF to images at specified DPI.
    Lower DPI (72) for boundary detection, higher (200) for OCR.
    """
    print(f"📄 Converting PDF to images (DPI={dpi})...")
    images = convert_from_path(pdf_path, dpi=dpi)
    print(f"✓ Converted {len(images)} pages")
    return images


def compute_image_hash(image: Image.Image) -> imagehash.ImageHash:
    """Compute perceptual hash for an image"""
    # Convert to grayscale and resize for consistent hashing
    img_gray = image.convert('L').resize((128, 128))
    return imagehash.phash(img_gray)


def is_blank_page(image: Image.Image, threshold: float = 0.98) -> bool:
    """
    Detect if a page is blank or nearly blank.
    Returns True if page is mostly white/empty.
    """
    # Convert to grayscale
    gray = np.array(image.convert('L'))
    # Calculate percentage of white/near-white pixels
    white_pixels = np.sum(gray > 240)
    total_pixels = gray.size
    white_ratio = white_pixels / total_pixels
    return white_ratio > threshold


def detect_document_boundaries(images: List[Image.Image], 
                               hash_threshold: int = 15,
                               blank_threshold: float = 0.95) -> List[Tuple[int, int]]:
    """
    Detect document boundaries using visual analysis.
    
    Returns list of (start_page, end_page) tuples (1-indexed).
    
    Detection methods:
    1. Blank pages act as separators
    2. Large visual hash difference indicates new document
    3. First page is always a boundary
    """
    print("\n🔍 Detecting document boundaries...")
    
    n_pages = len(images)
    boundaries = [0]  # First page is always start of a document
    
    # Compute hashes for all pages
    hashes = []
    blank_pages = []
    
    for i, img in enumerate(images):
        page_num = i + 1
        
        # Check if blank
        if is_blank_page(img, blank_threshold):
            blank_pages.append(i)
            hashes.append(None)
            print(f"  Page {page_num}: BLANK")
        else:
            hashes.append(compute_image_hash(img))
    
    # Find boundaries based on blank pages and hash differences
    for i in range(1, n_pages):
        is_boundary = False
        reason = ""
        
        # Method 1: Previous page was blank (separator)
        if i - 1 in blank_pages and i not in blank_pages:
            is_boundary = True
            reason = "after blank separator"
        
        # Method 2: Large visual difference from previous non-blank page
        elif hashes[i] is not None:
            # Find previous non-blank page
            prev_idx = i - 1
            while prev_idx >= 0 and hashes[prev_idx] is None:
                prev_idx -= 1
            
            if prev_idx >= 0 and hashes[prev_idx] is not None:
                diff = hashes[i] - hashes[prev_idx]
                if diff > hash_threshold:
                    is_boundary = True
                    reason = f"visual change (diff={diff})"
        
        if is_boundary:
            boundaries.append(i)
            print(f"  Page {i + 1}: NEW DOCUMENT ({reason})")
    
    # Convert boundaries to (start, end) ranges
    segments = []
    for i, start in enumerate(boundaries):
        if i + 1 < len(boundaries):
            end = boundaries[i + 1] - 1
        else:
            end = n_pages - 1
        
        # Skip if this segment is just blank pages
        segment_pages = range(start, end + 1)
        non_blank_count = sum(1 for p in segment_pages if p not in blank_pages)
        
        if non_blank_count > 0:
            segments.append((start + 1, end + 1))  # Convert to 1-indexed
    
    print(f"\n✓ Found {len(segments)} document segments")
    for i, (start, end) in enumerate(segments):
        print(f"  Document {i + 1}: Pages {start}-{end} ({end - start + 1} pages)")
    
    return segments


def image_to_jpeg_bytes(image: Image.Image, quality: int = 85) -> bytes:
    """Convert PIL Image to JPEG bytes"""
    if image.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[-1])
        image = bg
    elif image.mode != "RGB":
        image = image.convert("RGB")
    
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def ocr_page(doc_client, image: Image.Image, page_num: int) -> str:
    """
    Run OCR on a single page image using OCI Document Understanding.
    Returns extracted text.
    """
    from oci.ai_document.models import (
        AnalyzeDocumentDetails,
        InlineDocumentDetails,
        DocumentTextExtractionFeature,
    )
    
    # Convert image to JPEG bytes
    img_bytes = image_to_jpeg_bytes(image)
    encoded_image = base64.b64encode(img_bytes).decode("utf-8")
    
    inline_doc = InlineDocumentDetails(data=encoded_image, source="INLINE")
    analyze_details = AnalyzeDocumentDetails(
        compartment_id=COMPARTMENT_ID,
        features=[DocumentTextExtractionFeature()],
        document=inline_doc,
        language="en",
    )
    
    try:
        response = doc_client.analyze_document(analyze_details)
        du_dict = oci.util.to_dict(response.data)
        
        # Extract text from all pages (should be 1)
        text_parts = []
        for page in du_dict.get("pages", []):
            for line in page.get("lines", []):
                text_parts.append(line.get("text", ""))
        
        return "\n".join(text_parts)
    
    except Exception as e:
        print(f"  ⚠️ OCR error on page {page_num}: {e}")
        return ""


def batch_classify_documents(
    client,
    compartment_id: str,
    segments: List[DocumentSegment],
    categories: List[str]
) -> List[DocumentSegment]:
    """
    Classify multiple document segments in a single LLM call.
    Uses batch processing for efficiency.
    """
    print("\n🤖 Classifying documents with Llama 3.3...")
    
    # Build categories list
    categories_list = "\n".join([f"- {cat}" for cat in categories])
    
    # Build documents list for prompt
    docs_text = ""
    for i, seg in enumerate(segments):
        text_preview = seg.first_page_text[:500] if seg.first_page_text else "No text extracted"
        docs_text += f"""
--- DOCUMENT {i + 1} (Pages {seg.start_page}-{seg.end_page}) ---
{text_preview}
"""
    
    prompt = f"""
You are a document classification expert. Analyze the following document excerpts and classify each one.

ALLOWED_CATEGORIES (choose exactly one per document):
{categories_list}

DOCUMENTS TO CLASSIFY:
{docs_text}

For each document, provide:
1. The category (must be from the allowed list, or "INVALID_CATEGORY" if none fit)
2. Confidence level: high, medium, or low
3. Confidence score: 0.0 to 1.0
4. Brief reasoning (one sentence)

OUTPUT FORMAT (JSON array):
[
  {{
    "document": 1,
    "category": "Category Name",
    "confidence": "high",
    "confidence_score": 0.95,
    "reasoning": "Contains employment contract terms and signatures"
  }},
  ...
]

Return ONLY the JSON array, no other text.
"""
    
    # Create and send chat request
    chat_request = create_chat_request(prompt=prompt, max_tokens=4000, temperature=0.0)
    chat_detail = create_chat_details(
        chat_request,
        model_id=DEFAULT_MODEL_ID,
        compartment_id=compartment_id
    )
    
    try:
        response = client.chat(chat_detail)
        response_text = (
            response.data.chat_response.choices[0]
            .message.content[0]
            .text.strip()
        )
        
        # Clean markdown if present
        if response_text.startswith("```"):
            response_text = response_text.strip("`").strip()
            if response_text.lower().startswith("json"):
                response_text = response_text[4:].strip()
        
        # Parse JSON
        import re
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group()
        
        classifications = json.loads(response_text)
        
        # Apply classifications to segments
        for cls in classifications:
            doc_idx = cls.get("document", 0) - 1
            if 0 <= doc_idx < len(segments):
                segments[doc_idx].category = cls.get("category", "Unknown")
                segments[doc_idx].confidence = cls.get("confidence", "unknown")
                segments[doc_idx].confidence_score = cls.get("confidence_score", 0.0)
                segments[doc_idx].reasoning = cls.get("reasoning", "")
                
                # Flag sensitive documents
                if segments[doc_idx].category in SENSITIVE_CATEGORIES:
                    segments[doc_idx].is_sensitive = True
        
        return segments
    
    except Exception as e:
        print(f"  ❌ Classification error: {e}")
        # Return segments with "Unknown" classification
        for seg in segments:
            seg.category = "Unknown"
            seg.confidence = "low"
        return segments


def analyze_document_bundle(pdf_path: str, output_dir: str = OUTPUT_DIR) -> BundleAnalysis:
    """
    Main function to analyze a multi-document PDF bundle.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save results
    
    Returns:
        BundleAnalysis object with complete results
    """
    start_time = datetime.now()
    filename = os.path.basename(pdf_path)
    
    print("=" * 70)
    print(f"Multi-Document Bundle Classifier")
    print(f"File: {filename}")
    print("=" * 70)
    
    # Step 1: Convert PDF to low-res images for boundary detection
    low_res_images = pdf_to_images(pdf_path, dpi=72)
    total_pages = len(low_res_images)
    
    # Step 2: Detect document boundaries
    boundaries = detect_document_boundaries(low_res_images)
    
    # Create DocumentSegment objects
    segments = [DocumentSegment(start_page=s, end_page=e) for s, e in boundaries]
    
    # Step 3: Convert to high-res for OCR (only first page of each segment)
    print("\n📷 Converting key pages to high-res for OCR...")
    high_res_images = pdf_to_images(pdf_path, dpi=200)
    
    # Step 4: OCR first page of each segment
    print("\n🔤 Running OCR on first page of each document...")
    doc_client = init_document_client()
    pages_ocrd = 0
    
    for seg in segments:
        page_idx = seg.start_page - 1  # Convert to 0-indexed
        print(f"  OCR page {seg.start_page}...", end=" ")
        seg.first_page_text = ocr_page(doc_client, high_res_images[page_idx], seg.start_page)
        pages_ocrd += 1
        
        if seg.first_page_text:
            preview = seg.first_page_text[:50].replace('\n', ' ')
            print(f"✓ ({len(seg.first_page_text)} chars) \"{preview}...\"")
        else:
            print("✓ (no text)")
    
    # Step 5: Batch classify all documents
    categories = load_categories()
    gen_client, compartment_id = init_generative_ai_client()
    segments = batch_classify_documents(gen_client, compartment_id, segments, categories)
    
    # Calculate processing time
    processing_time = (datetime.now() - start_time).total_seconds()
    
    # Create analysis result
    analysis = BundleAnalysis(
        filename=filename,
        total_pages=total_pages,
        documents_found=len(segments),
        segments=segments,
        processing_time_seconds=processing_time,
        pages_ocrd=pages_ocrd,
        model_used="Meta Llama 3.3 70B Instruct"
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)
    print(f"Total pages: {total_pages}")
    print(f"Documents found: {len(segments)}")
    print(f"Pages OCR'd: {pages_ocrd} (saved {total_pages - pages_ocrd} OCR calls)")
    print(f"Processing time: {processing_time:.1f} seconds")
    
    print("\n📋 Documents in bundle:")
    sensitive_count = 0
    for i, seg in enumerate(segments):
        sensitive_flag = "🔴 SENSITIVE" if seg.is_sensitive else ""
        print(f"\n  {i + 1}. Pages {seg.start_page}-{seg.end_page}: {seg.category}")
        print(f"     Confidence: {seg.confidence} ({seg.confidence_score:.0%})")
        if seg.reasoning:
            print(f"     Reason: {seg.reasoning}")
        if seg.is_sensitive:
            print(f"     {sensitive_flag}")
            sensitive_count += 1
    
    if sensitive_count > 0:
        print(f"\n⚠️  Found {sensitive_count} potentially sensitive document(s)")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{os.path.splitext(filename)[0]}_analysis.json"
    output_path = os.path.join(output_dir, output_filename)
    
    # Convert to JSON-serializable format
    result_dict = {
        "filename": analysis.filename,
        "total_pages": analysis.total_pages,
        "documents_found": analysis.documents_found,
        "pages_ocrd": analysis.pages_ocrd,
        "processing_time_seconds": round(analysis.processing_time_seconds, 2),
        "model_used": analysis.model_used,
        "sensitive_documents_found": sensitive_count,
        "documents": [
            {
                "pages": f"{seg.start_page}-{seg.end_page}",
                "page_count": seg.end_page - seg.start_page + 1,
                "category": seg.category,
                "confidence": seg.confidence,
                "confidence_score": seg.confidence_score,
                "reasoning": seg.reasoning,
                "is_sensitive": seg.is_sensitive
            }
            for seg in analysis.segments
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_path}")
    print("=" * 70)
    
    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Classify multiple documents within a single PDF bundle"
    )
    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file containing multiple documents"
    )
    parser.add_argument(
        "-o", "--output",
        default=OUTPUT_DIR,
        help=f"Output directory for results (default: {OUTPUT_DIR})"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_path):
        print(f"❌ Error: File not found: {args.pdf_path}")
        sys.exit(1)
    
    analyze_document_bundle(args.pdf_path, args.output)


if __name__ == "__main__":
    main()
