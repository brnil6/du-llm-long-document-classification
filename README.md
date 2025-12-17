# Document Understanding + LLM Document Classification

A Python application that combines Oracle Document Understanding (OCR) with Meta Llama 3.3 LLM to automatically classify and categorize HR/Finance documents.

## Features

- **Document Classification**: Automatically classifies documents into 90+ HR/Finance categories
- **Excel Batch Processing**: Categorize documents from Excel metadata files
- **Multi-Document Bundle Analysis**: Process 70+ page scanned bundles containing multiple documents
- **Single Document Classification**: OCR + classify individual documents (PDF/images)
- **Interactive Web UI**: Streamlit apps for visual document classification
- **Sensitive Document Detection**: Automatically flags potentially sensitive documents
- **Multi-language OCR Support**: English, Japanese, Chinese, Arabic, German, French, and more

## Project Structure

```
├── config.py                           # Centralized OCI and LLM configuration
├── oci_utils.py                        # Shared OCI client utilities
├── categories.json                     # Document categories (90+ HR/Finance categories)
│
├── 1_categorize_excel_documents.py     # CLI: Batch categorize Excel document metadata
├── 2_classify_single_document.py       # CLI: Classify single document (PDF/image)
├── 3_classify_multi_document.py        # CLI: Process multi-document PDF bundles
│
├── pages/
│   ├── run_du_llm.py                   # Streamlit: Single document classifier
│   └── multi_document_viewer.py        # Streamlit: Visual multi-document browser
│
├── outputs/                            # Classification results
└── requirements.txt                    # Python dependencies
```

## Command-Line Scripts

### 1. Excel Document Categorizer (`1_categorize_excel_documents.py`)

Categorize documents from Excel file metadata using Llama 3.3 with batch processing.

```bash
# Categorize + analyze results
python 1_categorize_excel_documents.py "input.xlsx"

# Custom output file and sheet
python 1_categorize_excel_documents.py input.xlsx -o output.xlsx -s "HR"

# Analyze existing results only
python 1_categorize_excel_documents.py --analyze-only results.xlsx

# Skip analysis after categorization
python 1_categorize_excel_documents.py input.xlsx --skip-analysis
```

### 2. Single Document Classifier (`2_classify_single_document.py`)

Classify a single PDF or image using OCR + Llama 3.3.

```bash
python 2_classify_single_document.py document.pdf
python 2_classify_single_document.py scan.jpg -o results/
```

### 3. Multi-Document Bundle Classifier (`3_classify_multi_document.py`)

Process large PDFs containing multiple scanned documents (e.g., 70-page employee file bundles).

```bash
python 3_classify_multi_document.py bundle.pdf
python 3_classify_multi_document.py bundle.pdf -o results/
```

**How it works:**
- Detects document boundaries using visual analysis (blank pages, layout changes)
- OCRs only the first page of each detected sub-document (~80% cost savings)
- Classifies all documents in batch
- Flags sensitive documents automatically

## Streamlit Applications

### Single Document Classifier

```bash
streamlit run pages/run_du_llm.py
```

Upload a document, select OCR language, and get instant classification with confidence scores.

### Multi-Document Visual Browser

```bash
streamlit run pages/multi_document_viewer.py
```

Upload a multi-page PDF bundle to:
- Visually browse pages with Previous/Next navigation
- See detected document boundaries
- View classification for each document
- Export results to JSON

## Prerequisites

- Python 3.8+
- Oracle Cloud Infrastructure (OCI) account with:
  - Document Understanding service access
  - Generative AI service access (Llama 3.3)
- OCI CLI configured (`~/.oci/config`)
- Poppler (for PDF processing): `brew install poppler` on macOS

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd uksbs-runtime
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure OCI credentials**
   - Set up your OCI config file at `~/.oci/config`
   - Update `config.py` with your `COMPARTMENT_ID` if different

## Configuration

All configuration is centralized in `config.py`:

```python
# OCI Compartment
COMPARTMENT_ID = "ocid1.compartment..."

# Model ID (Meta Llama 3.3 70B Instruct)
OCI_LLAMA_3_3_MODEL_ID = "ocid1.generativeaimodel..."

# LLM Parameters
DEFAULT_MAX_TOKENS = 4000
DEFAULT_TEMPERATURE = 0.0
```

## Sensitive Document Categories

The following document types are automatically flagged:
- Bank Details
- Passport / Driving License / ID
- Birth / Marriage Certificates
- BPSS / Security checks
- Health declarations and medical reports
- Disclosure statements
- Compromise agreements

## Output Format

Results are saved as JSON with:
- Primary category classification
- Confidence score (high/medium/low)
- Reasoning for classification
- Alternative category suggestions
- Sensitive document flags

---

Copyright (c) 2025 Oracle and/or its affiliates.

MIT License — see LICENSE for details.
