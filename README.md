# Document Understanding + LLM Document Classification

A Python application that combines Oracle Document Understanding (OCR) with Meta Llama 3.3 LLM to automatically classify and categorize HR/Finance documents.

## Features

- **Document Classification**: Automatically classifies documents into 90+ HR/Finance categories
- **Batch Processing**: Process large Excel files with document metadata using `1_categorize_documents.py`
- **Multi-Document Bundle Analysis**: Process 70+ page scanned bundles containing multiple documents
- **Interactive Web UI**: Upload and classify individual documents via Streamlit app
- **Sensitive Document Detection**: Flags documents containing potentially sensitive information
- **Multi-language OCR Support**: English, Japanese, Chinese, Arabic, German, French, and more
- **Confidence Scoring**: Provides classification confidence levels
- **Centralized Configuration**: All OCI settings in one place for easy management

## Architecture

The codebase uses a centralized configuration approach to reduce redundancy:

```
├── config.py                       # Centralized OCI and LLM configuration
├── oci_utils.py                    # Shared OCI client utilities
├── categories.json                 # Document categories list (90+ categories)
├── 1_categorize_documents.py       # Batch Excel document categorization
├── 2_analyze_categories.py         # Analyze categorization results
├── 3_classify_multi_document.py    # Multi-document bundle classifier (NEW)
├── pages/
│   └── run_du_llm.py              # Streamlit interactive classifier
├── outputs/                        # Classification results
└── requirements.txt                # Python dependencies
```

### Key Modules

- **config.py**: Single source of truth for all configuration (compartment ID, model IDs, endpoints, file paths)
- **oci_utils.py**: Reusable OCI client initialization and helper functions (eliminates code duplication)
- **categories.json**: Master list of HR/Finance document categories

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
   - Update `config.py` with your:
     - `COMPARTMENT_ID`
     - `OCI_LLAMA_3_3_MODEL_ID` (if different from default)

## Usage

### 1. Batch Document Categorization

Process an Excel file containing document metadata:

```bash
python 1_categorize_documents.py
```

This reads `Orphaned documents for HR and Finance.xlsx` and outputs categorized results.

### 2. Analyze Results

View statistics on categorization results:

```bash
python 2_analyze_categories.py
```

### 3. Multi-Document Bundle Classifier (NEW)

Process large PDF files containing multiple scanned documents (e.g., 70 pages):

```bash
python 3_classify_multi_document.py <path_to_pdf>

# Example:
python 3_classify_multi_document.py employee_file_bundle.pdf
```

**How it works:**
1. **Smart boundary detection** - Identifies where one document ends and another begins using:
   - Blank page detection (separator pages)
   - Visual similarity analysis (perceptual hashing)
   - Layout pattern changes

2. **Selective OCR** - Only processes the first page of each detected sub-document (~80% cost reduction)

3. **Batch classification** - Classifies all found documents in a single LLM call

4. **Sensitive document flagging** - Automatically flags documents like passports, bank details, etc.

**Example output:**
```json
{
  "filename": "employee_bundle.pdf",
  "total_pages": 70,
  "documents_found": 8,
  "pages_ocrd": 8,
  "sensitive_documents_found": 2,
  "documents": [
    {"pages": "1-12", "category": "Contract", "confidence": "high", "is_sensitive": false},
    {"pages": "13-14", "category": "Passport", "confidence": "high", "is_sensitive": true},
    {"pages": "15-25", "category": "Bank Details", "confidence": "medium", "is_sensitive": true}
  ]
}
```

### 4. Interactive Web Classifier

Run the Streamlit application for single document classification:

```bash
streamlit run pages/run_du_llm.py
```

Then:
1. Upload a PDF or image document
2. Select OCR language (Auto-detect recommended)
3. View classification results with confidence scores
4. Results are automatically saved to `outputs/`

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

The following document types are automatically flagged as sensitive:
- Bank Details
- Passport
- Driving License
- ID
- Birth/Marriage Certificates
- BPSS / Security checks
- Health declarations and medical reports
- Disclosure statements
- Compromise agreements

## Output

Classification results are saved to `outputs/` as JSON files containing:
- Primary category classification
- Confidence score (high/medium/low)
- Reasoning for classification
- Alternative category suggestions
- Sensitive document flags
- OCR language used
- Model information

## License

Proprietary
