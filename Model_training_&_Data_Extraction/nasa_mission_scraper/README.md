# NASA Mission Scraper Engine

A fully automated pipeline that:
1. Scrapes every NASA mission name from the official A-to-Z page
2. Searches Google for the highest-ranked PDF for each mission
3. Extracts text from the PDF — with **OCR fallback** for scanned / handwritten pages
4. Saves structured records to a local **JSON file** and **MongoDB Atlas**

---

## Project Layout

```
nasa_scraper/
├── scraper.py          ← Main engine (all logic lives here)
├── requirements.txt    ← Python dependencies
├── .env.example        ← Copy to .env and fill your credentials
├── nasa_missions.json  ← Created at runtime (output)
└── nasa_scraper.log    ← Created at runtime (logs)
```

---

## Prerequisites

### System packages

| Tool | Purpose | Install |
|------|---------|---------|
| **Python 3.11+** | Runtime | [python.org](https://python.org) |
| **Tesseract OCR** | Handwritten / scanned PDF pages | See below |

#### Install Tesseract

```bash
# macOS
brew install tesseract

# Ubuntu / Debian
sudo apt install tesseract-ocr

# Windows
# Download installer from https://github.com/UB-Mannheim/tesseract/wiki
# Add install path to your system PATH
```

---

## Setup

```bash
# 1. Clone / copy this folder
cd nasa_scraper

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Configure credentials
cp .env.example .env
# → Open .env and paste your MongoDB Atlas URI
```

---

## MongoDB Atlas — Quick Setup

1. Go to [cloud.mongodb.com](https://cloud.mongodb.com) → create a free M0 cluster
2. **Database Access** → add a user with read/write permissions
3. **Network Access** → add your IP (or `0.0.0.0/0` for testing)
4. **Connect** → choose "Connect your application" → copy the URI
5. Paste it into `.env` as `MONGO_URI`

---

## Running the Scraper

```bash
# Process ALL missions (takes a long time — hundreds of missions)
python scraper.py

# Process only the first 5 missions (great for testing)
python scraper.py --limit 5

# Filter missions by name substring
python scraper.py --filter "Hubble"

# Combine: first 3 missions matching 'Apollo'
python scraper.py --filter "Apollo" --limit 3
```

---

## Output

### `nasa_missions.json`

Each key is the mission name; the value is a structured record:

```json
{
  "Hubble Space Telescope": {
    "mission_name": "Hubble Space Telescope",
    "source_url": "https://science.nasa.gov/mission/hubble/",
    "pdf_url": "https://example.nasa.gov/hubble_overview.pdf",
    "pdf_checksum": "a1b2c3d4...",
    "extraction": {
      "method": "text",          // "text" | "ocr" | "mixed"
      "page_count": 12,
      "full_text": "..."
    },
    "structured": {
      "launch_date": "April 24, 1990",
      "objective": "Observe the universe in ultraviolet, visible, and near-infrared light",
      "agency": "NASA / ESA",
      "spacecraft": "HST",
      "orbit": "Low Earth Orbit, ~340 miles altitude",
      "summary": "The Hubble Space Telescope is one of the largest ..."
    },
    "scraped_at": "2024-01-15T10:30:00Z"
  }
}
```

### MongoDB Atlas

Same document shape, stored in `nasa_data.missions` (configurable via `.env`).
Each mission is **upserted** by `mission_name`, so re-running is safe.

---

## How OCR Works

Pages are processed in two passes:

```
PDF page
  │
  ├─ Native text layer present?  ──yes──▶  Use PyMuPDF text extraction
  │
  └─ No text / image-only page   ──────▶  Rasterise at 200 DPI with PyMuPDF
                                           │
                                           └─ Tesseract OCR → text string
```

The `extraction.method` field reports `"text"`, `"ocr"`, or `"mixed"` per document.

---

## Rate Limiting

The engine sleeps **2 seconds** between Google searches (`SEARCH_DELAY` in `scraper.py`) to be a polite client. For large runs, consider:
- Using a paid search API (SerpAPI, Bing Search) instead of `googlesearch-python`
- Increasing `SEARCH_DELAY`
- Running in batches overnight

---

## Extending the Engine

| Want to… | Change |
|----------|--------|
| Use SerpAPI instead of Google | Replace `find_best_pdf()` with a SerpAPI call |
| Store more structured fields | Add patterns in `structure_mission_info()` |
| Export to CSV | Add a `save_to_csv()` function next to `save_to_json()` |
| Run in parallel | Wrap the loop in `concurrent.futures.ThreadPoolExecutor` |
| Add a retry mechanism | Wrap download with `tenacity` retry decorator |

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `TesseractNotFoundError` | Install Tesseract and ensure it's on PATH |
| `pymongo.errors.ServerSelectionTimeoutError` | Check Atlas URI and Network Access whitelist |
| Google search returns empty | Google rate-limited you — add a longer delay or use a paid API |
| PDF download timeout | Increase `PDF_TIMEOUT` in `scraper.py` |
