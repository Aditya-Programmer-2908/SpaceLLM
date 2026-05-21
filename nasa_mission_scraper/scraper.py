"""
NASA Mission Scraper Engine  v3
==================================================================
Pipeline for each mission:
  1. Scrape mission names from NASA A-to-Z page (A-Z order, p tags only)
  2. DuckDuckGo: "NASA mission Report pdf <name>"
     Collect ALL pdf candidates from results (not just the first)
  3. Try each PDF candidate with smart download (browser headers +
     referrer spoofing + retry). Move to next candidate on 403/404.
  4. FALLBACK A - scrape the NASA mission page itself (source_url)
     and extract any PDF links embedded there
  5. FALLBACK B - scrape the NASA mission page and extract all
     visible text directly (no PDF needed)
  6. Extract text from PDF with OCR fallback for scanned pages
  7. Save to nasa_missions.json + MongoDB Atlas (both upserted)
==================================================================
"""

import os, re, json, time, logging, hashlib, requests, io, fitz, sys
import io as _io
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin

from bs4 import BeautifulSoup
from pymongo import MongoClient
from dotenv import load_dotenv

# OCR - optional (warn if missing, don't crash)
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# DuckDuckGo
try:
    from ddgs import DDGS
    DDGS_BACKEND = "ddgs"
except ImportError:
    try:
        from duckduckgo_search import DDGS
        DDGS_BACKEND = "duckduckgo_search"
    except ImportError:
        DDGS = None
        DDGS_BACKEND = None

load_dotenv()

# ==================================================================
# LOGGING - force UTF-8 on Windows so arrows / symbols don't crash
# ==================================================================
if sys.platform == "win32":
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("nasa_scraper.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# Silence noisy library loggers
for _lib in ("httpx", "httpcore", "urllib3", "ddgs", "duckduckgo_search"):
    logging.getLogger(_lib).setLevel(logging.WARNING)

# ==================================================================
# CONFIG
# ==================================================================
NASA_AZ_URL  = "https://www.nasa.gov/a-to-z-of-nasa-missions/"
OUTPUT_JSON  = Path("nasa_missions.json")
MONGO_URI    = os.getenv("MONGO_URI", "")
DB_NAME      = os.getenv("MONGO_DB", "nasa_data")
COLLECTION   = os.getenv("MONGO_COLLECTION", "missions")
PDF_TIMEOUT  = 30
SEARCH_DELAY = 3        # seconds between DDG queries
MAX_PDF_MB   = 60
DDG_RESULTS  = 15       # results to fetch per search

MISSION_URL_RE = re.compile(
    r"https?://(science\.nasa\.gov/mission/|www\.nasa\.gov/mission/)",
    re.IGNORECASE,
)

# Rotating browser User-Agents to reduce 403s
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
]
_ua_idx = 0

def _next_ua() -> str:
    global _ua_idx
    ua = USER_AGENTS[_ua_idx % len(USER_AGENTS)]
    _ua_idx += 1
    return ua

def _browser_headers(referer: str = "") -> dict:
    h = {
        "User-Agent": _next_ua(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    if referer:
        h["Referer"] = referer
    return h

def _pdf_headers(referer: str = "") -> dict:
    """Headers specifically for PDF downloads."""
    h = {
        "User-Agent": _next_ua(),
        "Accept": "application/pdf,application/octet-stream,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }
    if referer:
        h["Referer"] = referer
    return h


# ==================================================================
# 1.  SCRAPE MISSION NAMES  (A-Z order, <p> tags only)
# ==================================================================

def scrape_mission_names() -> list[dict]:
    """
    Collect real mission entries from the NASA A-to-Z page.
    Only <a> inside <p> tags that point to /mission/ URLs.
    Nav menu links (inside <li>) are skipped automatically.
    """
    log.info("Fetching NASA A-to-Z missions page ...")
    resp = requests.get(NASA_AZ_URL, headers=_browser_headers(), timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    missions, seen = [], set()
    for p in soup.find_all("p"):
        a = p.find("a", href=True)
        if not a:
            continue
        href = a.get("href", "")
        name = a.get_text(strip=True)
        if not MISSION_URL_RE.search(href) or not name or len(name) < 3:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        if href.startswith("/"):
            href = "https://www.nasa.gov" + href
        missions.append({"name": name, "source_url": href})

    log.info(f"Found {len(missions)} real mission entries (A-Z order).")
    return missions


# ==================================================================
# 2.  COLLECT PDF CANDIDATES via DuckDuckGo
# ==================================================================

def collect_pdf_candidates(mission_name: str) -> list[str]:
    """
    Search DDG and return ALL candidate PDF URLs found in results.
    Does NOT download anything - just collects URLs to try later.
    Returns a list ordered by confidence (direct .pdf links first).
    """
    if DDGS is None:
        log.error("Install ddgs:  pip install ddgs")
        return []

    query = f"NASA mission Report pdf {mission_name}"
    log.info(f"  DDG [{DDGS_BACKEND}]: {query}")

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=DDG_RESULTS))
    except Exception as e:
        log.warning(f"  DDG error: {e}")
        return []

    direct_pdfs = []
    page_urls   = []

    for r in results:
        url = r.get("href") or r.get("url") or ""
        if not url:
            continue
        if _is_pdf_url(url):
            log.info(f"  [candidate-direct] {url}")
            direct_pdfs.append(url)
        else:
            page_urls.append(url)

    # Sniff result pages for embedded PDF links
    sniffed_pdfs = []
    for page_url in page_urls[:6]:
        pdf = _sniff_pdf_from_page(page_url)
        if pdf and pdf not in direct_pdfs:
            log.info(f"  [candidate-sniff]  {pdf}  (from {page_url[:60]})")
            sniffed_pdfs.append(pdf)

    return direct_pdfs + sniffed_pdfs


def _is_pdf_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    qs   = urlparse(url).query.lower()
    return (
        path.endswith(".pdf")
        or ("/pdf" in path and path.endswith("/"))
        or ("format=pdf" in qs)
        or ("type=pdf" in qs)
    )


def _sniff_pdf_from_page(page_url: str) -> str | None:
    """Fetch the first 80 KB of a page and look for .pdf links."""
    try:
        r = requests.get(
            page_url,
            headers=_browser_headers(referer="https://duckduckgo.com/"),
            timeout=10,
            stream=True,
        )
        content = b""
        for chunk in r.iter_content(chunk_size=20_000):
            content += chunk
            if len(content) >= 80_000:
                break
        r.close()
        soup = BeautifulSoup(content, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".pdf"):
                if href.startswith("http"):
                    return href
                return urljoin(page_url, href)
    except Exception:
        pass
    return None


# ==================================================================
# 3.  DOWNLOAD PDF  (smart: rotate UA, use referer, retry once)
# ==================================================================

def download_pdf(url: str) -> bytes | None:
    """
    Try to download a PDF using browser-like headers.
    - Uses the domain root as Referer (avoids hotlink blocks)
    - Rotates User-Agent on retry
    - Returns None on permanent failure (403, 404, too large)
    """
    parsed  = urlparse(url)
    referer = f"{parsed.scheme}://{parsed.netloc}/"

    for attempt in range(2):
        headers = _pdf_headers(referer=referer)
        try:
            with requests.get(url, headers=headers, timeout=PDF_TIMEOUT, stream=True) as r:
                if r.status_code in (403, 401):
                    log.warning(f"  [download] {r.status_code} on attempt {attempt+1}: {url[:70]}")
                    time.sleep(1)
                    continue
                r.raise_for_status()

                content_length = int(r.headers.get("Content-Length", 0))
                if content_length > MAX_PDF_MB * 1_000_000:
                    log.warning(f"  [download] Too large ({content_length/1e6:.0f} MB), skip.")
                    return None

                # Verify it's actually a PDF
                content_type = r.headers.get("Content-Type", "")
                if "html" in content_type and "pdf" not in content_type:
                    log.warning(f"  [download] Not a PDF (Content-Type: {content_type[:40]})")
                    return None

                chunks, total = [], 0
                for chunk in r.iter_content(chunk_size=131_072):
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > MAX_PDF_MB * 1_000_000:
                        log.warning("  [download] Exceeded size limit mid-stream, skip.")
                        return None
                data = b"".join(chunks)

                # Final check: real PDFs start with %PDF
                if not data.startswith(b"%PDF"):
                    log.warning("  [download] Response is not a valid PDF (no %PDF header).")
                    return None

                return data

        except requests.exceptions.SSLError:
            # Some old gov sites have broken SSL - retry without verification
            try:
                with requests.get(url, headers=headers, timeout=PDF_TIMEOUT,
                                  stream=True, verify=False) as r:
                    r.raise_for_status()
                    data = r.content
                    return data if data.startswith(b"%PDF") else None
            except Exception as e2:
                log.warning(f"  [download] SSL retry failed: {e2}")
                return None
        except Exception as e:
            log.warning(f"  [download] Error: {e}")
            return None

    return None


# ==================================================================
# 4.  FALLBACK A - scrape the NASA mission page for PDF links
# ==================================================================

def find_pdf_on_mission_page(source_url: str) -> str | None:
    """
    Fetch the mission's own NASA page and return the first PDF link
    found in the page body.
    """
    try:
        r = requests.get(source_url, headers=_browser_headers(), timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".pdf"):
                url = href if href.startswith("http") else urljoin(source_url, href)
                log.info(f"  [fallback-A] PDF on mission page: {url[:80]}")
                return url
    except Exception as e:
        log.warning(f"  [fallback-A] Could not fetch mission page: {e}")
    return None


# ==================================================================
# 5.  FALLBACK B - extract text directly from the NASA mission page
# ==================================================================

def extract_text_from_mission_page(source_url: str) -> dict | None:
    """
    When no PDF is available, scrape the mission page HTML and
    extract meaningful text content. Returns the same dict shape
    as extract_text_from_pdf() so the rest of the pipeline works.
    """
    log.info(f"  [fallback-B] Extracting from mission page: {source_url}")
    try:
        r = requests.get(source_url, headers=_browser_headers(), timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Remove nav / footer / script noise
        for tag in soup(["nav", "header", "footer", "script", "style",
                         "noscript", "aside", "form"]):
            tag.decompose()

        # Collect paragraphs and headings
        blocks = []
        for tag in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "td"]):
            text = tag.get_text(" ", strip=True)
            if len(text) > 30:
                blocks.append(text)

        if not blocks:
            return None

        return {
            "pages":      blocks,
            "method":     "webpage",
            "page_count": len(blocks),
        }
    except Exception as e:
        log.warning(f"  [fallback-B] Failed: {e}")
        return None


# ==================================================================
# 6.  EXTRACT TEXT FROM PDF  (OCR fallback for scanned pages)
# ==================================================================

def extract_text_from_pdf(pdf_bytes: bytes) -> dict:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_text, methods_used = [], set()

    for page_num, page in enumerate(doc, start=1):
        native = page.get_text("text").strip()

        if len(native) >= 50:
            pages_text.append(native)
            methods_used.add("text")
        elif OCR_AVAILABLE:
            log.debug(f"    Page {page_num}: image-only -> OCR")
            try:
                pix = page.get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_text = pytesseract.image_to_string(img, config="--oem 3 --psm 6")
                pages_text.append(ocr_text.strip())
                methods_used.add("ocr")
            except Exception as e:
                log.warning(f"    OCR page {page_num}: {e}")
                pages_text.append("")
                methods_used.add("ocr")
        else:
            pages_text.append("")
            methods_used.add("text")

    doc.close()

    if not methods_used:
        method = "none"
    elif len(methods_used) == 1:
        method = methods_used.pop()
    else:
        method = "mixed"

    return {"pages": pages_text, "method": method, "page_count": len(pages_text)}


# ==================================================================
# 7.  STRUCTURE EXTRACTED TEXT INTO FIELDS
# ==================================================================

def structure_mission_info(
    mission_name: str,
    extraction: dict,
    pdf_url: str,
) -> dict:
    full_text = "\n".join(extraction["pages"])

    def find(patterns: list) -> str:
        for pat in patterns:
            m = re.search(pat, full_text, re.IGNORECASE | re.DOTALL)
            if m:
                return " ".join(m.group(1).split())[:300]
        return ""

    return {
        "mission_name": mission_name,
        "pdf_url":      pdf_url,
        "pdf_checksum": "",
        "source_url":   "",
        "data_source":  "pdf" if pdf_url else "webpage",
        "extraction": {
            "method":     extraction["method"],
            "page_count": extraction["page_count"],
            "full_text":  full_text[:60_000],
        },
        "structured": {
            "launch_date": find([
                r"launch\s+date[:\s]+([A-Za-z0-9 ,/\-]+)",
                r"launched[:\s]+([A-Za-z0-9 ,/\-]+)",
                r"launch[:\s]+([A-Za-z]+ \d{1,2},?\s+\d{4})",
            ]),
            "end_date": find([
                r"end\s+date[:\s]+([A-Za-z0-9 ,/\-]+)",
                r"decommission(?:ed)?[:\s]+([A-Za-z0-9 ,/\-]+)",
                r"mission end[:\s]+([A-Za-z0-9 ,/\-]+)",
            ]),
            "objective": find([
                r"objective[s]?[:\s]+(.+?)(?:\n\n|\Z)",
                r"purpose[:\s]+(.+?)(?:\n\n|\Z)",
                r"mission overview[:\s]+(.+?)(?:\n\n|\Z)",
                r"goals?[:\s]+(.+?)(?:\n\n|\Z)",
            ]),
            "agency": find([
                r"agency[:\s]+(.+?)(?:\n|\.)",
                r"managed by[:\s]+(.+?)(?:\n|\.)",
                r"lead center[:\s]+(.+?)(?:\n|\.)",
            ]),
            "spacecraft": find([
                r"spacecraft[:\s]+(.+?)(?:\n|\.)",
                r"satellite[:\s]+(.+?)(?:\n|\.)",
                r"vehicle[:\s]+(.+?)(?:\n|\.)",
            ]),
            "orbit": find([
                r"orbit(?:al)? (?:type|altitude|parameters?)[:\s]+(.+?)(?:\n|\.)",
                r"altitude[:\s]+(.+?)(?:\n|\.)",
                r"inclination[:\s]+(.+?)(?:\n|\.)",
            ]),
            "instruments": find([
                r"instrument[s]?[:\s]+(.+?)(?:\n\n|\Z)",
                r"payload[s]?[:\s]+(.+?)(?:\n\n|\Z)",
            ]),
            "summary": next(
                (p[:500] for p in extraction["pages"] if len(p) >= 100), ""
            ),
        },
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


# ==================================================================
# 8.  MONGODB
# ==================================================================

def get_mongo_collection():
    if not MONGO_URI:
        log.warning("MONGO_URI not set - skipping MongoDB.")
        return None
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        col = client[DB_NAME][COLLECTION]
        col.create_index("mission_name", unique=True)
        log.info(f"MongoDB connected -> {DB_NAME}.{COLLECTION}")
        return col
    except Exception as e:
        log.error(f"MongoDB connection failed: {e}")
        return None


def upsert_to_mongo(col, record: dict) -> bool:
    try:
        col.update_one(
            {"mission_name": record["mission_name"]},
            {"$set": record},
            upsert=True,
        )
        log.info(f"  [OK] MongoDB upserted: {record['mission_name']}")
        return True
    except Exception as e:
        log.error(f"  [ERR] MongoDB: {e}")
        return False


# ==================================================================
# 9.  JSON
# ==================================================================

def save_to_json(record: dict):
    data: dict = {}
    if OUTPUT_JSON.exists():
        try:
            with open(OUTPUT_JSON, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            data = {}
    data[record["mission_name"]] = record
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log.info(f"  [OK] JSON saved -> {OUTPUT_JSON}")


# ==================================================================
# 10.  ORCHESTRATOR
# ==================================================================

def process_mission(mission: dict) -> dict:
    """
    Full pipeline for one mission. Returns a status dict.

    Priority order:
      1. DDG search -> try each PDF candidate until one downloads OK
      2. Fallback A -> look for PDF links on the NASA mission page
      3. Fallback B -> scrape text directly from the NASA mission page
    """
    name       = mission["name"]
    source_url = mission["source_url"]

    # ── Step 1: DDG search -> collect candidates
    time.sleep(SEARCH_DELAY)
    candidates = collect_pdf_candidates(name)

    pdf_bytes = None
    used_pdf_url = ""

    # ── Step 2: Try each candidate PDF
    for url in candidates:
        log.info(f"  Trying PDF: {url[:80]}")
        data = download_pdf(url)
        if data:
            pdf_bytes    = data
            used_pdf_url = url
            log.info(f"  [OK] Downloaded PDF ({len(data)//1024} KB)")
            break
        else:
            log.warning(f"  [SKIP] Could not download: {url[:80]}")

    # ── Fallback A: check the NASA mission page for PDF links
    if not pdf_bytes:
        log.info("  No working PDF from search - trying mission page (fallback A) ...")
        page_pdf_url = find_pdf_on_mission_page(source_url)
        if page_pdf_url:
            data = download_pdf(page_pdf_url)
            if data:
                pdf_bytes    = data
                used_pdf_url = page_pdf_url
                log.info(f"  [OK] Fallback-A PDF downloaded ({len(data)//1024} KB)")

    # ── Extract from PDF
    if pdf_bytes:
        try:
            extraction = extract_text_from_pdf(pdf_bytes)
            checksum   = hashlib.md5(pdf_bytes).hexdigest()
        except Exception as e:
            log.error(f"  PDF extraction error: {e}")
            pdf_bytes = None  # fall through to webpage extraction

    # ── Fallback B: scrape the mission page directly
    if not pdf_bytes:
        log.info("  No PDF available - extracting from mission page (fallback B) ...")
        extraction = extract_text_from_mission_page(source_url)
        if not extraction:
            log.warning(f"  [FAIL] No data at all for: {name}")
            return {"name": name, "status": "no_data"}
        used_pdf_url = ""
        checksum     = ""

    # ── Structure and persist
    record = structure_mission_info(name, extraction, used_pdf_url)
    record["pdf_checksum"] = checksum if pdf_bytes else ""
    record["source_url"]   = source_url

    save_to_json(record)
    return {"name": name, "status": "ok",
            "method": extraction["method"],
            "source": "pdf" if pdf_bytes else "webpage"}


def run(limit: int | None = None, mission_filter: str | None = None,
        skip: int = 0):
    missions = scrape_mission_names()

    if mission_filter:
        missions = [m for m in missions
                    if mission_filter.lower() in m["name"].lower()]
        log.info(f"Filter '{mission_filter}' -> {len(missions)} missions.")

    if skip:
        missions = missions[skip:]
        log.info(f"Skipping first {skip}.")

    if limit:
        missions = missions[:limit]

    col     = get_mongo_collection()
    summary = []

    for idx, mission in enumerate(missions, start=1 + skip):
        log.info(f"\n[{idx}/{len(missions) + skip}] {mission['name']}")
        result = process_mission(mission)
        summary.append(result)

        # Upsert to Mongo if we have a record
        if result["status"] == "ok" and col is not None:
            # Re-read the just-saved record from JSON for Mongo
            try:
                with open(OUTPUT_JSON, encoding="utf-8") as f:
                    all_data = json.load(f)
                rec = all_data.get(mission["name"])
                if rec:
                    upsert_to_mongo(col, rec)
            except Exception as e:
                log.error(f"  Mongo upsert read error: {e}")

    # Summary
    log.info("\n" + "=" * 65)
    log.info("FINAL SUMMARY")
    log.info("=" * 65)
    ok  = [r for r in summary if r["status"] == "ok"]
    bad = [r for r in summary if r["status"] != "ok"]
    pdf_ok  = [r for r in ok if r.get("source") == "pdf"]
    page_ok = [r for r in ok if r.get("source") == "webpage"]
    log.info(f"  Total processed : {len(summary)}")
    log.info(f"  PDF success     : {len(pdf_ok)}")
    log.info(f"  Webpage fallback: {len(page_ok)}")
    log.info(f"  Failed (no data): {len(bad)}")
    for r in bad:
        log.info(f"    [FAIL] {r['name']}")
    return summary


# ==================================================================
# CLI
# ==================================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="NASA Mission Scraper v3")
    ap.add_argument("--limit",  type=int, default=None)
    ap.add_argument("--filter", type=str, default=None,
                    help="Filter by mission name substring")
    ap.add_argument("--skip",   type=int, default=0,
                    help="Skip first N missions (resume)")
    args = ap.parse_args()
    run(limit=args.limit, mission_filter=args.filter, skip=args.skip)