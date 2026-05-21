"""
ESA Mission Scraper Engine  v1
==================================================================
Mirrors the NASA Mission Scraper v3 pipeline, adapted for ESA.

Pipeline for each mission:
  1. Scrape mission names from ESA "Our Missions" page
     (extracts <h3 class="heading"> inside .grid-item.mission cards)
  2. DuckDuckGo: "ESA mission Report pdf <name>"
     Collect ALL pdf candidates from results
  3. Try each PDF candidate with smart download (browser headers +
     referrer spoofing + retry). Move to next candidate on 403/404.
  4. FALLBACK A – scrape the ESA mission page itself (source_url)
     and extract any PDF links embedded there
  5. FALLBACK B – scrape the ESA mission page and extract all
     visible text directly (no PDF needed)
  6. Extract text from PDF with OCR fallback for scanned pages
  7. Save to esa_missions.json + MongoDB Atlas (both upserted)
==================================================================
"""

import os, re, json, time, logging, hashlib, requests, io, fitz, sys
import io as _io
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin

from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Optional MongoDB
try:
    from pymongo import MongoClient
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

# OCR – optional (warn if missing, don't crash)
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
# LOGGING – force UTF-8 on Windows so symbols don't crash
# ==================================================================
if sys.platform == "win32":
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("esa_scraper.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

for _lib in ("httpx", "httpcore", "urllib3", "ddgs", "duckduckgo_search"):
    logging.getLogger(_lib).setLevel(logging.WARNING)

# ==================================================================
# CONFIG
# ==================================================================
ESA_MISSIONS_URL = "https://www.esa.int/ESA/Our_Missions"
ESA_BASE         = "https://www.esa.int"
OUTPUT_JSON      = Path("esa_missions.json")
MONGO_URI        = os.getenv("MONGO_URI", "")
DB_NAME          = os.getenv("MONGO_DB", "esa_data")
COLLECTION       = os.getenv("MONGO_COLLECTION", "missions")
PDF_TIMEOUT      = 30
SEARCH_DELAY     = 3        # seconds between DDG queries
MAX_PDF_MB       = 60
DDG_RESULTS      = 15       # results to fetch per search

# Rotating browser User-Agents
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
# 1.  SCRAPE MISSION NAMES FROM ESA "OUR MISSIONS" PAGE
# ==================================================================

def _esa_session() -> requests.Session:
    """
    Build a requests.Session that mimics a browser visiting ESA.
    - Visits the ESA homepage first to collect cookies / CF clearance
    - Uses a consistent Accept header set expected by ESA's CDN
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENTS[0],
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    })
    # Warm-up: get ESA homepage to collect any session cookies
    try:
        session.get("https://www.esa.int/", timeout=15)
        time.sleep(1)
    except Exception:
        pass
    return session


def scrape_mission_names() -> list[dict]:
    """
    Fetch ESA's Our Missions grid page and extract mission cards.

    Target HTML structure:
        <div class="grid-item highlight mission">
          <a class="card ... mission ..." href="/Applications/.../Aeolus">
            <figure class="thumbnail">
              <header class="entry">
                <span class="pillar">Applications</span>
                <h3 class="heading">Aeolus</h3>
                <p class="description">Launched: 2018<br>...</p>
              </header>
            </figure>
          </a>
        </div>

    Falls back to a broader search if the strict selector finds nothing
    (ESA sometimes loads cards via JavaScript; we fetch the static HTML).

    NOTE: If ESA's page is JavaScript-rendered and returns an empty grid,
    install playwright (`pip install playwright && playwright install chromium`)
    and set ESA_USE_PLAYWRIGHT=1 in your .env – the function will switch
    automatically to a headless-browser fetch.
    """
    log.info("Fetching ESA Our Missions page ...")

    use_playwright = os.getenv("ESA_USE_PLAYWRIGHT", "0") == "1"
    html = ""

    if use_playwright:
        try:
            from playwright.sync_api import sync_playwright
            log.info("  Using Playwright (headless Chromium) ...")
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                page    = browser.new_page()
                page.goto(ESA_MISSIONS_URL, wait_until="networkidle", timeout=30_000)
                # Wait for at least one mission card
                try:
                    page.wait_for_selector("div.grid-item.mission", timeout=10_000)
                except Exception:
                    pass
                html = page.content()
                browser.close()
        except ImportError:
            log.warning("  Playwright not installed – falling back to requests.")

    if not html:
        session = _esa_session()
        resp = session.get(
            ESA_MISSIONS_URL,
            headers={"Referer": "https://www.esa.int/"},
            timeout=20,
        )
        resp.raise_for_status()
        html = resp.text
    soup = BeautifulSoup(html, "html.parser")

    missions, seen = [], set()

    # ── Primary: look for the exact card structure described in the spec
    for div in soup.find_all("div", class_=lambda c: c and "mission" in c.split()):
        a_tag = div.find("a", href=True)
        if not a_tag:
            continue
        href = a_tag.get("href", "")
        h3   = div.find("h3", class_="heading")
        name = h3.get_text(strip=True) if h3 else ""
        if not name or len(name) < 2:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        source_url = href if href.startswith("http") else ESA_BASE + href
        # Grab extra meta from the description paragraph
        desc_tag = div.find("p", class_="description")
        description = desc_tag.get_text(" ", strip=True) if desc_tag else ""
        pillar_tag = div.find("span", class_="pillar")
        pillar = pillar_tag.get_text(strip=True) if pillar_tag else ""
        missions.append({
            "name":        name,
            "source_url":  source_url,
            "pillar":      pillar,
            "description": description,
        })

    # ── Fallback: scan ALL <a> tags whose href contains a mission keyword
    if not missions:
        log.warning("Primary card parser found 0 missions – trying link fallback.")
        ESA_MISSION_RE = re.compile(
            r"/(?:Applications|Science_Exploration|Enabling_Support|Observing_the_Earth)/",
            re.IGNORECASE,
        )
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            if not ESA_MISSION_RE.search(href):
                continue
            name = a.get_text(strip=True)
            if not name or len(name) < 2 or name.lower() in seen:
                continue
            seen.add(name.lower())
            source_url = href if href.startswith("http") else ESA_BASE + href
            missions.append({"name": name, "source_url": source_url,
                              "pillar": "", "description": ""})

    log.info(f"Found {len(missions)} ESA mission entries.")
    return missions


# ==================================================================
# 2.  COLLECT PDF CANDIDATES via DuckDuckGo
# ==================================================================

def collect_pdf_candidates(mission_name: str) -> list[str]:
    """
    Search DDG and return ALL candidate PDF URLs found in results.
    Direct .pdf links come first; sniffed links follow.
    """
    if DDGS is None:
        log.error("Install ddgs:  pip install ddgs")
        return []

    query = f"ESA mission Report pdf {mission_name}"
    log.info(f"  DDG [{DDGS_BACKEND}]: {query}")

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=DDG_RESULTS))
    except Exception as e:
        log.warning(f"  DDG error: {e}")
        return []

    direct_pdfs, page_urls = [], []

    for r in results:
        url = r.get("href") or r.get("url") or ""
        if not url:
            continue
        if _is_pdf_url(url):
            log.info(f"  [candidate-direct] {url}")
            direct_pdfs.append(url)
        else:
            page_urls.append(url)

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
                return href if href.startswith("http") else urljoin(page_url, href)
    except Exception:
        pass
    return None


# ==================================================================
# 3.  DOWNLOAD PDF  (smart: rotate UA, use referer, retry once)
# ==================================================================

def download_pdf(url: str) -> bytes | None:
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

                if not data.startswith(b"%PDF"):
                    log.warning("  [download] Response is not a valid PDF (no %PDF header).")
                    return None

                return data

        except requests.exceptions.SSLError:
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
# 4.  FALLBACK A – scrape the ESA mission page for PDF links
# ==================================================================

def find_pdf_on_mission_page(source_url: str) -> str | None:
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
# 5.  FALLBACK B – extract text directly from the ESA mission page
# ==================================================================

def extract_text_from_mission_page(source_url: str) -> dict | None:
    """
    ESA mission pages use a consistent layout with rich article text.
    We strip navigation chrome and pull headings + paragraphs.
    """
    log.info(f"  [fallback-B] Extracting from mission page: {source_url}")
    try:
        r = requests.get(source_url, headers=_browser_headers(), timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Remove ESA chrome
        for tag in soup(["nav", "header", "footer", "script", "style",
                         "noscript", "aside", "form", "iframe"]):
            tag.decompose()

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
    mission: dict,
    extraction: dict,
    pdf_url: str,
) -> dict:
    full_text = "\n".join(extraction["pages"])

    def find(patterns: list) -> str:
        for pat in patterns:
            m = re.search(pat, full_text, re.IGNORECASE | re.DOTALL)
            if m:
                # Use group(1) when a capture group exists, else the whole match
                text = m.group(1) if m.lastindex and m.lastindex >= 1 else m.group(0)
                return " ".join(text.split())[:300]
        return ""

    return {
        "mission_name": mission["name"],
        "pdf_url":      pdf_url,
        "pdf_checksum": "",
        "source_url":   mission["source_url"],
        "pillar":       mission.get("pillar", ""),
        "description":  mission.get("description", ""),
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
                r"launched:\s*(\d{4})",                  # ESA cards often say "Launched: 2018"
            ]),
            "end_date": find([
                r"end\s+date[:\s]+([A-Za-z0-9 ,/\-]+)",
                r"decommission(?:ed)?[:\s]+([A-Za-z0-9 ,/\-]+)",
                r"mission end[:\s]+([A-Za-z0-9 ,/\-]+)",
                r"end of (?:mission|operations)[:\s]+([A-Za-z0-9 ,/\-]+)",
            ]),
            "objective": find([
                r"objective[s]?[:\s]+(.+?)(?:\n\n|\Z)",
                r"purpose[:\s]+(.+?)(?:\n\n|\Z)",
                r"mission overview[:\s]+(.+?)(?:\n\n|\Z)",
                r"goals?[:\s]+(.+?)(?:\n\n|\Z)",
                r"aims?\s+to\s+(.+?)(?:\.|$)",
            ]),
            "agency": find([
                r"agency[:\s]+(.+?)(?:\n|\.)",
                r"managed by[:\s]+(.+?)(?:\n|\.)",
                r"lead centre[:\s]+(.+?)(?:\n|\.)",       # ESA spelling
                r"lead center[:\s]+(.+?)(?:\n|\.)",
                r"prime contractor[:\s]+(.+?)(?:\n|\.)",
            ]),
            "spacecraft": find([
                r"spacecraft[:\s]+(.+?)(?:\n|\.)",
                r"satellite[:\s]+(.+?)(?:\n|\.)",
                r"vehicle[:\s]+(.+?)(?:\n|\.)",
                r"probe[:\s]+(.+?)(?:\n|\.)",
            ]),
            "orbit": find([
                r"orbit(?:al)? (?:type|altitude|parameters?)[:\s]+(.+?)(?:\n|\.)",
                r"altitude[:\s]+(.+?)(?:\n|\.)",
                r"inclination[:\s]+(.+?)(?:\n|\.)",
                r"sun[- ]synchronous orbit[:\s]*(.+?)(?:\n|\.)",
            ]),
            "instruments": find([
                r"instrument[s]?[:\s]+(.+?)(?:\n\n|\Z)",
                r"payload[s]?[:\s]+(.+?)(?:\n\n|\Z)",
                r"sensor[s]?[:\s]+(.+?)(?:\n\n|\Z)",
            ]),
            "launch_vehicle": find([
                r"launch vehicle[:\s]+(.+?)(?:\n|\.)",
                r"launched (?:by|on|aboard)[:\s]+(.+?)(?:\n|\.)",
                r"rocket[:\s]+(.+?)(?:\n|\.)",
                r"(ariane\s*\d+[^.]*)",                  # ESA uses Ariane rockets often
                r"(vega[^.]*)",
                r"(soyuz[^.]*)",
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
    if not MONGO_AVAILABLE:
        log.warning("pymongo not installed - skipping MongoDB.")
        return None
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
    Full pipeline for one ESA mission. Returns a status dict.

    Priority order:
      1. DDG search  -> try each PDF candidate until one downloads OK
      2. Fallback A  -> look for PDF links on the ESA mission page
      3. Fallback B  -> scrape text directly from the ESA mission page
    """
    name       = mission["name"]
    source_url = mission["source_url"]

    # Step 1: DDG search -> collect candidates
    time.sleep(SEARCH_DELAY)
    candidates = collect_pdf_candidates(name)

    pdf_bytes    = None
    used_pdf_url = ""

    # Step 2: Try each candidate PDF
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

    # Fallback A: check the ESA mission page for PDF links
    if not pdf_bytes:
        log.info("  No working PDF from search - trying mission page (fallback A) ...")
        page_pdf_url = find_pdf_on_mission_page(source_url)
        if page_pdf_url:
            data = download_pdf(page_pdf_url)
            if data:
                pdf_bytes    = data
                used_pdf_url = page_pdf_url
                log.info(f"  [OK] Fallback-A PDF downloaded ({len(data)//1024} KB)")

    # Extract from PDF
    checksum = ""
    extraction = None
    if pdf_bytes:
        try:
            extraction = extract_text_from_pdf(pdf_bytes)
            checksum   = hashlib.md5(pdf_bytes).hexdigest()
        except Exception as e:
            log.error(f"  PDF extraction error: {e}")
            pdf_bytes = None

    # Fallback B: scrape mission page directly
    if not pdf_bytes:
        log.info("  No PDF available - extracting from mission page (fallback B) ...")
        extraction = extract_text_from_mission_page(source_url)
        if not extraction:
            log.warning(f"  [FAIL] No data at all for: {name}")
            return {"name": name, "status": "no_data"}
        used_pdf_url = ""
        checksum     = ""

    # Structure and persist
    record = structure_mission_info(mission, extraction, used_pdf_url)
    record["pdf_checksum"] = checksum

    save_to_json(record)
    return {
        "name":   name,
        "status": "ok",
        "method": extraction["method"],
        "source": "pdf" if pdf_bytes else "webpage",
    }


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

        if result["status"] == "ok" and col is not None:
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
    ok      = [r for r in summary if r["status"] == "ok"]
    bad     = [r for r in summary if r["status"] != "ok"]
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
    ap = argparse.ArgumentParser(description="ESA Mission Scraper v1")
    ap.add_argument("--limit",  type=int,  default=None,
                    help="Max missions to process")
    ap.add_argument("--filter", type=str,  default=None,
                    help="Filter by mission name substring")
    ap.add_argument("--skip",   type=int,  default=0,
                    help="Skip first N missions (resume from checkpoint)")
    args = ap.parse_args()
    run(limit=args.limit, mission_filter=args.filter, skip=args.skip)