"""
ISRO Mission Scraper  v2  –  Fixed Table Parser
==================================================================
Root cause of v1 bug:
  - ISRO pages use JavaScript-rendered tables (list.js / DataTables).
    A plain requests.get() returns the full HTML including the <tbody>
    rows ONLY if the server renders them server-side, which it does —
    but the old heuristic was looking for <a> links instead of <td>
    cells, finding only 1 "Click here" anchor per page.

Fix applied:
  - Parse every <table> on each category page
  - Extract rows from <tbody class="list"> (or any <tbody>)
  - Map TD cells by their CSS class names first; fall back to column
    position for tables whose TDs have no class attribute
  - One unified extraction function handles all 7 category layouts

Category pages & their table column profiles:
  SpacecraftMissions  → Mission Name, Launch Date, Orbit, Remarks
  Student_Satellite   → Name, Organisation, Launch Date, Vehicle
  ForeignSatellites   → Satellite Name, Country, Launch Date, Mass, Vehicle
  LaunchMissions      → Mission/Vehicle, Launch Date, Remarks
  ReEntryMissions     → Mission, Launch Date, Remarks
  Indian_private      → Mission, Date, Remarks
  Gaganyaan           → single mission, no table
==================================================================
"""

import os, re, json, time, logging, hashlib, requests, io, sys
import io as _io
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin

from bs4 import BeautifulSoup

try:
    import fitz
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

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

try:
    from pymongo import MongoClient
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()

# ── UTF-8 stdout on Windows ──────────────────────────────────────
if sys.platform == "win32":
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("isro_scraper.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)
for _lib in ("httpx", "httpcore", "urllib3", "ddgs", "duckduckgo_search"):
    logging.getLogger(_lib).setLevel(logging.WARNING)

# ================================================================
# CONFIG
# ================================================================
ISRO_BASE    = "https://www.isro.gov.in"
ISRO_ROOT    = f"{ISRO_BASE}/index.html"
OUTPUT_JSON  = Path("isro_missions.json")
MONGO_URI    = os.getenv("MONGO_URI", "")
DB_NAME      = os.getenv("MONGO_DB",         "spacellm")
COLLECTION   = os.getenv("MONGO_COLLECTION", "missions_isro")
PDF_TIMEOUT  = 30
SEARCH_DELAY = 3
MAX_PDF_MB   = 60
DDG_RESULTS  = 15

# All 7 category pages with expected column hints
CATEGORY_PAGES = [
    {
        "category": "Spacecraft Missions",
        "url": f"{ISRO_BASE}/SpacecraftMissions.html",
        "name_classes": ["MissionName", "Mission_Name", "SatelliteName",
                         "mission_name", "name"],
    },
    {
        "category": "Student / Private Satellites",
        "url": f"{ISRO_BASE}/Student_Satellite.html",
        "name_classes": ["SatelliteName", "Name", "name", "MissionName"],
    },
    {
        "category": "Foreign Satellites Launched by ISRO",
        "url": f"{ISRO_BASE}/ForeignSatellites.html",
        "name_classes": ["SatelliteName", "Name", "name"],
    },
    {
        "category": "Launch Missions",
        "url": f"{ISRO_BASE}/LaunchMissions.html",
        "name_classes": ["MissionName", "LaunchVehicle", "Launch_Vehicle",
                         "Mission_Name", "name"],
    },
    {
        "category": "Re-entry Missions & POEMS",
        "url": f"{ISRO_BASE}/ReEntryMissions.html",
        "name_classes": ["MissionName", "Mission_Name", "name",
                         "SatelliteName"],
    },
    {
        "category": "Launch Missions Facilitated by ISRO",
        "url": f"{ISRO_BASE}/Indian_private_players.html",
        "name_classes": ["MissionName", "Mission_Name", "name",
                         "SatelliteName"],
    },
    {
        "category": "Gaganyaan",
        "url": f"{ISRO_BASE}/Gaganyaan_Mission.html",
        "name_classes": [],   # no table – text extraction
    },
]

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
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin" if referer else "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
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


# ================================================================
# SESSION  – prime cookies from homepage first
# ================================================================
_session: requests.Session | None = None

def get_session() -> requests.Session:
    global _session
    if _session is not None:
        return _session
    _session = requests.Session()
    try:
        log.info("Priming session from ISRO homepage ...")
        r = _session.get(ISRO_ROOT, headers=_browser_headers(), timeout=15)
        log.info(f"  Homepage status: {r.status_code}  cookies: {dict(_session.cookies)}")
    except Exception as e:
        log.warning(f"  Homepage prime failed: {e}")
    return _session


def fetch_page(url: str, referer: str = ISRO_BASE) -> BeautifulSoup | None:
    """Fetch a URL with session + retry on 403 (rotate UA)."""
    sess = get_session()
    for attempt in range(3):
        try:
            r = sess.get(url, headers=_browser_headers(referer=referer),
                         timeout=20)
            if r.status_code == 403:
                log.warning(f"  403 on attempt {attempt+1} for {url[:70]} – retrying …")
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return BeautifulSoup(r.text, "html.parser")
        except requests.exceptions.HTTPError as e:
            log.warning(f"  HTTP error: {e}")
            return None
        except Exception as e:
            log.warning(f"  Fetch error ({url[:60]}): {e}")
            time.sleep(1)
    log.error(f"  All attempts failed for: {url}")
    return None


# ================================================================
# TABLE PARSER  –  the core fix
# ================================================================

# Maps TD CSS class names → canonical field names.
# Add / extend as you discover new ISRO table layouts.
TD_CLASS_MAP: dict[str, str] = {
    # name variants
    "SatelliteName":    "name",
    "MissionName":      "name",
    "Mission_Name":     "name",
    "LaunchVehicle":    "launch_vehicle",   # some tables put name here
    # date variants
    "Date_of_Launch":   "launch_date",
    "Date_Of_Launch":   "launch_date",
    "LaunchDate":       "launch_date",
    "Launch_Date":      "launch_date",
    # country
    "Country":          "country",
    # mass
    "Mass_Kg":          "mass_kg",
    "Mass":             "mass_kg",
    # vehicle
    "Launch_Vehicle":   "launch_vehicle",
    "LaunchVehicleName":"launch_vehicle",
    # orbit / remarks
    "Orbit":            "orbit",
    "Remarks":          "remarks",
    "Remark":           "remarks",
    # serial / index (ignored)
    "counter":          "_sl",
    "SNo":              "_sl",
}

# Column-position fallbacks keyed by (number_of_columns,):
#   list of field names in left-to-right order
COL_FALLBACKS: dict[int, list[str]] = {
    6: ["_sl", "name", "country",  "launch_date", "mass_kg", "launch_vehicle"],
    5: ["_sl", "name", "launch_date", "launch_vehicle", "remarks"],
    4: ["_sl", "name", "launch_date", "remarks"],
    3: ["_sl", "name", "launch_date"],
    2: ["_sl", "name"],
}


def _td_class_to_field(td) -> str | None:
    """Return canonical field name for a <td> given its CSS class(es)."""
    for cls in td.get("class", []):
        if cls in TD_CLASS_MAP:
            return TD_CLASS_MAP[cls]
    return None


def parse_isro_table(table) -> list[dict]:
    """
    Extract all data rows from a BeautifulSoup <table> element.

    Strategy:
      1. Read <thead> to get column count and header labels.
      2. For each <tr> in <tbody> (any tbody), extract <td> cells.
         a. If TDs carry known CSS classes → use class map.
         b. Otherwise → map by column position using COL_FALLBACKS.
      3. Skip rows that are all-empty or only have a serial number.
    """
    # ── thead: count columns ──────────────────────────────────────
    thead = table.find("thead")
    th_count = len(thead.find_all("th")) if thead else 0

    rows_out = []

    # ── iterate all tbodies (some pages have >1) ──────────────────
    tbodies = table.find_all("tbody") or [table]
    for tbody in tbodies:
        for tr in tbody.find_all("tr"):
            tds = tr.find_all("td")
            if not tds:
                continue

            # Try class-based extraction first
            record: dict = {}
            used_class_map = False
            for td in tds:
                field = _td_class_to_field(td)
                if field:
                    used_class_map = True
                    if field != "_sl":          # skip serial numbers
                        record[field] = td.get_text(" ", strip=True)

            # Fallback: positional mapping
            if not used_class_map or not record:
                n = len(tds)
                fallback = COL_FALLBACKS.get(n, COL_FALLBACKS.get(
                    min(COL_FALLBACKS.keys(), key=lambda k: abs(k - n)),
                    ["_sl", "name"]
                ))
                for i, td in enumerate(tds):
                    field = fallback[i] if i < len(fallback) else f"col_{i}"
                    if field != "_sl":
                        record[field] = td.get_text(" ", strip=True)

            # Clean & validate
            record = {k: v for k, v in record.items()
                      if v and v not in ("-", "–", "N/A", "NA")}
            if not record or "name" not in record:
                continue
            if len(record.get("name", "")) < 2:
                continue

            rows_out.append(record)

    return rows_out


# ================================================================
# SCRAPE ONE CATEGORY PAGE
# ================================================================

def scrape_category(cat: dict) -> list[dict]:
    """
    Fetch a category page and return a list of mission dicts.
    Each dict has at minimum: name, category, source_url.
    Additional fields depend on what columns the table has.
    """
    category_name = cat["category"]
    url           = cat["url"]
    name_classes  = cat["name_classes"]

    log.info(f"  Fetching: {category_name}  ->  {url}")
    soup = fetch_page(url, referer=ISRO_BASE)
    if soup is None:
        log.warning(f"  Could not fetch: {url}")
        return []

    missions = []

    # ── 1. Try every <table> on the page ─────────────────────────
    tables = soup.find_all("table")
    if tables:
        for tbl_idx, tbl in enumerate(tables):
            rows = parse_isro_table(tbl)
            log.info(f"    table[{tbl_idx}]: {len(rows)} rows")
            for row in rows:
                mission = {
                    "name":       row.pop("name", "").strip(),
                    "category":   category_name,
                    "source_url": url,
                    **row,          # remaining fields (launch_date, country …)
                }
                if mission["name"]:
                    missions.append(mission)
    else:
        # ── 2. No table: try <li> / <a> links (student page etc.) ──
        log.info(f"    No <table> found – trying link/list extraction …")
        seen = set()
        for a in soup.find_all("a", href=True):
            name = a.get_text(strip=True)
            href = a["href"]
            if not name or len(name) < 3 or name.lower() in seen:
                continue
            if re.search(r"(mission|satellite|spacecraft|launch|isro)",
                         href, re.I):
                full_url = href if href.startswith("http") \
                           else urljoin(ISRO_BASE, href)
                seen.add(name.lower())
                missions.append({
                    "name":       name,
                    "category":   category_name,
                    "source_url": full_url,
                })

        # ── 3. Last resort: heading / paragraph text ───────────────
        if not missions:
            log.info(f"    Trying text fallback …")
            for tag in soup.find_all(["h1", "h2", "h3", "h4", "strong", "b"]):
                text = tag.get_text(strip=True)
                if 4 <= len(text) <= 120:
                    missions.append({
                        "name":       text,
                        "category":   category_name,
                        "source_url": url,
                    })

    log.info(f"  -> {len(missions)} missions in '{category_name}'")
    return missions


# ================================================================
# COLLECT ALL MISSIONS
# ================================================================

def scrape_all_missions() -> list[dict]:
    """Scrape all 7 category pages and return deduplicated list."""
    all_missions = []
    seen_keys    = set()          # (name.lower(), category) deduplicate

    for cat in CATEGORY_PAGES:
        entries = scrape_category(cat)
        for m in entries:
            key = (m["name"].lower(), m["category"])
            if key not in seen_keys:
                seen_keys.add(key)
                all_missions.append(m)
        time.sleep(1)             # polite crawl rate

    log.info(f"\nTotal unique missions across all categories: {len(all_missions)}")
    return all_missions


# ================================================================
# PDF PIPELINE  (unchanged from v1 – included for completeness)
# ================================================================

def _is_pdf_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    qs   = urlparse(url).query.lower()
    return (
        path.endswith(".pdf")
        or ("/pdf" in path and path.endswith("/"))
        or "format=pdf" in qs
        or "type=pdf"   in qs
    )


def _sniff_pdf_from_page(page_url: str) -> str | None:
    try:
        r = requests.get(page_url,
                         headers=_browser_headers(referer="https://duckduckgo.com/"),
                         timeout=10, stream=True)
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


def collect_pdf_candidates(mission_name: str) -> list[str]:
    if DDGS is None:
        return []
    query = f"ISRO mission Report pdf {mission_name}"
    log.info(f"  DDG [{DDGS_BACKEND}]: {query}")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=DDG_RESULTS))
    except Exception as e:
        log.warning(f"  DDG error: {e}")
        return []

    direct, pages = [], []
    for r in results:
        url = r.get("href") or r.get("url") or ""
        if not url:
            continue
        (direct if _is_pdf_url(url) else pages).append(url)

    sniffed = []
    for pu in pages[:6]:
        p = _sniff_pdf_from_page(pu)
        if p and p not in direct:
            sniffed.append(p)
    return direct + sniffed


def download_pdf(url: str) -> bytes | None:
    parsed  = urlparse(url)
    referer = f"{parsed.scheme}://{parsed.netloc}/"
    for attempt in range(2):
        headers = _pdf_headers(referer=referer)
        try:
            with requests.get(url, headers=headers, timeout=PDF_TIMEOUT,
                              stream=True) as r:
                if r.status_code in (403, 401):
                    time.sleep(1)
                    continue
                r.raise_for_status()
                ct = r.headers.get("Content-Type", "")
                if "html" in ct and "pdf" not in ct:
                    return None
                chunks, total = [], 0
                for chunk in r.iter_content(131_072):
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > MAX_PDF_MB * 1_000_000:
                        return None
                data = b"".join(chunks)
                return data if data.startswith(b"%PDF") else None
        except Exception as e:
            log.warning(f"  [download] {e}")
            return None
    return None


def extract_text_from_pdf(pdf_bytes: bytes) -> dict:
    if not FITZ_AVAILABLE:
        return {"pages": [], "method": "none", "page_count": 0}
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_text, methods = [], set()
    for pn, page in enumerate(doc, 1):
        native = page.get_text("text").strip()
        if len(native) >= 50:
            pages_text.append(native)
            methods.add("text")
        elif OCR_AVAILABLE:
            try:
                pix = page.get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr = pytesseract.image_to_string(img, config="--oem 3 --psm 6")
                pages_text.append(ocr.strip())
                methods.add("ocr")
            except Exception:
                pages_text.append("")
        else:
            pages_text.append("")
    doc.close()
    method = "mixed" if len(methods) > 1 else (methods.pop() if methods else "none")
    return {"pages": pages_text, "method": method, "page_count": len(pages_text)}


def extract_text_from_mission_page(source_url: str) -> dict | None:
    log.info(f"  [fallback-B] Extracting from: {source_url}")
    try:
        soup = fetch_page(source_url, referer=ISRO_BASE)
        if not soup:
            return None
        for tag in soup(["nav", "header", "footer", "script", "style",
                         "noscript", "aside", "form"]):
            tag.decompose()
        blocks = [
            t.get_text(" ", strip=True)
            for t in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "td"])
            if len(t.get_text(" ", strip=True)) > 30
        ]
        return {"pages": blocks, "method": "webpage",
                "page_count": len(blocks)} if blocks else None
    except Exception as e:
        log.warning(f"  [fallback-B] {e}")
        return None


def structure_mission_info(mission: dict, extraction: dict,
                           pdf_url: str) -> dict:
    full_text = "\n".join(extraction["pages"])

    def find(patterns):
        for pat in patterns:
            m = re.search(pat, full_text, re.IGNORECASE | re.DOTALL)
            if m:
                return " ".join(m.group(1).split())[:300]
        return ""

    # Merge table-scraped fields with text-extracted fields
    return {
        "mission_name": mission["name"],
        "category":     mission["category"],
        "pdf_url":      pdf_url,
        "pdf_checksum": "",
        "source_url":   mission.get("source_url", ""),
        "data_source":  "pdf" if pdf_url else "webpage",
        # direct table fields (may already be populated)
        "launch_date":  mission.get("launch_date", ""),
        "launch_vehicle": mission.get("launch_vehicle", ""),
        "country":      mission.get("country", ""),
        "mass_kg":      mission.get("mass_kg", ""),
        "orbit":        mission.get("orbit", ""),
        "remarks":      mission.get("remarks", ""),
        "extraction": {
            "method":     extraction["method"],
            "page_count": extraction["page_count"],
            "full_text":  full_text[:60_000],
        },
        "structured": {
            "launch_date": mission.get("launch_date") or find([
                r"launch\s+date[:\s]+([A-Za-z0-9 ,/\-]+)",
                r"launched[:\s]+([A-Za-z0-9 ,/\-]+)",
            ]),
            "objective": find([
                r"objective[s]?[:\s]+(.+?)(?:\n\n|\Z)",
                r"purpose[:\s]+(.+?)(?:\n\n|\Z)",
                r"mission overview[:\s]+(.+?)(?:\n\n|\Z)",
            ]),
            "agency": find([
                r"agency[:\s]+(.+?)(?:\n|\.)",
                r"managed by[:\s]+(.+?)(?:\n|\.)",
            ]),
            "spacecraft": find([
                r"spacecraft[:\s]+(.+?)(?:\n|\.)",
                r"satellite[:\s]+(.+?)(?:\n|\.)",
            ]),
            "orbit": mission.get("orbit") or find([
                r"orbit(?:al)? (?:type|altitude)[:\s]+(.+?)(?:\n|\.)",
                r"altitude[:\s]+(.+?)(?:\n|\.)",
            ]),
            "summary": next(
                (p[:500] for p in extraction["pages"] if len(p) >= 100), ""
            ),
        },
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


# ================================================================
# MONGODB
# ================================================================

def get_mongo_collection():
    if not MONGO_URI or not MONGO_AVAILABLE:
        log.warning("MONGO_URI not set or pymongo not installed – skipping MongoDB.")
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


# ================================================================
# JSON PERSISTENCE
# ================================================================

def save_to_json(record: dict):
    data: dict = {}
    if OUTPUT_JSON.exists():
        try:
            with open(OUTPUT_JSON, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            data = {}
    key = f"{record['category']}::{record['mission_name']}"
    data[key] = record
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log.info(f"  [OK] JSON saved -> {OUTPUT_JSON}")


# ================================================================
# PROCESS ONE MISSION  (table entry → optional PDF → persist)
# ================================================================

def process_mission(mission: dict) -> dict:
    name       = mission["name"]
    source_url = mission.get("source_url", "")

    # ── DDG search → PDF candidates
    time.sleep(SEARCH_DELAY)
    candidates = collect_pdf_candidates(name)

    pdf_bytes, used_pdf_url = None, ""

    for url in candidates:
        log.info(f"  Trying PDF: {url[:80]}")
        data = download_pdf(url)
        if data:
            pdf_bytes    = data
            used_pdf_url = url
            log.info(f"  [OK] PDF downloaded ({len(data)//1024} KB)")
            break
        log.warning(f"  [SKIP] {url[:80]}")

    # ── Fallback A: look for PDF on the mission's own page
    if not pdf_bytes and source_url:
        soup = fetch_page(source_url, referer=ISRO_BASE)
        if soup:
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.lower().endswith(".pdf"):
                    full = href if href.startswith("http") \
                           else urljoin(source_url, href)
                    data = download_pdf(full)
                    if data:
                        pdf_bytes    = data
                        used_pdf_url = full
                        log.info(f"  [OK] Fallback-A PDF ({len(data)//1024} KB)")
                        break

    # ── Extract text
    checksum = ""
    if pdf_bytes:
        try:
            extraction = extract_text_from_pdf(pdf_bytes)
            checksum   = hashlib.md5(pdf_bytes).hexdigest()
        except Exception as e:
            log.error(f"  PDF extraction error: {e}")
            pdf_bytes = None

    if not pdf_bytes:
        extraction = extract_text_from_mission_page(source_url) \
                     if source_url else None
        if not extraction:
            # Build a minimal record from table data alone
            extraction = {
                "pages": [
                    f"Mission: {name}",
                    f"Category: {mission.get('category', '')}",
                    f"Launch Date: {mission.get('launch_date', '')}",
                    f"Launch Vehicle: {mission.get('launch_vehicle', '')}",
                ],
                "method": "table_only",
                "page_count": 4,
            }
            used_pdf_url = ""

    # ── Build & persist record
    record = structure_mission_info(mission, extraction, used_pdf_url)
    record["pdf_checksum"] = checksum
    save_to_json(record)
    return {
        "name":     name,
        "status":   "ok",
        "method":   extraction["method"],
        "source":   "pdf" if pdf_bytes else "webpage/table",
    }


# ================================================================
# ORCHESTRATOR
# ================================================================

def run(limit: int | None = None,
        mission_filter: str | None = None,
        category_filter: str | None = None,
        skip: int = 0,
        scrape_only: bool = False):
    """
    Main entry point.

    scrape_only=True  – only collect mission names (no PDF/text extraction).
                        Useful to verify the table parser works first.
    """
    missions = scrape_all_missions()

    if category_filter:
        missions = [m for m in missions
                    if category_filter.lower() in m["category"].lower()]
        log.info(f"Category filter '{category_filter}' -> {len(missions)} missions.")

    if mission_filter:
        missions = [m for m in missions
                    if mission_filter.lower() in m["name"].lower()]
        log.info(f"Mission filter '{mission_filter}' -> {len(missions)} missions.")

    if skip:
        missions = missions[skip:]
        log.info(f"Skipping first {skip}.")

    if limit:
        missions = missions[:limit]

    # Print summary table before extracting
    log.info("\n" + "=" * 65)
    log.info("MISSION LIST")
    log.info("=" * 65)
    by_cat: dict[str, list] = {}
    for m in missions:
        by_cat.setdefault(m["category"], []).append(m["name"])
    for cat, names in by_cat.items():
        log.info(f"\n[{cat}]  ({len(names)} missions)")
        for n in names:
            log.info(f"  • {n}")

    if scrape_only:
        log.info("\n--scrape-only mode: stopping before PDF/text extraction.")
        return missions

    col     = get_mongo_collection()
    summary = []

    for idx, mission in enumerate(missions, start=1 + skip):
        log.info(f"\n[{idx}/{len(missions) + skip}] [{mission['category']}] {mission['name']}")
        result = process_mission(mission)
        summary.append(result)

        if result["status"] == "ok" and col is not None:
            try:
                with open(OUTPUT_JSON, encoding="utf-8") as f:
                    all_data = json.load(f)
                key = f"{mission['category']}::{mission['name']}"
                rec = all_data.get(key)
                if rec:
                    upsert_to_mongo(col, rec)
            except Exception as e:
                log.error(f"  Mongo upsert read error: {e}")

    # ── Final summary ─────────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("FINAL SUMMARY")
    log.info("=" * 65)
    ok   = [r for r in summary if r["status"] == "ok"]
    bad  = [r for r in summary if r["status"] != "ok"]
    pdf  = [r for r in ok if r.get("source") == "pdf"]
    page = [r for r in ok if r.get("source") != "pdf"]
    log.info(f"  Total processed  : {len(summary)}")
    log.info(f"  PDF success      : {len(pdf)}")
    log.info(f"  Webpage/table    : {len(page)}")
    log.info(f"  Failed (no data) : {len(bad)}")
    for r in bad:
        log.info(f"    [FAIL] {r['name']}")
    return summary


# ================================================================
# CLI
# ================================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="ISRO Mission Scraper v2")
    ap.add_argument("--limit",    type=int,  default=None,
                    help="Cap total missions processed")
    ap.add_argument("--filter",   type=str,  default=None,
                    help="Filter by mission name substring")
    ap.add_argument("--category", type=str,  default=None,
                    help="Filter by category name substring")
    ap.add_argument("--skip",     type=int,  default=0,
                    help="Skip first N missions (resume)")
    ap.add_argument("--scrape-only", action="store_true",
                    help="Only list missions; skip PDF/text extraction")
    args = ap.parse_args()
    run(
        limit           = args.limit,
        mission_filter  = args.filter,
        category_filter = args.category,
        skip            = args.skip,
        scrape_only     = args.scrape_only,
    )