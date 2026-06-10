"""
EtherCAT device identification — vendor names, product decoding, online lookup.

Identification sources, in order of preference:
  1. CoE Identity Object (0x1018) read from the device itself.
  2. Built-in offline vendor table (subset of the ETG registry).
  3. Optional online refresh of the official ETG vendor-ID registry
     (https://www.ethercat.org/en/vendor_id_list.html), cached locally so
     subsequent runs work offline.
  4. Heuristic product decoding for vendors with a known product-code
     scheme (currently Beckhoff) and the CiA device-profile number from
     object 0x1000.
"""

import html
import json
import logging
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Offline vendor table — subset of the ETG (EtherCAT Technology Group)
# vendor registry.  Used as the baseline; the cached online registry
# (see fetch_etg_vendors) overrides/extends it when available.
# ---------------------------------------------------------------------------
VENDOR_NAMES: dict[int, str] = {
    0x00000001: "EtherCAT Technology Group",
    0x00000002: "Beckhoff Automation",
    0x00000004: "KEB Automation",
    0x0000000E: "Bosch Rexroth",
    0x00000022: "Lenze",
    0x00000044: "Wago",
    0x00000048: "B&R Industrial Automation",
    0x0000004C: "ifm electronic",
    0x0000006A: "Festo",
    0x00000083: "Omron",
    0x000000AB: "Trio Motion Technology",
    0x000002DE: "Trio Motion Technology",
    0x000000B9: "SEW-Eurodrive",
    0x000000C7: "Pilz",
    0x000000E4: "Hilscher",
    0x000000FB: "SMC Corporation",
    0x00000127: "Mitsubishi Electric",
    0x0000014E: "Baumer",
    0x00000195: "Sick",
    0x000001DD: "Delta Electronics",
    0x00000226: "Oriental Motor",
    0x0000029C: "Keyence",
    0x000002BE: "Sanyo Denki",
    0x00000539: "Yaskawa Electric",
    0x0000054D: "Panasonic",
    0x00000569: "Maxon Motor",
    0x000005A2: "Nanotec Electronic",
    0x00000659: "Schneider Electric",
    0x0000066F: "Inovance Technology",
    0x00000A13: "Elmo Motion Control",
    0x00100000: "Copley Controls",
}

# CiA device profile numbers (low 16 bits of CoE object 0x1000 Device Type)
_PROFILE_NAMES: dict[int, str] = {
    401: "I/O module (CiA 401)",
    402: "Servo drive (CiA 402)",
    404: "Measurement device (CiA 404)",
    406: "Encoder (CiA 406)",
    5001: "Modular device (MDP)",
}

# Beckhoff encodes the model number in the high 16 bits of the product
# code and the product family in the low 16 bits, e.g.
#   EK1100 → 0x044C2C52  (0x044C = 1100, 0x2C52 = EK coupler family)
#   EL2002 → 0x07D23052  (0x07D2 = 2002, 0x3052 = EL terminal family)
_BECKHOFF_FAMILIES: dict[int, str] = {
    0x2C52: "EK",   # bus couplers
    0x3052: "EL",   # terminals
    0x4052: "EP",   # IP67 box modules
    0x6012: "AX",   # AX5xxx servo drives
}

# Trio DRIVE_TYPE axis-parameter values → product label
DRIVE_TYPE_LABELS: dict[int, str] = {
    41: "DX3",
    42: "DX4",
    43: "DX1",
    45: "DX5",
}

# ---------------------------------------------------------------------------
# Online ETG vendor registry
# ---------------------------------------------------------------------------
ETG_VENDOR_LIST_URL = "https://www.ethercat.org/en/vendor_id_list.html"

_CACHE_FILE = Path.home() / ".trioscope" / "etg_vendors.json"

# Cached online registry, loaded lazily.  Maps vendor ID → company name.
_online_vendors: Optional[dict[int, str]] = None


def _load_vendor_cache() -> dict[int, str]:
    """Load the locally cached ETG registry (empty dict if absent/corrupt)."""
    global _online_vendors
    if _online_vendors is not None:
        return _online_vendors
    _online_vendors = {}
    try:
        if _CACHE_FILE.is_file():
            raw = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
            _online_vendors = {int(k): str(v) for k, v in raw.items()}
            logger.debug("Loaded %d cached ETG vendors", len(_online_vendors))
    except Exception as exc:
        logger.warning("Failed to load ETG vendor cache: %s", exc)
    return _online_vendors


def _save_vendor_cache(vendors: dict[int, str]) -> None:
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(
            json.dumps({str(k): v for k, v in vendors.items()}, indent=0),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Failed to save ETG vendor cache: %s", exc)


def parse_etg_vendor_html(page: str) -> dict[int, str]:
    """Extract vendor ID → company name pairs from the ETG registry page.

    Parsing is deliberately tolerant: it scans every table row for a cell
    that looks like a vendor ID (hex ``0x…`` or decimal) followed by a
    non-numeric cell with the company name.
    """
    vendors: dict[int, str] = {}
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", page, re.S | re.I):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S | re.I)
        texts = []
        for c in cells:
            txt = html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
            texts.append(" ".join(txt.split()))
        vid: Optional[int] = None
        for i, txt in enumerate(texts):
            m = re.fullmatch(r"(?:0x|#x)?([0-9A-Fa-f]{1,8})", txt)
            if m and vid is None:
                base = 16 if txt.lower().startswith(("0x", "#x")) else 10
                try:
                    vid = int(m.group(1), base)
                except ValueError:
                    continue
                # company name = next non-empty, non-numeric cell
                for name in texts[i + 1:]:
                    if name and not re.fullmatch(r"[0-9A-Fa-fx#]+", name):
                        vendors[vid] = name
                        break
                break
    return vendors


def fetch_etg_vendors(timeout: float = 15.0) -> dict[int, str]:
    """Download the official ETG vendor-ID registry and cache it locally.

    Returns the parsed registry.  Raises on network/parse failure so the
    caller can report the error (the cached/offline tables stay intact).
    """
    req = urllib.request.Request(
        ETG_VENDOR_LIST_URL,
        headers={"User-Agent": "TrioScope EtherCAT Map"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        page = resp.read().decode("utf-8", errors="replace")

    vendors = parse_etg_vendor_html(page)
    if not vendors:
        raise ValueError(
            "No vendor entries found in the ETG page — site layout may have changed"
        )

    global _online_vendors
    merged = dict(_load_vendor_cache())
    merged.update(vendors)
    _online_vendors = merged
    _save_vendor_cache(merged)
    logger.info("Fetched %d vendors from ETG registry", len(vendors))
    return vendors


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def vendor_name(vendor_id: int) -> str:
    """Human-readable vendor name from the cached ETG registry or the
    built-in table, or the hex ID if unknown."""
    if not vendor_id:
        return ""
    cached = _load_vendor_cache().get(vendor_id)
    if cached:
        return cached
    return VENDOR_NAMES.get(vendor_id, f"Unknown (0x{vendor_id:08X})")


def device_profile_name(device_type: int) -> str:
    """Name of the CiA device profile encoded in CoE object 0x1000."""
    if not device_type:
        return ""
    profile = device_type & 0xFFFF
    return _PROFILE_NAMES.get(profile, f"Profile {profile}")


def product_label(vendor_id: int, product_code: int,
                  device_type: int = 0, drive_type: int = 0) -> str:
    """Best-effort short product name for a slave.

    Order: Trio DRIVE_TYPE → Beckhoff product-code decode →
    CiA profile guess → hex product code.
    """
    if drive_type in DRIVE_TYPE_LABELS:
        return DRIVE_TYPE_LABELS[drive_type]
    if drive_type:
        return f"T{drive_type}"

    if vendor_id == 0x00000002 and product_code:  # Beckhoff
        family = _BECKHOFF_FAMILIES.get(product_code & 0xFFFF)
        if family:
            return f"{family}{product_code >> 16}"

    profile = device_type & 0xFFFF
    if profile == 402:
        return "Drive"
    if profile in (401, 5001):
        return "I/O"
    if profile == 406:
        return "Encoder"

    if product_code:
        return f"0x{product_code:08X}"
    return ""


def revision_str(revision: int) -> str:
    """Format an EtherCAT revision number (major.minor in hi/lo words)."""
    if not revision:
        return ""
    return f"{(revision >> 16) & 0xFFFF}.{revision & 0xFFFF}"


def web_search_url(vendor_id: int, product_code: int, drive_type: int = 0) -> str:
    """Build a web-search URL to look up a device on the internet."""
    terms = ["EtherCAT"]
    vn = vendor_name(vendor_id)
    if vn and not vn.startswith("Unknown"):
        terms.append(vn)
    elif vendor_id:
        terms.append(f"vendor id 0x{vendor_id:08X}")
    label = product_label(vendor_id, product_code, drive_type=drive_type)
    if label and not label.startswith("0x"):
        terms.append(label)
    elif product_code:
        terms.append(f"product code 0x{product_code:08X}")
    query = urllib.parse.quote_plus(" ".join(terms))
    return f"https://www.google.com/search?q={query}"
