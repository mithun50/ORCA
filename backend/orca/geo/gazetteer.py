"""Coastal gazetteer.

Location resolution for a fisherman's phrasing: "near Rameswaram", "off
Versova", "Paradip". Coordinates are placed a short distance *offshore* of each
landing centre or harbour, because a model grid point on the shoreline is
usually land-masked and because the question is always about the sea, not the
village.

Kept deliberately small and auditable rather than wired to a geocoding API, so
the prototype has no per-request third-party dependency for something this
critical. `source` on the resolved Location records how it was matched.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Place:
    name: str
    lat: float
    lon: float
    district: str
    state: str
    kind: str = "landing-centre"
    aliases: tuple[str, ...] = ()


# lat/lon are nudged seaward of the harbour mouth
PLACES: tuple[Place, ...] = (
    # --- Gujarat ---
    Place("Okha", 22.46, 69.03, "Devbhumi Dwarka", "Gujarat", "harbour"),
    Place("Porbandar", 21.62, 69.58, "Porbandar", "Gujarat", "harbour"),
    Place("Veraval", 20.88, 70.35, "Gir Somnath", "Gujarat", "harbour"),
    Place("Jakhau", 23.21, 68.58, "Kutch", "Gujarat", "harbour"),
    Place("Mangrol", 21.10, 70.10, "Junagadh", "Gujarat"),
    Place("Diu", 20.68, 70.98, "Diu", "Daman and Diu"),
    # --- Maharashtra ---
    Place("Mumbai", 18.90, 72.75, "Mumbai", "Maharashtra", "harbour",
          ("bombay", "sassoon dock")),
    Place("Versova", 19.14, 72.78, "Mumbai Suburban", "Maharashtra",
          aliases=("versoa",)),
    Place("Satpati", 19.71, 72.68, "Palghar", "Maharashtra"),
    Place("Ratnagiri", 16.98, 73.26, "Ratnagiri", "Maharashtra", "harbour"),
    Place("Malvan", 16.05, 73.44, "Sindhudurg", "Maharashtra"),
    # --- Goa ---
    Place("Panaji", 15.52, 73.78, "North Goa", "Goa", aliases=("panjim",)),
    Place("Vasco da Gama", 15.38, 73.76, "South Goa", "Goa",
          aliases=("mormugao", "vasco")),
    # --- Karnataka ---
    Place("Karwar", 14.80, 74.09, "Uttara Kannada", "Karnataka", "harbour"),
    Place("Malpe", 13.36, 74.67, "Udupi", "Karnataka", "harbour"),
    Place("Mangaluru", 12.86, 74.78, "Dakshina Kannada", "Karnataka", "harbour",
          ("mangalore",)),
    # --- Kerala ---
    Place("Kasaragod", 12.50, 74.95, "Kasaragod", "Kerala"),
    Place("Kozhikode", 11.25, 75.74, "Kozhikode", "Kerala",
          aliases=("calicut", "beypore")),
    Place("Munambam", 10.18, 76.14, "Ernakulam", "Kerala"),
    Place("Kochi", 9.93, 76.22, "Ernakulam", "Kerala", "harbour",
          ("cochin", "ernakulam")),
    Place("Alappuzha", 9.49, 76.30, "Alappuzha", "Kerala", aliases=("alleppey",)),
    Place("Kollam", 8.88, 76.55, "Kollam", "Kerala", "harbour", ("quilon", "neendakara")),
    Place("Vizhinjam", 8.38, 76.97, "Thiruvananthapuram", "Kerala", "harbour"),
    Place("Thiruvananthapuram", 8.44, 76.92, "Thiruvananthapuram", "Kerala",
          aliases=("trivandrum",)),
    # --- Tamil Nadu ---
    Place("Kanyakumari", 8.05, 77.55, "Kanyakumari", "Tamil Nadu",
          aliases=("cape comorin",)),
    Place("Colachel", 8.16, 77.23, "Kanyakumari", "Tamil Nadu", "harbour"),
    Place("Tuticorin", 8.74, 78.20, "Thoothukudi", "Tamil Nadu", "harbour",
          ("thoothukudi",)),
    Place("Rameswaram", 9.27, 79.35, "Ramanathapuram", "Tamil Nadu", "harbour",
          ("rameshwaram", "ramanathapuram")),
    Place("Mandapam", 9.26, 79.14, "Ramanathapuram", "Tamil Nadu"),
    Place("Nagapattinam", 10.76, 79.90, "Nagapattinam", "Tamil Nadu", "harbour",
          ("nagapattinam", "nagore")),
    Place("Karaikal", 10.92, 79.88, "Karaikal", "Puducherry"),
    Place("Cuddalore", 11.72, 79.82, "Cuddalore", "Tamil Nadu", "harbour"),
    Place("Puducherry", 11.93, 79.86, "Puducherry", "Puducherry",
          aliases=("pondicherry", "pondy")),
    Place("Mahabalipuram", 12.62, 80.22, "Chengalpattu", "Tamil Nadu",
          aliases=("mamallapuram",)),
    Place("Kasimedu", 13.14, 80.31, "Chennai", "Tamil Nadu", "harbour",
          ("royapuram",)),
    Place("Chennai", 13.08, 80.40, "Chennai", "Tamil Nadu", "harbour",
          ("madras", "chennai port")),
    Place("Pulicat", 13.42, 80.34, "Tiruvallur", "Tamil Nadu",
          aliases=("pazhaverkadu",)),
    # --- Andhra Pradesh ---
    Place("Nellore", 14.45, 80.19, "Nellore", "Andhra Pradesh",
          aliases=("krishnapatnam",)),
    Place("Ongole", 15.55, 80.16, "Prakasam", "Andhra Pradesh"),
    Place("Machilipatnam", 16.16, 81.20, "Krishna", "Andhra Pradesh",
          aliases=("bandar",)),
    Place("Kakinada", 16.94, 82.32, "East Godavari", "Andhra Pradesh", "harbour"),
    Place("Visakhapatnam", 17.68, 83.32, "Visakhapatnam", "Andhra Pradesh",
          "harbour", ("vizag", "vishakhapatnam")),
    # --- Odisha ---
    Place("Gopalpur", 19.26, 84.94, "Ganjam", "Odisha", "harbour"),
    Place("Puri", 19.78, 85.85, "Puri", "Odisha"),
    Place("Paradip", 20.25, 86.73, "Jagatsinghpur", "Odisha", "harbour",
          ("paradeep",)),
    Place("Dhamra", 20.78, 87.00, "Bhadrak", "Odisha"),
    # --- West Bengal ---
    Place("Digha", 21.61, 87.55, "Purba Medinipur", "West Bengal", "harbour"),
    Place("Frasergunj", 21.55, 88.25, "South 24 Parganas", "West Bengal",
          "harbour", ("fraserganj", "bakkhali")),
    Place("Haldia", 21.85, 88.12, "Purba Medinipur", "West Bengal", "harbour"),
    # --- Islands ---
    Place("Port Blair", 11.62, 92.78, "South Andaman", "Andaman and Nicobar",
          "harbour"),
    Place("Kavaratti", 10.56, 72.60, "Lakshadweep", "Lakshadweep", "harbour"),
    Place("Minicoy", 8.28, 73.02, "Lakshadweep", "Lakshadweep"),
    # --- open-water reference areas ---
    Place("Wadge Bank", 7.80, 77.20, "offshore", "Tamil Nadu", "fishing-ground"),
    Place("Gulf of Mannar", 9.00, 79.00, "Ramanathapuram", "Tamil Nadu",
          "fishing-ground"),
    Place("Palk Bay", 9.60, 79.55, "Ramanathapuram", "Tamil Nadu", "fishing-ground"),
    Place("Gulf of Kutch", 22.50, 69.60, "Kutch", "Gujarat", "fishing-ground"),
    Place("Bay of Bengal", 15.00, 84.00, "offshore", "-", "sea-area"),
    Place("Arabian Sea", 15.00, 70.00, "offshore", "-", "sea-area"),
)

DEFAULT_PLACE = next(p for p in PLACES if p.name == "Chennai")


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]", " ", text.lower())


_INDEX: list[tuple[str, Place]] = []
for _p in PLACES:
    _INDEX.append((_norm(_p.name), _p))
    for _alias in _p.aliases:
        _INDEX.append((_norm(_alias), _p))
# longest keys first so "chennai port" wins over "chennai"
_INDEX.sort(key=lambda pair: len(pair[0]), reverse=True)


def find_place(text: str) -> Place | None:
    """Longest-token-match over the gazetteer."""
    haystack = f" {_norm(text)} "
    for key, place in _INDEX:
        if f" {key} " in haystack:
            return place
    return None


def find_all_places(text: str) -> list[Place]:
    """All distinct gazetteer hits, in order of appearance (for routes)."""
    haystack = f" {_norm(text)} "
    hits: list[tuple[int, Place]] = []
    seen: set[str] = set()
    for key, place in _INDEX:
        idx = haystack.find(f" {key} ")
        if idx >= 0 and place.name not in seen:
            seen.add(place.name)
            hits.append((idx, place))
    hits.sort(key=lambda pair: pair[0])
    return [place for _, place in hits]


LAT_LON_RE = re.compile(
    r"(-?\d{1,2}(?:\.\d+)?)\s*(?:deg|°)?\s*([NnSs])?\s*[, ]\s*"
    r"(-?\d{1,3}(?:\.\d+)?)\s*(?:deg|°)?\s*([EeWw])?"
)


def parse_coords(text: str) -> tuple[float, float] | None:
    """Pull an explicit lat/lon out of free text, if present."""
    match = LAT_LON_RE.search(text)
    if not match:
        return None
    lat = float(match.group(1))
    lon = float(match.group(3))
    if (match.group(2) or "").lower() == "s":
        lat = -lat
    if (match.group(4) or "").lower() == "w":
        lon = -lon
    # sanity-check against the Indian Ocean region before trusting it
    if not (-10.0 <= lat <= 30.0 and 55.0 <= lon <= 100.0):
        return None
    return lat, lon
