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
    Place("Kaup", 13.22, 74.72, "Udupi", "Karnataka",
          aliases=("kapu", "kappu", "kaup beach", "kapu beach", "kappu beach")),
    Place("Udupi", 13.34, 74.70, "Udupi", "Karnataka",
          aliases=("udipi", "malpe beach")),
    Place("Bhatkal", 13.97, 74.54, "Uttara Kannada", "Karnataka"),
    Place("Honnavar", 14.28, 74.42, "Uttara Kannada", "Karnataka"),
    Place("Kundapura", 13.63, 74.66, "Udupi", "Karnataka",
          aliases=("kundapur", "gangolli")),
    Place("Someshwara", 12.78, 74.85, "Dakshina Kannada", "Karnataka",
          aliases=("someshwar", "ullal")),
    Place("Mangaluru", 12.86, 74.78, "Dakshina Kannada", "Karnataka", "harbour",
          ("mangalore", "panambur", "tannirbhavi", "surathkal")),
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

#: Native-script names for the places a coastal user is most likely to type in
#: their own language. `_norm` strips every non-ASCII character, which is right
#: for "Mangalore" vs "Mangaluru" but silently erases "ರಾಮೇಶ್ವರಂ" to nothing. A
#: Kannada or Tamil question would therefore resolve to no location at all and
#: get answered with a clarification request, which is the single most visible
#: way the multilingual path can fail.
NATIVE_ALIASES: dict[str, tuple[str, ...]] = {
    "Rameswaram": (
        "ರಾಮೇಶ್ವರಂ", "ராமேஸ்வரம்", "రామేశ్వరం", "രാമേശ്വരം", "रामेश्वरम",
    ),
    "Chennai": ("ಚೆನ್ನೈ", "சென்னை", "చెన్నై", "ചെന്നൈ", "चेन्नई", "মাদ্রাজ"),
    "Kochi": ("ಕೊಚ್ಚಿ", "கொச்சி", "కొచ్చి", "കൊച്ചി", "कोच्चि", "ಕೊಚ್ಚಿನ್"),
    "Mangaluru": ("ಮಂಗಳೂರು", "மங்களூரு", "మంగళూరు", "മംഗളൂരു", "मंगळूरु"),
    "Malpe": ("ಮಲ್ಪೆ", "மல்பே", "മൽപെ"),
    "Karwar": ("ಕಾರವಾರ", "கார்வார்", "കാർവാർ"),
    "Udupi": ("ಉಡುಪಿ", "உடுப்பி", "ഉടുപ്പി"),
    "Visakhapatnam": (
        "ವಿಶಾಖಪಟ್ಟಣಂ", "விசாகப்பட்டினம்", "విశాఖపట్నం", "വിശാഖപട്ടണം", "विशाखापत्तनम",
    ),
    "Nagapattinam": ("ನಾಗಪಟ್ಟಿಣಂ", "நாகப்பட்டினம்", "నాగపట్నం", "നാഗപട്ടണം"),
    "Kanyakumari": ("ಕನ್ಯಾಕುಮಾರಿ", "கன்னியாகுமரி", "కన్యాకుమారి", "കന്യാകുമാരി"),
    "Tuticorin": ("ತೂತುಕುಡಿ", "தூத்துக்குடி", "തൂത്തുക്കുടി"),
    "Mumbai": ("ಮುಂಬೈ", "மும்பை", "ముంబై", "മുംബൈ", "मुंबई"),
    "Kozhikode": ("ಕೋಝಿಕ್ಕೋಡ್", "கோழிக்கோடு", "കോഴിക്കോട്", "कोझिकोड"),
    "Thiruvananthapuram": (
        "ತಿರುವನಂತಪುರಂ", "திருவனந்தபுரம்", "തിരുവനന്തപുരം", "तिरुवनंतपुरम",
    ),
    "Kakinada": ("ಕಾಕಿನಾಡ", "காக்கிநாடா", "కాకినాడ", "കാക്കിനാഡ"),
    "Paradip": ("ಪರದೀಪ", "பரதீப்", "పరదీప్", "पारादीप"),
    "Kolkata": ("ಕೋಲ್ಕತ್ತಾ", "கொல்கத்தா", "కోల్‌కతా", "കൊൽക്കത്ത", "कोलकाता", "কলকাতা"),
    "Veraval": ("ವೇರಾವಳ", "வேராவல்", "વેરાવળ", "वेरावळ"),
    "Porbandar": ("ಪೋರಬಂದರ್", "போர்பந்தர்", "પોરબંદર", "पोरबंदर"),
    "Ratnagiri": ("ರತ್ನಗಿರಿ", "ரத்னகிரி", "रत्नागिरी"),
    "Gulf of Mannar": ("ಮನ್ನಾರ್ ಕೊಲ್ಲಿ", "மன்னார் வளைகுடா", "മന്നാർ ഉൾക്കടൽ"),
    "Palk Bay": ("ಪಾಕ್ ಜಲಸಂಧಿ", "பாக் வளைகுடா", "പാക് ഉൾക്കടൽ"),
    "Bay of Bengal": ("ಬಂಗಾಳ ಕೊಲ್ಲಿ", "வங்காள விரிகுடா", "ബംഗാൾ ഉപസാഗരം", "बंगाल की खाड़ी"),
    "Arabian Sea": ("ಅರಬ್ಬಿ ಸಮುದ್ರ", "அரபிக் கடல்", "അറബിക്കടൽ", "अरब सागर"),
}


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

#: Raw-substring index for native scripts, which must not go through `_norm`.
_BY_NAME = {p.name: p for p in PLACES}
_NATIVE_INDEX: list[tuple[str, Place]] = []
for _name, _natives in NATIVE_ALIASES.items():
    _place = _BY_NAME.get(_name)
    if _place is None:  # pragma: no cover - guards a typo in the table above
        continue
    for _native in _natives:
        _NATIVE_INDEX.append((_native, _place))
_NATIVE_INDEX.sort(key=lambda pair: len(pair[0]), reverse=True)


def _find_native(text: str) -> Place | None:
    """Match a place written in an Indian script, by raw substring."""
    for key, place in _NATIVE_INDEX:
        if key in text:
            return place
    return None


def find_place(text: str) -> Place | None:
    """Longest-token-match over the gazetteer, native scripts included."""
    native = _find_native(text)
    if native is not None:
        return native
    haystack = f" {_norm(text)} "
    for key, place in _INDEX:
        if f" {key} " in haystack:
            return place
    return None


def find_all_places(text: str) -> list[Place]:
    """All distinct gazetteer hits, in order of appearance (for routes)."""
    hits: list[tuple[int, Place]] = []
    seen: set[str] = set()
    # native script first, positioned by where they appear in the raw text
    for key, place in _NATIVE_INDEX:
        idx = text.find(key)
        if idx >= 0 and place.name not in seen:
            seen.add(place.name)
            hits.append((idx, place))
    haystack = f" {_norm(text)} "
    for key, place in _INDEX:
        idx = haystack.find(f" {key} ")
        if idx >= 0 and place.name not in seen:
            seen.add(place.name)
            hits.append((idx, place))
    hits.sort(key=lambda pair: pair[0])
    return [place for _, place in hits]


#: Well-known inland places. A marine question about one of these is not a
#: missing location, it is a misconceived one: there is no sea at Bangalore. The
#: nearest coast is offered so the user can redirect in one step rather than
#: being told "I need a location" about a place they clearly named.
INLAND_PLACES: dict[str, tuple[str, ...]] = {
    "Bengaluru": ("bangalore", "banglore", "bengaluru", "bangluru", "bangaluru",
                  "benguluru", "bengluru", "blr"),
    "Delhi": ("delhi", "new delhi"),
    "Hyderabad": ("hyderabad", "secunderabad"),
    "Pune": ("pune", "poona"),
    "Coimbatore": ("coimbatore", "kovai"),
    "Madurai": ("madurai",),
    "Mysuru": ("mysuru", "mysore"),
    "Nagpur": ("nagpur",),
    "Jaipur": ("jaipur",),
    "Lucknow": ("lucknow",),
    "Bhopal": ("bhopal",),
    "Indore": ("indore",),
    "Ahmedabad": ("ahmedabad", "amdavad"),
    "Tiruchirappalli": ("tiruchirappalli", "trichy"),
    "Salem": ("salem",),
    "Hubballi": ("hubballi", "hubli", "dharwad"),
    "Warangal": ("warangal",),
    "Vijayawada": ("vijayawada", "bezawada"),
    "Kanpur": ("kanpur",),
    "Patna": ("patna",),
}

#: Nearest coastal gazetteer entry for each inland place, so the reply can offer
#: a usable alternative instead of a dead end.
NEAREST_COAST: dict[str, str] = {
    "Bengaluru": "Mangaluru",
    "Delhi": "Mumbai",
    "Hyderabad": "Visakhapatnam",
    "Pune": "Mumbai",
    "Coimbatore": "Kochi",
    "Madurai": "Rameswaram",
    "Mysuru": "Mangaluru",
    "Nagpur": "Mumbai",
    "Jaipur": "Mumbai",
    "Lucknow": "Kolkata",
    "Bhopal": "Mumbai",
    "Indore": "Mumbai",
    "Ahmedabad": "Veraval",
    "Tiruchirappalli": "Nagapattinam",
    "Salem": "Chennai",
    "Hubballi": "Karwar",
    "Warangal": "Visakhapatnam",
    "Vijayawada": "Kakinada",
    "Kanpur": "Kolkata",
    "Patna": "Kolkata",
}

LAT_LON_RE = re.compile(
    r"(-?\d{1,2}(?:\.\d+)?)\s*(?:deg|°)?\s*([NnSs])?\s*[, ]\s*"
    r"(-?\d{1,3}(?:\.\d+)?)\s*(?:deg|°)?\s*([EeWw])?"
)

#: Words that introduce a place in a fisherman's phrasing, and the words that
#: end one. Used to tell "the user named a place I do not know" apart from "the
#: user named no place at all". Those two cases need opposite handling: the
#: first must never silently inherit an earlier location.
_PLACE_LEAD = re.compile(
    r"\b(?:near|off|at|around|close to|outside|beside|by|in|into|from|to)\s+"
    r"(?P<name>[^,.?!;]{2,40})",
    re.IGNORECASE,
)
_STOP_WORDS = {
    "the", "sea", "coast", "shore", "water", "waters", "beach", "harbour",
    "harbor", "port", "jetty", "me", "us", "my", "our", "here", "there", "home",
    "today", "tomorrow", "tonight", "now", "morning", "evening", "night",
    "place", "position", "location", "area", "region", "side", "village",
    "town", "city", "district", "landing", "centre", "center", "point",
}


def find_inland(text: str) -> tuple[str, str] | None:
    """(inland place, nearest coastal place) if the text names an inland city.

    Checked before the "unknown place" path, because "there is no sea at
    Bengaluru" is a far more useful answer than "I do not know Bengaluru".
    """
    haystack = f" {_norm(text)} "
    best: tuple[int, str] | None = None
    for canonical, aliases in INLAND_PLACES.items():
        for alias in aliases:
            if f" {_norm(alias)} " in haystack:
                if best is None or len(alias) > best[0]:
                    best = (len(alias), canonical)
    if best is None:
        return None
    name = best[1]
    return name, NEAREST_COAST.get(name, "Chennai")


def unresolved_place(message: str) -> str | None:
    """A place the user named that the gazetteer does not know, if any.

    Returns None when either every named place resolved, or no place was named
    at all. Only a genuine miss comes back, because that is the one case where
    answering about somewhere else would be dangerous.
    """
    for match in _PLACE_LEAD.finditer(message):
        phrase = match.group("name").strip()
        if not phrase:
            continue
        # drop trailing time words: "near kappu beach tomorrow morning"
        words = [w for w in re.split(r"\s+", phrase) if w]
        kept: list[str] = []
        for word in words:
            if _norm(word).strip() in _STOP_WORDS:
                continue
            kept.append(word)
        candidate = " ".join(kept).strip()
        if not candidate or len(_norm(candidate).strip()) < 3:
            continue
        if find_place(candidate) is None and find_place(phrase) is None:
            return candidate
    return None


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
