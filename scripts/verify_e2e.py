"""End-to-end verification against the problem statement's sample queries.

Runs each query through the live API and asserts the things that actually matter:
the intent was segregated correctly, the right knowledge sources were consulted,
evidence came back, and at least one item carries an official Indian agency
provenance where the query needs one.

Usage:
    python scripts/verify_e2e.py [--base http://127.0.0.1:8017]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

# (query, expected intent, must consult these domains, needs an official source)
CASES: list[tuple[str, str, tuple[str, ...], bool]] = [
    (
        "Where is the nearest Potential Fishing Zone today near Chennai?",
        "pfz_locate",
        ("ocean", "advisory", "geospatial"),
        True,
    ),
    (
        "Is it safe to venture into the sea tomorrow morning near Rameswaram?",
        "safety_go_nogo",
        ("weather", "ocean", "hazard", "advisory"),
        True,
    ),
    (
        "What are the tide, weather and sea conditions near Kochi?",
        "conditions_summary",
        ("ocean", "weather", "advisory"),
        True,
    ),
    (
        "Are there any lightning or cyclone alerts near Visakhapatnam?",
        "hazard_alerts",
        ("hazard", "weather", "advisory"),
        False,
    ),
    (
        "Which regions show high chlorophyll concentration and favourable sea "
        "surface temperature off Malpe?",
        "productivity_scan",
        ("ocean", "geospatial", "advisory"),
        True,
    ),
    (
        "What is the safest route for a fishing vessel from Chennai to Kakinada "
        "considering weather and sea-state conditions?",
        "route_planning",
        ("weather", "ocean", "geospatial", "hazard"),
        True,
    ),
    (
        "Why has fish productivity declined near Nagapattinam?",
        "productivity_diagnosis",
        ("ocean", "advisory", "catalog"),
        True,
    ),
    (
        "Which fishing zones should be avoided due to hazardous marine conditions "
        "or geofencing restrictions near Rameswaram?",
        "geofence_check",
        ("geospatial", "hazard", "ocean"),
        False,
    ),
]


def post(base: str, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode())


def get(base: str, path: str) -> dict:
    with urllib.request.urlopen(f"{base}{path}", timeout=60) as response:
        return json.loads(response.read().decode())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8017")
    args = parser.parse_args()

    try:
        health = get(args.base, "/health")
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"FAIL: API not reachable at {args.base}: {exc}")
        return 2

    print(f"API up. knowledge={health['vector_rag']['chunks']} chunks, "
          f"zones={health['geo_rag']['zone_features']}, "
          f"llm={health['llm']}, eez={health['geo_rag']['eez_loaded']}")
    print("=" * 78)

    failures = 0
    for query, expected_intent, expected_domains, needs_official in CASES:
        print(f"\nQ: {query}")
        classified = post(
            args.base,
            "/internal/classify",
            {"message": query, "session_id": "verify"},
        )
        intent_ok = classified["intent"] == expected_intent
        domains = set(classified["domains"])
        domains_ok = set(expected_domains).issubset(domains)

        answer = post(
            args.base, "/chat", {"message": query, "session_id": "verify"}
        )
        evidence = answer.get("evidence", [])
        tiers = {e["provenance"]["tier"] for e in evidence}
        official = {"tier1-isro", "tier2-incois", "tier3-imd"} & tiers
        evidence_ok = len(evidence) > 0
        official_ok = bool(official) if needs_official else True
        answer_ok = len(answer.get("answer", "").split()) >= 12
        trace_ok = len(answer.get("trace", [])) >= 4

        checks = {
            f"intent == {expected_intent}": intent_ok,
            f"domains >= {sorted(expected_domains)}": domains_ok,
            "evidence returned": evidence_ok,
            "official agency source present": official_ok,
            "answer is substantive": answer_ok,
            "reasoning trace recorded": trace_ok,
        }
        for label, ok in checks.items():
            print(f"   {'PASS' if ok else 'FAIL'}  {label}")
            if not ok:
                failures += 1
        print(
            f"   -> intent={classified['intent']} ({classified['confidence']}, "
            f"{classified['decided_by']}) domains={sorted(domains)}"
        )
        print(
            f"   -> risk={(answer.get('risk') or {}).get('band')} "
            f"evidence={len(evidence)} tiers={sorted(tiers)} "
            f"trace_steps={len(answer.get('trace', []))} "
            f"layers={len(answer.get('layers', []))} "
            f"charts={len(answer.get('charts', []))}"
        )
        first_line = answer.get("answer", "").strip().splitlines()[0][:150]
        print(f'   -> "{first_line}"')

    print("\n" + "=" * 78)
    if failures:
        print(f"{failures} check(s) FAILED")
        return 1
    print(f"all checks passed across {len(CASES)} sample queries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
