"""Geometry helpers.

Distances are computed on a local equirectangular projection centred on the
point of interest. Over the few hundred kilometres that matter for a fishing
trip the error is well under a percent, and it keeps the dependency surface to
plain shapely rather than pyproj/geopandas.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from shapely.geometry import LineString, Point, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points

EARTH_R_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (
        math.sin((p2 - p1) / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_R_KM * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def compass(bearing: float) -> str:
    points = (
        "north", "north-northeast", "northeast", "east-northeast",
        "east", "east-southeast", "southeast", "south-southeast",
        "south", "south-southwest", "southwest", "west-southwest",
        "west", "west-northwest", "northwest", "north-northwest",
    )
    return points[int((bearing % 360) / 22.5 + 0.5) % 16]


def rad_to_compass(radians: float | None) -> str:
    if radians is None:
        return "unknown"
    return compass(math.degrees(radians) % 360.0)


def destination_point(
    lat: float, lon: float, bearing: float, distance_km: float
) -> tuple[float, float]:
    br = math.radians(bearing)
    p1 = math.radians(lat)
    l1 = math.radians(lon)
    dr = distance_km / EARTH_R_KM
    p2 = math.asin(math.sin(p1) * math.cos(dr) + math.cos(p1) * math.sin(dr) * math.cos(br))
    l2 = l1 + math.atan2(
        math.sin(br) * math.sin(dr) * math.cos(p1),
        math.cos(dr) - math.sin(p1) * math.sin(p2),
    )
    return math.degrees(p2), (math.degrees(l2) + 540) % 360 - 180


def geodesic_distance_km(point_lat: float, point_lon: float, geom: BaseGeometry) -> float:
    """Shortest distance from a lat/lon to any geometry, in km."""
    nearest = nearest_point_on(point_lat, point_lon, geom)
    return haversine_km(point_lat, point_lon, nearest[0], nearest[1])


def nearest_point_on(
    point_lat: float, point_lon: float, geom: BaseGeometry
) -> tuple[float, float]:
    p = Point(point_lon, point_lat)
    near = nearest_points(p, geom)[1]
    return near.y, near.x


def load_features(collection: dict[str, Any]) -> list[tuple[dict[str, Any], BaseGeometry]]:
    out: list[tuple[dict[str, Any], BaseGeometry]] = []
    for feature in collection.get("features", []) or []:
        geom_raw = feature.get("geometry")
        if not geom_raw:
            continue
        try:
            geom = shape(geom_raw)
        except Exception:
            continue
        if geom.is_empty:
            continue
        out.append((feature.get("properties", {}) or {}, geom))
    return out


def great_circle_waypoints(
    lat1: float, lon1: float, lat2: float, lon2: float, segments: int = 8
) -> list[tuple[float, float]]:
    """Evenly spaced waypoints along the direct track, endpoints included."""
    total = haversine_km(lat1, lon1, lat2, lon2)
    if total <= 0 or segments < 1:
        return [(lat1, lon1), (lat2, lon2)]
    br = bearing_deg(lat1, lon1, lat2, lon2)
    points = [(lat1, lon1)]
    for i in range(1, segments):
        points.append(destination_point(lat1, lon1, br, total * i / segments))
    points.append((lat2, lon2))
    return points


def offset_track(
    lat1: float, lon1: float, lat2: float, lon2: float, offset_km: float, segments: int = 8
) -> list[tuple[float, float]]:
    """A track bowed sideways by `offset_km` at its midpoint.

    Used to generate candidate detours around bad sea state without pulling in a
    full routing engine. Positive offset bows to starboard of the direct track.
    """
    direct = great_circle_waypoints(lat1, lon1, lat2, lon2, segments)
    br = bearing_deg(lat1, lon1, lat2, lon2)
    normal = (br + 90.0) % 360.0
    out: list[tuple[float, float]] = []
    for i, (la, lo) in enumerate(direct):
        # sine taper so the ends stay pinned to origin and destination
        weight = math.sin(math.pi * i / max(1, len(direct) - 1))
        if abs(weight) < 1e-6:
            out.append((la, lo))
            continue
        out.append(destination_point(la, lo, normal, offset_km * weight))
    return out


def as_linestring(points: Iterable[tuple[float, float]]) -> LineString:
    return LineString([(lon, lat) for lat, lon in points])


def track_length_km(points: list[tuple[float, float]]) -> float:
    return sum(
        haversine_km(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])
        for i in range(len(points) - 1)
    )
