"""Geographic helpers.

Distances are derived from the provider's own coordinates and a documented reference
point, so `distance_to_center_km` is a computed fact, not an estimate we invented.
"""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

from travel_intel.domain.models import GeoPoint

EARTH_RADIUS_KM = 6371.0088

CITY_CENTERS: dict[str, GeoPoint] = {
    # Tokyo Station: the conventional km-zero for the city and the main long-distance hub.
    "tokyo": GeoPoint(lat=35.681236, lon=139.767125),
}


def haversine_km(a: GeoPoint, b: GeoPoint) -> float:
    """Great-circle distance in kilometres.

    Straight-line distance is a deliberate simplification: walking or transit time would be
    the better feature, but it needs a routing API we do not have. The README records this.
    """
    lat1, lon1, lat2, lon2 = map(radians, (a.lat, a.lon, b.lat, b.lon))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return round(2 * EARTH_RADIUS_KM * asin(sqrt(h)), 3)


def city_center(destination: str) -> GeoPoint | None:
    """Reference point for a destination key, or None when we have no documented one."""
    return CITY_CENTERS.get(destination)
