"""HTTP surface. Thin by design — the domain contracts are the API contract."""

from travel_intel.api.app import app, create_app

__all__ = ["app", "create_app"]
