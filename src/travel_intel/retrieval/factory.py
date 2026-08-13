"""Provider selection. The only place in the codebase that knows which mode is active."""

from __future__ import annotations

from travel_intel.config import DataMode, Settings, get_settings
from travel_intel.retrieval.base import AccommodationProvider, ActivityProvider
from travel_intel.retrieval.live import LiveProvider
from travel_intel.retrieval.snapshot import SnapshotProvider


def build_providers(
    settings: Settings | None = None,
) -> tuple[AccommodationProvider, ActivityProvider]:
    resolved = settings or get_settings()
    if resolved.data_mode is DataMode.LIVE:
        live = LiveProvider()
        return live, live
    snapshot = SnapshotProvider(resolved.fixtures_dir)
    return snapshot, snapshot
