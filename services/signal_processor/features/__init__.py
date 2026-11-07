"""Window aggregation exports."""

from services.signal_processor.features.windows import (
    CONTEXT_WINDOW_SECONDS,
    EVAL_CADENCE_SECONDS,
    SCRAPE_INTERVAL_SECONDS,
    SHORT_WINDOW_SECONDS,
    RawScrape,
    aggregate_window,
    build_all_service_windows,
    build_evaluation_windows,
    evaluation_times,
)

__all__ = [
    "CONTEXT_WINDOW_SECONDS",
    "EVAL_CADENCE_SECONDS",
    "SCRAPE_INTERVAL_SECONDS",
    "SHORT_WINDOW_SECONDS",
    "RawScrape",
    "aggregate_window",
    "build_all_service_windows",
    "build_evaluation_windows",
    "evaluation_times",
]
