"""Legacy pre-production Dashboard modules.

These modules are retained for compatibility tests and historical diagnostics.
They intentionally live outside a directory named ``pages`` so Streamlit does
not expose direct storage, ingestion, or retrieval code as production routes.
The supported UI is :mod:`src.observability.dashboard.app`, which only calls
FastAPI.
"""
