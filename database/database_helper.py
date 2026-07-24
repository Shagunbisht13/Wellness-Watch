"""
Module 11 — Database Module (compatibility shim)
================================================
Canonical implementation lives in ``modules.database_module``.
This package path is kept so dashboard/api imports continue to work.
"""

from modules.database_module import (  # noqa: F401
    fetch_recent,
    fetch_stats,
    initialize_db,
    insert_record,
)
