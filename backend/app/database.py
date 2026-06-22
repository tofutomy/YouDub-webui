"""Database access facade.

Historically this module held all SQLite CRUD logic. The implementation
has been split into the ``backend.app.db`` subpackage (connection /
migrations / tasks / settings / translate_providers). This module
re-exports the public surface so existing ``from . import database`` +
``database.xxx`` call sites keep working without changes.
"""

from __future__ import annotations

from .db import *  # noqa: F401,F403
from .db import __all__ as _db_all  # noqa: F401
