"""Fail fast when users import the PyPI name instead of ``semanticcache``.

After ``pip install fastapi-semcache``, use ``import semanticcache``. Importing
``fastapi_semcache`` raises ImportError with an actionable message.
"""

raise ImportError(
    "The install name is fastapi-semcache but the import name is semanticcache. "
    "Use import semanticcache instead."
)
