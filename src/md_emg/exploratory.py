"""
md_emg.exploratory — REMOVED FROM CORE PACKAGE (Prompt 3 refactoring)
=======================================================================
The exploratory analysis functions (PCA, permutation importance) have been
moved to the ``md_emg_extras`` package, which is distributed separately from
the core reproducible pipeline.

To use these tools, import from the new location::

    from md_emg_extras.exploratory import (
        run_exploratory,
        run_permutation_importance,
        run_pca,
        emg_feature_family,
        feature_family,
        _grouped_permutation_importance,
    )

Why they were moved
-------------------
The primary ML analyses (nested CV, OOF metrics, bootstrap, permutation tests,
FDR) do NOT depend on PCA or permutation feature importance.  Keeping them in
the core package created unnecessary coupling between the reproducible pipeline
and post-hoc descriptive analyses that require fitted ``.joblib`` models.

The CLI commands ``explore``, ``pca`` and ``permutation-importance`` have been
removed from the ``md-emg`` entrypoint for the same reason.  They can be called
directly as Python functions from ``md_emg_extras.exploratory``.

This stub is intentionally minimal so that import errors are clear and
informative.
"""

raise ImportError(
    "md_emg.exploratory has been moved to md_emg_extras.exploratory.\n"
    "Install the extras package (it is in src/md_emg_extras/) and use:\n"
    "    from md_emg_extras.exploratory import <name>"
)
