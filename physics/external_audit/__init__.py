"""External validation audit infrastructure for AeroTwin.

Implements the pipeline recommended in
``AeroTwin_External_Dataset_Audit_Package.md``:

* ``audit_utils`` – dataset-agnostic phase, energy, sparsity, intervals
* ``dashlink_loader`` – NASA DASHlink Project 85 MATLAB flight data
* ``opensky_loader`` – OpenSky Trino historical state vectors
* ``build_featured_audit`` – common featured-dataset builder
* ``run_audit_pilot`` – minimal pilot experiment suite

OpenSky fuel labels are **physics-derived** (OpenAP), not independent ground
truth. DASHlink fuel targets are reconstructed from fuel-flow parameters when
native interval labels are unavailable — both cases are logged explicitly.

Import submodules explicitly, e.g.::

    from physics.external_audit.audit_utils import synthesize_demo_trajectory
    from physics.external_audit.build_featured_audit import build_demo_featured
"""

from __future__ import annotations

__all__ = [
    "audit_utils",
    "dashlink_loader",
    "opensky_loader",
    "build_featured_audit",
    "run_audit_pilot",
]
