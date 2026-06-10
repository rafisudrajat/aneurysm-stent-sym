"""Shared fixtures for the test suite."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the flat-script modules importable from the repo root.
# Phase 2 replaces this with a proper package install.
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def simple_helical_stent():
    """A small helical FlowDiverter for fast geometry tests."""
    import PyStenting as ps

    pattern = ps.helical(size=2)
    return ps.FlowDiverter(pattern, radius=1.5, height=10, tcopy=6, hcopy=3)


@pytest.fixture
def enterprise_stent():
    """An enterprise-pattern FlowDiverter for cap/pattern tests."""
    import PyStenting as ps

    pattern = ps.enterprise(N=1)
    return ps.FlowDiverter(pattern, radius=1.5, height=10, tcopy=6, hcopy=3)
