"""Shared fixtures for the test suite."""

import numpy as np
import pytest


@pytest.fixture
def simple_helical_stent():
    """A small helical FlowDiverter for fast geometry tests."""
    from stenting import FlowDiverter, helical

    pattern = helical(size=2)
    return FlowDiverter(pattern, radius=1.5, height=10, tcopy=6, hcopy=3)


@pytest.fixture
def enterprise_stent():
    """An enterprise-pattern FlowDiverter for cap/pattern tests."""
    from stenting import FlowDiverter, enterprise

    pattern = enterprise(N=1)
    return FlowDiverter(pattern, radius=1.5, height=10, tcopy=6, hcopy=3)
