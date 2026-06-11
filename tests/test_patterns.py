"""Tests for stent pattern constructors and FlowDiverter mesh building."""

import numpy as np

import stenting as s


class TestPatternConstructors:
    def test_helical_returns_pattern(self):
        p = s.helical(size=2)
        assert p.pattern.ndim == 3  # (n_lines, 2, 2)
        assert p.size_lon >= 1
        assert p.size_tgn >= 1

    def test_helical_size_scales_pattern(self):
        p2 = s.helical(size=2)
        p3 = s.helical(size=3)
        assert len(p3.pattern) > len(p2.pattern)

    def test_enterprise_n1(self):
        p = s.enterprise(N=1)
        assert p.pattern.shape[1:] == (2, 2)
        assert p.bot_cap.any()
        assert p.top_cap.any()

    def test_enterprise_n2(self):
        p = s.enterprise(N=2)
        assert len(p.pattern) > len(s.enterprise(N=1).pattern)

    def test_honeycomb_has_bot_cap(self):
        p = s.honeycomb()
        assert p.bot_cap.any()

    def test_semienterprise_has_bot_cap(self):
        p = s.semienterprise()
        assert p.bot_cap.any()


class TestFlowDiverterMesh:
    def test_mesh_has_points(self, simple_helical_stent):
        assert len(simple_helical_stent.mesh.points) > 0

    def test_mesh_has_lines(self, simple_helical_stent):
        assert len(simple_helical_stent.mesh.lines) > 0

    def test_node_count_matches_layers(self, simple_helical_stent):
        fd = simple_helical_stent
        expected_nodes = fd.layers * fd.nodes_per_layer
        # After clean() some isolated points may be removed
        assert len(fd.mesh.points) <= expected_nodes

    def test_connected_list_length_matches_points(self, simple_helical_stent):
        fd = simple_helical_stent
        assert len(fd.connected) == len(fd.mesh.points)

    def test_every_node_has_at_least_one_connection(self, simple_helical_stent):
        for neighbors in simple_helical_stent.connected:
            assert len(neighbors) >= 1

    def test_offset_angle_shifts_pattern(self):
        """Two stents with different offset_angle must have different point coords."""
        pattern = s.helical(size=2)
        fd0 = s.FlowDiverter(pattern, radius=1.5, height=10, tcopy=6, hcopy=3, offset_angle=0.0)
        fd1 = s.FlowDiverter(pattern, radius=1.5, height=10, tcopy=6, hcopy=3, offset_angle=0.25)
        assert not np.allclose(
            np.sort(fd0.mesh.points, axis=0),
            np.sort(fd1.mesh.points, axis=0),
        )

    def test_enterprise_stent_builds(self, enterprise_stent):
        assert len(enterprise_stent.mesh.points) > 0
        assert len(enterprise_stent.connected) == len(enterprise_stent.mesh.points)
