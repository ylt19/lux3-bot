import unittest
from pathfinding import (
    Graph,
    Grid,
    ResumableBFS,
    ReservationTable,
    ResumableDijkstra,
    ResumableSpaceTimeDijkstra,
)


class TestRS(unittest.TestCase):
    """
    pytest pathfinding/tests/test_resumable_search.py::TestRS
    """

    def test_with_graph(self):
        graph = Graph(4, edges=[(0, 1), (1, 2), (2, 1)])
        inf = float("inf")

        for a in (ResumableBFS, ResumableDijkstra):
            with self.subTest(a.__name__):
                rs = a(graph, 0)
                for n, ans in [(0, 0), (1, 1), (2, 2), (3, inf)]:
                    self.assertEqual(rs.distance(n), ans)

                rs.start_node = 2
                for n, ans in [(0, inf), (1, 1), (2, 0), (3, inf)]:
                    self.assertEqual(rs.distance(n), ans)

    def test_with_grid(self):
        grid = Grid([[1, 1, 1], [-1, -1, 1], [1, 1, 1]])
        inf = float("inf")

        for a in (ResumableBFS, ResumableDijkstra):
            with self.subTest(a.__name__):
                rs = a(grid, (0, 0))
                for n, ans in [((0, 0), 0), ((0, 1), inf), ((0, 2), 6)]:
                    self.assertEqual(rs.distance(n), ans)

                rs.start_node = (2, 1)
                for n, ans in [((0, 0), 3), ((0, 1), inf), ((0, 2), 3)]:
                    self.assertEqual(rs.distance(n), ans)


class TestResumableSpaceTimeDijkstra(unittest.TestCase):
    """
    pytest pathfinding/tests/test_resumable_search.py::TestResumableSpaceTimeDijkstra
    """

    def test_with_grid(self):
        grid = Grid([[1, 1, 1], [-1, -1, 1], [1, 1, 1]])
        rt = ReservationTable(grid)
        rs = ResumableSpaceTimeDijkstra(grid, (0, 0), rt)

        inf = float("inf")

        for n, ans in [((0, 0), 0), ((0, 1), inf), ((0, 2), 6)]:
            self.assertEqual(rs.distance(n), ans)

        rs.start_node = (2, 1)
        for n, ans in [((0, 0), 3), ((0, 1), inf), ((0, 2), 3)]:
            self.assertEqual(rs.distance(n), ans)

    def test_with_grid_with_reservation_table(self):
        grid = Grid([[1, 1, 1], [1, -1, 1], [1, 1, 1]], edge_collision=True)
        rt = ReservationTable(grid)
        rt.add_path([(0, 2), (0, 2), (0, 1), (0, 0)], reserve_destination=True)

        rs = ResumableSpaceTimeDijkstra(grid, (0, 0), rt)

        path = rs.find_path((1, 2))
        self.assertEqual(path, [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2)])

        path = rs.find_path((0, 2))
        self.assertEqual(path, [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)])

        path = rs.find_path((0, 1))
        self.assertEqual(path, [(0, 0), (0, 1)])
