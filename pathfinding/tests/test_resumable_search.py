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

    def test_with_time(self):
        grid = Grid(
            [[1, 1, 0.9], [1, -1, 1], [0.2, 0.1, 0.2]], pause_action_cost="node.weight"
        )
        rt = ReservationTable(grid)

        rs = ResumableSpaceTimeDijkstra(grid, (0, 0), rt, search_depth=10)

        inf = float("inf")
        for time, distance_answer, path_answer in [
            (0, inf, []),
            (1, inf, []),
            (2, inf, []),
            (3, 2.9, [(0, 0), (1, 0), (2, 0), (2, 1)]),
            (4, 3.8, [(0, 0), (1, 0), (2, 0), (2, 0), (2, 1)]),
            (5, 2.5, [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1)]),
            (6, 2.6, [(0, 0), (0, 1), (0, 2), (1, 2), (1, 2), (2, 2), (2, 1)]),
            (None, 2.5, [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1)]),
        ]:
            path = rs.find_path((2, 1), time)
            self.assertEqual(path, path_answer)

            distance = rs.distance((2, 1), time)
            self.assertAlmostEqual(distance, distance_answer, delta=0.001)

    def test_with_reserved_goal(self):
        grid = Grid([[1, 0.5, 1, 1]], pause_action_cost="node.weight")
        rt = ReservationTable(grid)
        rt.add_path([(2, 0), (2, 0), (2, 0), (2, 0), (3, 0)], reserve_destination=True)

        rs = ResumableSpaceTimeDijkstra(grid, (0, 0), rt, search_depth=10)

        inf = float("inf")
        for time, distance_answer, path_answer in [
            (None, 2.5, [(0, 0), (1, 0), (1, 0), (1, 0), (2, 0)]),
            (1, inf, []),
            (2, inf, []),
            (3, inf, []),
            (4, 2.5, [(0, 0), (1, 0), (1, 0), (1, 0), (2, 0)]),
            (5, 3.0, [(0, 0), (1, 0), (1, 0), (1, 0), (1, 0), (2, 0)]),
        ]:
            path = rs.find_path((2, 0), time)
            self.assertEqual(path, path_answer)

            distance = rs.distance((2, 0), time)
            self.assertAlmostEqual(distance, distance_answer, delta=0.001)
