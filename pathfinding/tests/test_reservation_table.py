import unittest
from copy import copy
from pathfinding import Grid, ReservationTable


class TestReservationTable(unittest.TestCase):
    """
    pytest pathfinding/tests/test_reservation_table.py::TestReservationTable
    """

    def test_with_grid(self):
        grid = Grid(width=3, height=3)
        rt = ReservationTable(grid)

        self.assertFalse(rt.is_reserved(2, (0, 1)))
        rt.add_vertex_constraint(2, (0, 1))

        self.assertTrue(rt.is_reserved(2, (0, 1)))

        self.assertFalse(rt.is_edge_reserved(2, (0, 1), (1, 1)))
        rt.add_edge_constraint(2, (0, 1), (1, 1))
        self.assertTrue(rt.is_edge_reserved(2, (0, 1), (1, 1)))

    def test_reserve_permanently(self):
        grid = Grid(width=3, height=3)
        rt = ReservationTable(grid)

        rt.add_vertex_constraint(time=2, node=(0, 1), permanent=True)

        for time, reserved in (
            (0, False),
            (1, False),
            (2, True),
            (3, True),
            (100, True),
        ):
            self.assertEqual(rt.is_reserved(time, (0, 1)), reserved)

    def test_copy(self):
        grid = Grid(width=3, height=3)
        rt = ReservationTable(grid)

        rt.add_vertex_constraint(time=2, node=(0, 1))

        rt_copy = copy(rt)
        rt_copy.add_vertex_constraint(3, node=(0, 1))

        self.assertEqual(id(rt.graph), id(rt_copy.graph))

        self.assertTrue(rt.is_reserved(2, (0, 1)))
        self.assertFalse(rt.is_reserved(3, (0, 1)))

        self.assertTrue(rt_copy.is_reserved(2, (0, 1)))
        self.assertTrue(rt_copy.is_reserved(3, (0, 1)))

    def test_add_additional_weight(self):
        grid = Grid(width=3, height=3)
        rt = ReservationTable(grid)

        rt.add_additional_weight(time=2, node=(0, 1), weight=11.5)

        for time, weight in (
            (0, 0),
            (1, 0),
            (2, 11.5),
            (3, 0),
            (100, 0),
        ):
            self.assertEqual(rt.get_additional_weight(time, (0, 1)), weight)

    def test_add_weight_path(self):
        grid = Grid(width=3, height=3)
        rt = ReservationTable(grid)

        rt.add_weight_path([(0, 0), (1, 0), (1, 1)], weight=2, start_time=1)

        self.assertEqual(rt.get_additional_weight(0, (0, 0)), 0)
        self.assertEqual(rt.get_additional_weight(1, (0, 0)), 2)
        self.assertEqual(rt.get_additional_weight(2, (1, 0)), 2)
        self.assertEqual(rt.get_additional_weight(3, (1, 1)), 2)
        self.assertEqual(rt.get_additional_weight(4, (1, 1)), 0)

    def test_copy_with_additional_weights(self):
        grid = Grid(width=3, height=3)
        rt = ReservationTable(grid)

        rt.add_additional_weight(time=2, node=(0, 1), weight=11.5)

        rt_copy = copy(rt)
        rt_copy.add_additional_weight(time=3, node=(0, 1), weight=12.5)

        self.assertEqual(id(rt.graph), id(rt_copy.graph))

        self.assertEqual(rt.get_additional_weight(2, (0, 1)), 11.5)
        self.assertEqual(rt.get_additional_weight(3, (0, 1)), 0)

        self.assertEqual(rt_copy.get_additional_weight(2, (0, 1)), 11.5)
        self.assertEqual(rt_copy.get_additional_weight(3, (0, 1)), 12.5)
