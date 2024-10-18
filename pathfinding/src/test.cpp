#include <iostream>
#include "include/grid.h"
#include "include/bfs.h"
#include "include/dijkstra.h"
#include "include/a_star.h"
#include "include/space_time_a_star.h"


void test_grid() {
    cout << "\nTest grid" << endl;

    Grid grid(4, 5);
    grid.passable_left_right_border = false;
    grid.passable_up_down_border = false;
    grid.set_diagonal_movement(0);

    grid.add_obstacle(grid.get_node_id({1, 2}));
    grid.add_obstacle(grid.get_node_id({2, 1}));
    grid.add_obstacle(grid.get_node_id({2, 2}));

    cout << "Obstacle map:" << endl;
    grid.show_obstacle_map();

    cout << "neighbors (1, 1) :: ";
    for (auto &[n, cost]: grid.get_neighbors(grid.get_node_id({1, 1}))) {
        cout << grid.get_coordinates(n) << " ";
    }
    cout << endl;

    cout << "neighbors (1, 0) :: ";
    for (auto &[n, cost]: grid.get_neighbors(grid.get_node_id({1, 0}))) {
        cout << grid.get_coordinates(n) << " ";
    }
    cout << endl;

    Grid::Point start = {0, 0}, end = {3, 3};
    int start_node = grid.get_node_id(start);
    int end_node = grid.get_node_id(end);
    cout << "\nFinding path in the grid from " << start << " to " << end << endl;

    BFS bfs(&grid);
    Dijkstra dijkstra(&grid);
    AStar astar(&grid);
    SpaceTimeAStar stastar(&grid);

    vector<pair<std::string, AbsPathFinder*>> finders = {
        {"BFS", &bfs},
        {"Dijkstra", &dijkstra},
        {"A*", &astar},
        {"Space-Time A*", &stastar}
    };

    vector<int> path;
    for (auto &[name, finder] : finders) {
        path = finder->find_path(start_node, end_node);
        cout << name << ": ";
        grid.print_path(path);
    }

    cout << "\nSet passable_left_right_border = true" << endl;
    grid.passable_left_right_border = true;
    path = astar.find_path(start_node, end_node);
    cout << "A*: ";
    grid.print_path(path);

    cout << "\nSet diagonal_movement = 3 (always)" << endl;
    grid.set_diagonal_movement(3);
    path = astar.find_path(start_node, end_node);
    cout << "A*: ";
    grid.print_path(path);

    cout << "\nSet passable_up_down_border = true" << endl;
    grid.passable_up_down_border = true;
    path = astar.find_path(start_node, end_node);
    cout << "A*: ";
    grid.print_path(path);
}

int main() {
    test_grid();
    return 0;
}
