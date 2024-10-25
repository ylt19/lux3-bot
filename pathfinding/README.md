# pathfinding

This directory contains code that has been copied from [w9PcJLyb/pathfinding](https://github.com/w9PcJLyb/pathfinding).

### Modifications
The following changes have been made to the copied code:

- Remove unused data structures and algorithms (HexGrid, Grid3D, MAPF, some pathfinding algorithms)
- Add reservation_table support in the animate_grid function to visualize dynamic obstacles
- Add variable pause action cost.

    In the original repository, the pause_action_cost was constant across all nodes.
Here i’ve added a new option: the pause action cost can be set to be equal to the weight of the node.
To enable new option, use the keyword "node.weight" like this:

    ```python
    grid = Grid(weights, pause_action_cost="node.weight")
    ```

    This option can be used in the SpaceTimeAStar algorithm when searching for the shortest path with dynamic obstacles.

- Add moving weights.

    For example, this adds an additional weight of 3.5 to the nodes (1, 0), (2, 0), and (3, 0) at timesteps 0, 1, and 2, respectively:

    ```python
    reservation_table.add_weight_path([(1, 0), (2, 0), (3, 0)], weight=3.5)
    ```

    Сan be used in the SpaceTimeAStar algorithm when searching for the shortest path in a dynamic environment where these dynamics can be predicted.

### License
The code in this directory is under the Apache License 2.0, as originally licensed in w9PcJLyb/pathfinding. For more details, please refer to the LICENSE file in w9PcJLyb/pathfinding https://github.com/w9PcJLyb/pathfinding/blob/main/LICENSE.
