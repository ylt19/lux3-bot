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

### License
The code in this directory is under the Apache License 2.0, as originally licensed in w9PcJLyb/pathfinding. For more details, please refer to the LICENSE file in w9PcJLyb/pathfinding https://github.com/w9PcJLyb/pathfinding/blob/main/LICENSE.
