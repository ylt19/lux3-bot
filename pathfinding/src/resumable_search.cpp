#include "include/resumable_search.h"


ResumableBFS::ResumableBFS(AbsGraph *graph, int start, bool reverse) : ResumableSearch(graph, start, reverse) {
    nodes_.resize(graph->size());

    openset_.push(start);
    Node &n0 = nodes_[start];
    n0.distance = 0;
}

void ResumableBFS::clear() {
    std::queue<int> empty;
    std::swap(openset_, empty);
    for (Node &node : nodes_)
        node.clear();
}

void ResumableBFS::set_start_node(int start) {
    if (start_ != start) {
        start_ = start;
        clear();

        openset_.push(start);
        Node &n0 = nodes_[start];
        n0.distance = 0;
    }
}

double ResumableBFS::distance(int node_id) {
    Node& node = nodes_[node_id];
    if (node.distance < 0)
        search(node_id);

    return node.distance;
}

Path ResumableBFS::reconstruct_path(int node_id) {
    int p = node_id;
    Path path;
    while (p >= 0) {
        path.push_back(p);
        p = nodes_[p].parent;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

Path ResumableBFS::find_path(int node_id) {
    Node& node = nodes_[node_id];
    if (node.distance < 0)
        search(node_id);

    if (node.distance >= 0)
        return reconstruct_path(node_id);

    return {};
}

void ResumableBFS::search(int node_id) {
    while (!openset_.empty()) {
        int current_id = openset_.front();
        openset_.pop();

        Node& current = nodes_[current_id];

        for (auto& [n, cost] : graph->get_neighbors(current_id, reverse_)) {
            Node &node = nodes_[n];
            if (node.distance < 0) {
                node.parent = current_id;
                node.distance = current.distance + 1;
                openset_.push(n);
            }
        }

        if (current_id == node_id)
            return;
    }
}

ResumableAStar::ResumableAStar(AbsGraph *graph, int start, bool reverse) : ResumableSearch(graph, start, reverse) {
    nodes_.resize(graph->size());

    openset_.push({0, start});
    Node &n0 = nodes_[start];
    n0.distance = 0;
}

void ResumableAStar::clear() {
    openset_ = Queue();
    for (Node& node : nodes_)
        node.clear();
}

void ResumableAStar::set_start_node(int start) {
    if (start_ != start) {
        start_ = start;
        end_ = -1;
        clear();

        openset_.push({0, start});
        Node &n0 = nodes_[start];
        n0.distance = 0;
    }
}

double ResumableAStar::distance(int node_id) {
    Node& node = nodes_[node_id];
    if (!node.closed)
        search(node_id);

    return node.distance;
}

Path ResumableAStar::reconstruct_path(int node_id) {
    int p = node_id;
    Path path;
    while (p >= 0) {
        path.push_back(p);
        p = nodes_[p].parent;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

Path ResumableAStar::find_path(int node_id) {
    Node& node = nodes_[node_id];
    if (!node.closed)
        search(node_id);

    if (node.distance >= 0)
        return reconstruct_path(node_id);

    return {};
}

void ResumableAStar::search(int node_id) {
    if (end_ == -1)
        end_ = node_id;

    while (!openset_.empty()) {
        key top = openset_.top();
        openset_.pop();

        int current_id = top.second;
        Node& current = nodes_[current_id];

        if (current.closed)
            continue;

        current.closed = true;

        for (auto& [n, cost] : graph->get_neighbors(current_id, reverse_)) {
            Node &node = nodes_[n];
            double new_distance = current.distance + cost;
            if (node.distance < 0) {
                node.f = new_distance + graph->estimate_distance(n, end_);
                node.distance = new_distance;
                node.parent = current_id;
                openset_.push({node.f, n});
            }
            else if (node.distance > new_distance) {
                node.f = node.f - node.distance + new_distance;
                node.distance = new_distance;
                node.parent = current_id;
                openset_.push({node.f, n});
            }
        }

        if (current_id == node_id)
            return;
    }

    Node& node = nodes_[node_id];
    node.closed = true;
}

ResumableDijkstra::ResumableDijkstra(AbsGraph *graph, int start, bool reverse) : ResumableSearch(graph, start, reverse) {
    nodes_.resize(graph->size());

    openset_.push({0, start});
    Node &n0 = nodes_[start]; 
    n0.distance = 0;
}

void ResumableDijkstra::clear() {
    openset_ = Queue();
    for (Node &node : nodes_)
        node.clear();
}

void ResumableDijkstra::set_start_node(int start) {
    if (start_ != start) {
        start_ = start;
        clear();

        openset_.push({0, start});
        Node &n0 = nodes_[start];
        n0.distance = 0;
    }
}

double ResumableDijkstra::distance(int node_id) {
    Node& node = nodes_[node_id];
    if (!node.closed)
        search(node_id);

    return node.distance;
}

Path ResumableDijkstra::reconstruct_path(int node_id) {
    int p = node_id;
    Path path;
    while (p >= 0) {
        path.push_back(p);
        p = nodes_[p].parent;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

Path ResumableDijkstra::find_path(int node_id) {
    Node& node = nodes_[node_id];
    if (!node.closed)
        search(node_id);

    if (node.distance >= 0)
        return reconstruct_path(node_id);

    return {};
}

void ResumableDijkstra::search(int node_id) {
    while (!openset_.empty()) {
        key top = openset_.top();
        openset_.pop();

        int current_id = top.second;
        Node& current = nodes_[current_id];

        if (current.closed)
            continue;

        current.closed = true;

        for (auto& [n, cost] : graph->get_neighbors(current_id, reverse_)) {
            Node &node = nodes_[n];
            double new_distance = current.distance + cost;
            if (node.distance < 0 || node.distance > new_distance) {
                node.distance = new_distance;
                node.parent = current_id;
                openset_.push({new_distance, n});
            }
        }

        if (current_id == node_id)
            return;
    }

    Node& node = nodes_[node_id];
    node.closed = true;
}

ResumableSpaceTimeDijkstra::ResumableSpaceTimeDijkstra(
    AbsGraph *graph,
    int start,
    int terminal_time,
    const ReservationTable *rt
) : ResumableSearch(graph, start, false) {

    graph_size_ = graph->size();
    terminal_time_ = terminal_time;
    rt_ = rt;

    nodes_.emplace(start, Node(nullptr, start, 0, 0));
    openset_.push({0, &nodes_.at(start)});
}

void ResumableSpaceTimeDijkstra::clear() {
    openset_ = Queue();
    nodes_.clear();
}

void ResumableSpaceTimeDijkstra::set_start_node(int start) {
    if (start_ != start) {
        start_ = start;
        clear();

        nodes_.emplace(start, Node(nullptr, start, 0, 0));
        openset_.push({0, &nodes_.at(start)});
    }
}

double ResumableSpaceTimeDijkstra::distance(int node_id) {
    return distance(node_id, -1);
}

double ResumableSpaceTimeDijkstra::distance(int node_id, int time) {
    Node* node = nullptr;

    if (time < 0) {
        double min_distance = -1;
        for (int t = 0; t <= terminal_time_; t++) {
            int st = node_id + t * graph_size_;
            if (nodes_.count(st) && nodes_[st].closed) {
                if (min_distance == -1 || nodes_[st].distance < min_distance) {
                    min_distance = nodes_[st].distance;
                    node = &nodes_[st];
                }
            }
        }
    }
    else {
        int st = node_id + time * graph_size_;
        if (nodes_.count(st))
            node = &nodes_[st];
    }

    if (!node || !node->closed)
        node = search(node_id, time);

    if (node)
        return node->distance;

    return -1;  // inf
}

Path ResumableSpaceTimeDijkstra::reconstruct_path(Node* node) {
    Path path = {node->node_id};
    while (node->parent != nullptr) {
        node = node->parent;
        path.push_back(node->node_id);
    }
    std::reverse(path.begin(), path.end());
    return path;
}

Path ResumableSpaceTimeDijkstra::find_path(int node_id) {
    return find_path(node_id, -1);
}

Path ResumableSpaceTimeDijkstra::find_path(int node_id, int time) {
    Node* node = nullptr;

    if (time < 0) {
        double min_distance = -1;
        for (int t = 0; t <= terminal_time_; t++) {
            int st = node_id + t * graph_size_;
            if (nodes_.count(st) && nodes_[st].closed) {
                if (min_distance == -1 || nodes_[st].distance < min_distance) {
                    min_distance = nodes_[st].distance;
                    node = &nodes_[st];
                }
            }
        }
    }
    else {
        int st = node_id + time * graph_size_;
        if (nodes_.count(st) && nodes_[st].closed)
            node = &nodes_[st];
    }

    if (!node)
        node = search(node_id, time);

    if (node)
        return reconstruct_path(node);

    return {};
}

ResumableSpaceTimeDijkstra::Node* ResumableSpaceTimeDijkstra::search(int goal, int time) {
    // cout << "start search to " << graph->node_to_string(goal) << " time=" << time << endl;

    auto process_node = [&] (int node_id, double cost, Node* current) {
        int next_time = current->time + 1;
        double distance = current->distance + cost + rt_->get_additional_weight(next_time, node_id);

        int st = node_id + next_time * graph_size_;
        if (!nodes_.count(st)) {
            nodes_.emplace(st, Node(current, node_id, next_time, distance));
            openset_.push({distance, &nodes_.at(st)});
            // cout << " - add node ";
            // print_node(nodes_[st]);
        }
        else if (nodes_.at(st).distance > distance) {
            Node& n = nodes_.at(st);
            n.distance = distance;
            n.parent = current;
            openset_.push({distance, &n});
            // cout << " - update node ";
            // print_node(nodes_[st]);
        }
    };

    while (!openset_.empty()) {
        auto [distance, current] = openset_.top();
        openset_.pop();

        int current_time = current->time;

        // cout << "explore node ";
        // print_node(*current);

        if (current->closed)
            continue;

        // expand node
        if (current_time < terminal_time_) {
            // pause action
            if (!rt_->is_reserved(current_time + 1, current->node_id))
                process_node(current->node_id, graph->get_pause_action_cost(current->node_id), current);

            // movement actions
            auto reserved_edges = rt_->get_reserved_edges(current_time, current->node_id);
            for (auto &[node_id, cost] : graph->get_neighbors(current->node_id)) {
                if (!reserved_edges.count(node_id) && !rt_->is_reserved(current_time + 1, node_id))
                    process_node(node_id, cost, current);
            }
        }

        current->closed = true;

        if (current->node_id == goal)
            if (time < 0 || current_time == time) {
                return current;
            }
        }

    return nullptr;
}

void ResumableSpaceTimeDijkstra::print_node(Node& node) {
    cout << "Node(" << graph->node_to_string(node.node_id);
    cout << ", time=" << node.time << ", distance=" << node.distance;
    cout << ", closed=" << node.closed << ")" << endl;
}
