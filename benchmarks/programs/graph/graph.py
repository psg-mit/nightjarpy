from typing import Any, Dict, List, Tuple

import nightjarpy as nj


class Graph:
    """A directed graph. Nodes are represented by a set of node values. Edges are represented by a dictionary of source node value to a set of target node values."""

    nodes: set[int]
    edges: dict[int, set[int]]

    def __init__(self, nodes: set[int], edges: dict[int, set[int]]):
        self.nodes = nodes
        self.edges = edges

    def __str__(self):
        return f"Graph(nodes={self.nodes}, edges={self.edges})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other: "Graph"):
        for src in self.edges:
            if src not in other.edges:
                return False

            if not self.edges[src] == other.edges[src]:
                return False

        for src in other.edges:
            if src not in self.edges:
                return False

            if not self.edges[src] == other.edges[src]:
                return False

        return self.nodes == other.nodes

    def __hash__(self):
        return hash((self.nodes, self.edges))

    def __len__(self):
        return len(self.nodes)

    def __contains__(self, node):
        return node in self.nodes


@nj.fn
def main(queries: list[str], graph: Graph):
    responses = []
    for query in queries:
        """natural
        Perform the <query> with respect to <graph>,
        where nodes are paper IDs and edges point
        from a cited paper to a set of papers that cite it.
        Break if the <query> indicates termination.
        Else, save a <:response> and update <graph> to answer <query>.
        <:response> should contain only the value, no prefix or suffix.
        """
        print(f"A: {response}")
        responses.append(response)
    return responses


import random

#### Tests ####
from copy import deepcopy


def make_graph(n_nodes: int = 10, edge_density: float = 0.5):
    graph = Graph(nodes=set(range(n_nodes)), edges={})
    nodes = list(range(n_nodes))
    random.shuffle(nodes)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if random.random() < edge_density:
                if nodes[i] not in graph.edges:
                    graph.edges[nodes[i]] = set()
                graph.edges[nodes[i]].add(nodes[j])
    return graph


def run() -> Tuple[Dict[str, Tuple[Any, Any]], Dict[str, Any], Dict[str, bool]]:

    random.seed(42)
    original_graph = make_graph(n_nodes=25)
    random.seed()

    def check_out_degree(responses, graph):
        return len(responses) == 1 and responses[0] == len(original_graph.edges[19])

    def check_path(responses, graph):
        def check_path_exists(graph: Graph, nodes: tuple[int, int]) -> bool:
            a, b = nodes
            visited = set()
            stack = [a]
            while stack:
                current = stack.pop()
                if current == b:
                    return True
                if current not in visited:
                    visited.add(current)
                    stack.extend(graph.edges.get(current, set()))  # Follow only outgoing edges
            return False

        ground_truth = check_path_exists(original_graph, (23, 4))
        ground_truth_yes_no = "yes" if ground_truth else "no"
        correct_ans = (responses[0] == ground_truth) or (str(responses[0]).lower() == ground_truth_yes_no)

        return len(responses) == 1 and responses[0] == ground_truth

    def check_intersect(responses, graph):
        cites_x = original_graph.edges.get(5, set())
        cites_y = original_graph.edges.get(7, set())

        n_cites_both = len(cites_x.intersection(cites_y))
        return len(responses) == 1 and responses[0] == n_cites_both

    def check_update(responses, graph):
        correct_graph = deepcopy(original_graph)

        node_x = 14
        node_y = 5

        if node_x not in correct_graph.edges:
            correct_graph.edges[node_x] = set()
        correct_graph.edges[node_x].add(node_y)

        return graph == correct_graph

    def check_remove(responses, graph):
        correct_graph = deepcopy(original_graph)

        node_x = 0

        correct_graph.nodes.remove(node_x)
        for src, targets in correct_graph.edges.items():
            if node_x in targets:
                targets.remove(node_x)
        if node_x in correct_graph.edges:
            del correct_graph.edges[node_x]

        return graph == correct_graph

    def check_exit(responses, graph):
        return len(responses) == 0

    queries = [
        ("Give the number of papers that cite paper 19.", check_out_degree),
        ("Does paper 23 directly/indirectly get cited by paper 4.", check_path),
        ("How many papers cite both paper 7 and paper 5?", check_intersect),
        ("Update the graph so paper 5 cites paper 14.", check_update),
        ("Remove paper 0 from the graph completely.", check_remove),
        ("Exit, please.", check_exit),
    ]

    outputs = {}
    errors = {}
    hard_results = {}
    for i, query in enumerate(queries):
        hard_results[f"test_{i}"] = False

    for i, (query, check) in enumerate(queries):
        graph = deepcopy(original_graph)
        try:
            responses = main([query], graph)
        except Exception as e:
            errors["all"] = e
        else:
            outputs["all"] = responses

            try:
                hard_results[f"test_{i}"] = check(responses, graph)
            except Exception as e:
                errors[f"test_{i}"] = e
                hard_results[f"test_{i}"] = False

    return outputs, errors, hard_results


if __name__ == "__main__":
    results, errors, hard_results = run()
    print(results)
    print(hard_results)
    print(errors)
