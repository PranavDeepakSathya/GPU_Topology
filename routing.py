"""
routing.py

Topology = (V, E) where each node has a capacity and each edge belongs to a group.
Groups have a bandwidth. Flows have a source, destination, and size.

Minimize makespan T subject to:
  1. Demand:    forall f: sum_{p in P(f)} y(f,p) = f.size
  2. Bandwidth: forall g: sum_{f,p: p touches g} y(f,p) <= g.bandwidth * T
  3. Storage:   forall v: sum_{f,p: v in p} y(f,p) <= v.capacity
"""

from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import math
import pulp


# --- Data model ---

@dataclass(frozen=True)
class Group:
    name: str
    bandwidth: float

@dataclass(frozen=True)
class Node:
    name: str
    capacity: float = math.inf

@dataclass(frozen=True)
class Edge:
    src: Node
    dst: Node
    group: Group

@dataclass(frozen=True)
class Flow:
    src: Node
    dst: Node
    size: float

@dataclass
class Topology:
    nodes: list[Node]
    edges: list[Edge]


# --- Path enumeration ---

def find_paths(topo: Topology, src: Node, dst: Node, max_hops: int = 6) -> list[list[Edge]]:
    adj: dict[Node, list[Edge]] = defaultdict(list)
    for e in topo.edges:
        adj[e.src].append(e)

    results: list[list[Edge]] = []

    def dfs(cur: Node, path: list[Edge], visited: set[Node]):
        if cur == dst:
            results.append(list(path))
            return
        if len(path) >= max_hops:
            return
        for e in adj[cur]:
            if e.dst not in visited:
                visited.add(e.dst)
                path.append(e)
                dfs(e.dst, path, visited)
                path.pop()
                visited.remove(e.dst)

    dfs(src, [], {src})
    return results


def nodes_in_path(path: list[Edge]) -> set[Node]:
    nodes = set()
    for e in path:
        nodes.add(e.src)
        nodes.add(e.dst)
    return nodes


def groups_in_path(path: list[Edge]) -> set[Group]:
    return {e.group for e in path}


# --- Solver ---

@dataclass
class Solution:
    makespan: float
    assignments: dict[Flow, list[tuple[list[Edge], float]]]


def solve(topo: Topology, flows: list[Flow], max_hops: int = 6) -> Solution:
    # enumerate paths
    paths: dict[Flow, list[list[Edge]]] = {}
    for f in flows:
        ps = find_paths(topo, f.src, f.dst, max_hops)
        if not ps:
            raise ValueError(f"No path from {f.src.name} to {f.dst.name}")
        paths[f] = ps

    # collect all groups
    all_groups: set[Group] = set()
    for e in topo.edges:
        all_groups.add(e.group)

    # precompute: group -> list of (fi, pi) that touch it
    group_users: dict[Group, list[tuple[int, int]]] = defaultdict(list)
    for fi, f in enumerate(flows):
        for pi, p in enumerate(paths[f]):
            for g in groups_in_path(p):
                group_users[g].append((fi, pi))

    # precompute: node -> list of (fi, pi) that include it
    node_users: dict[Node, list[tuple[int, int]]] = defaultdict(list)
    for fi, f in enumerate(flows):
        for pi, p in enumerate(paths[f]):
            for n in nodes_in_path(p):
                node_users[n].append((fi, pi))

    # build LP
    prob = pulp.LpProblem("makespan", pulp.LpMinimize)
    T = pulp.LpVariable("T", lowBound=0)
    y = {}
    for fi, f in enumerate(flows):
        for pi in range(len(paths[f])):
            y[(fi, pi)] = pulp.LpVariable(f"y_{fi}_{pi}", lowBound=0)

    prob += T

    # constraint 1: demand
    for fi, f in enumerate(flows):
        prob += pulp.lpSum(y[(fi, pi)] for pi in range(len(paths[f]))) == f.size

    # constraint 2: bandwidth
    for g in all_groups:
        if group_users[g]:
            prob += pulp.lpSum(y[k] for k in group_users[g]) <= g.bandwidth * T

    # constraint 3: storage
    for n in topo.nodes:
        if n.capacity < math.inf and node_users[n]:
            prob += pulp.lpSum(y[k] for k in node_users[n]) <= n.capacity

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if prob.status != pulp.constants.LpStatusOptimal:
        raise RuntimeError(f"LP not optimal: {pulp.LpStatus[prob.status]}")

    # extract
    makespan = pulp.value(T)
    assignments: dict[Flow, list[tuple[list[Edge], float]]] = {}
    for fi, f in enumerate(flows):
        used = []
        for pi, p in enumerate(paths[f]):
            val = pulp.value(y[(fi, pi)])
            if val and val > 1e-9:
                used.append((p, val))
        assignments[f] = used

    return Solution(makespan=makespan, assignments=assignments)


# --- Pretty printer ---

def print_solution(sol: Solution):
    print(f"\nMakespan: {sol.makespan:.6f}")
    for flow, used in sol.assignments.items():
        print(f"\n  {flow.src.name} -> {flow.dst.name} ({flow.size} bytes):")
        for path, bytes_on in used:
            route = " -> ".join([path[0].src.name] + [e.dst.name for e in path])
            pct = 100 * bytes_on / flow.size
            print(f"    {route}: {bytes_on:.2f} bytes ({pct:.1f}%)")


# --- Tests ---

if __name__ == "__main__":

    # Test 1: single edge
    print("=== Test 1: single edge ===")
    g1 = Group("link", 100.0)
    a = Node("A"); b = Node("B")
    t = Topology([a, b], [Edge(a, b, g1)])
    s = solve(t, [Flow(a, b, 1.0)])
    print_solution(s)
    assert abs(s.makespan - 0.01) < 1e-6
    print("PASS\n")

    # Test 2: independent edges
    print("=== Test 2: independent edges ===")
    g_ac = Group("ac", 100.0); g_bc = Group("bc", 100.0)
    a = Node("A"); b = Node("B"); c = Node("C")
    t = Topology([a, b, c], [Edge(a, c, g_ac), Edge(b, c, g_bc)])
    s = solve(t, [Flow(a, c, 1.0), Flow(b, c, 1.0)])
    print_solution(s)
    assert abs(s.makespan - 0.01) < 1e-6
    print("PASS\n")

    # Test 3: shared group
    print("=== Test 3: shared group ===")
    shared = Group("shared", 100.0)
    a = Node("A"); b = Node("B"); c = Node("C")
    t = Topology([a, b, c], [Edge(a, c, shared), Edge(b, c, shared)])
    s = solve(t, [Flow(a, c, 1.0), Flow(b, c, 1.0)])
    print_solution(s)
    assert abs(s.makespan - 0.02) < 1e-6
    print("PASS\n")

    # Test 4: splittable flow
    print("=== Test 4: split across two paths ===")
    g_ab = Group("ab", 50.0); g_ac = Group("ac", 100.0); g_cb = Group("cb", 100.0)
    a = Node("A"); b = Node("B"); c = Node("C")
    t = Topology([a, b, c], [Edge(a, b, g_ab), Edge(a, c, g_ac), Edge(c, b, g_cb)])
    s = solve(t, [Flow(a, b, 150.0)])
    print_solution(s)
    assert abs(s.makespan - 1.0) < 1e-6
    print("PASS\n")

    # Test 5: storage constraint
    print("=== Test 5: storage limits relay ===")
    g_ab = Group("ab", 100.0); g_bc = Group("bc", 100.0); g_ac = Group("ac", 10.0)
    a = Node("A"); b = Node("B", capacity=5.0); c = Node("C")
    t = Topology([a, b, c], [Edge(a, b, g_ab), Edge(b, c, g_bc), Edge(a, c, g_ac)])
    s = solve(t, [Flow(a, c, 10.0)])
    print_solution(s)
    assert abs(s.makespan - 0.5) < 1e-6
    print("PASS\n")

    # Test 6: NVSwitch
    print("=== Test 6: NVSwitch 4 GPUs ===")
    gpus = [Node(f"G{i}", capacity=80_000.0) for i in range(4)]
    switches = [Node(f"S{i}") for i in range(6)]
    edges = []
    for gi, g in enumerate(gpus):
        for si, sw in enumerate(switches):
            fwd = Group(f"g{gi}_s{si}", 50.0)
            rev = Group(f"s{si}_g{gi}", 50.0)
            edges.append(Edge(g, sw, fwd))
            edges.append(Edge(sw, g, rev))
    t = Topology(gpus + switches, edges)
    s = solve(t, [Flow(gpus[0], gpus[1], 100.0), Flow(gpus[2], gpus[3], 100.0)])
    print_solution(s)
    assert abs(s.makespan - 1/3) < 1e-4
    print("PASS\n")

    print("All tests passed.")