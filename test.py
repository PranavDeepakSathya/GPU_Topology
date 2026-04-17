"""
routing.py

Mathematical definitions implemented verbatim:

Topology = (V, E, S, cap, bw)
  V:   set of nodes
  E:   set of directed edges (u, v)
  S:   partition of E into shared groups
  cap: V -> R+ u {inf}
  bw:  S -> R+

Flow = (s, d, sigma)
  s:     source node
  d:     destination node
  sigma: size in bytes

Path = sequence of edges (e1, ..., ek)
  e1.src = s, ek.dst = d, ei.dst = e(i+1).src, no repeated nodes

Routing Problem:
  Given topology and flows F, let P(f) = set of simple paths for flow f.

  Variables:
    T in R+                          (makespan)
    y(f, p) in R+ for f in F, p in P(f)  (bytes on path p for flow f)

  Minimize T subject to:
    1. Demand:    forall f: sum_{p in P(f)} y(f,p) = f.sigma
    2. Bandwidth: forall g in S: sum_{f,p: p cap g != empty} y(f,p) <= bw(g) * T
    3. Storage:   forall v in V: sum_{f,p: v in p} y(f,p) <= cap(v)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional
import math
import pulp


# =============================================================================
# Data model
# =============================================================================

@dataclass(frozen=True)
class Node:
    id: int
    name: str

@dataclass(frozen=True)
class Edge:
    src: Node
    dst: Node

@dataclass(frozen=True)
class Flow:
    src: Node
    dst: Node
    size: float  # sigma

@dataclass
class Topology:
    """(V, E, S, cap, bw)"""

    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    # cap: V -> R+ u {inf}
    capacity: dict[Node, float] = field(default_factory=dict)

    # S: partition of E into shared groups (group_name -> set of edges)
    shared_groups: dict[str, set[Edge]] = field(default_factory=dict)

    # bw: S -> R+
    group_bandwidth: dict[str, float] = field(default_factory=dict)

    # internal: track which group each edge belongs to
    _edge_group: dict[Edge, str] = field(default_factory=dict)

    def add_node(self, name: str, capacity: float = math.inf) -> Node:
        node = Node(id=len(self.nodes), name=name)
        self.nodes.append(node)
        self.capacity[node] = capacity
        return node

    def add_edge(self, src: Node, dst: Node, group: str, bandwidth: float | None = None) -> Edge:
        """Add a directed edge to a shared group.

        If the group doesn't exist yet, it's created with the given bandwidth.
        If it exists, bandwidth is ignored (the group already has its bandwidth).
        """
        edge = Edge(src=src, dst=dst)
        self.edges.append(edge)

        if group not in self.shared_groups:
            if bandwidth is None:
                raise ValueError(f"New group '{group}' requires a bandwidth")
            self.shared_groups[group] = set()
            self.group_bandwidth[group] = bandwidth

        self.shared_groups[group].add(edge)
        self._edge_group[edge] = group
        return edge

    def add_biedge(self, a: Node, b: Node, group_ab: str, group_ba: str | None = None,
                   bandwidth: float | None = None) -> tuple[Edge, Edge]:
        """Add two directed edges (a->b, b->a).

        If group_ba is None, the reverse edge goes in a group named group_ab + "_rev".
        For full-duplex links, the forward and reverse directions are independent
        shared groups.
        """
        if group_ba is None:
            group_ba = group_ab + "_rev"
        e1 = self.add_edge(a, b, group=group_ab, bandwidth=bandwidth)
        e2 = self.add_edge(b, a, group=group_ba, bandwidth=bandwidth)
        return (e1, e2)


# =============================================================================
# Path enumeration
# =============================================================================

def find_paths(topo: Topology, src: Node, dst: Node, max_hops: int = 6) -> list[list[Edge]]:
    """All simple paths from src to dst, up to max_hops edges."""
    # Build adjacency list once
    adj: dict[Node, list[Edge]] = defaultdict(list)
    for e in topo.edges:
        adj[e.src].append(e)

    results: list[list[Edge]] = []

    def dfs(current: Node, path: list[Edge], visited: set[Node]) -> None:
        if current == dst:
            results.append(list(path))
            return
        if len(path) >= max_hops:
            return
        for edge in adj[current]:
            if edge.dst not in visited:
                visited.add(edge.dst)
                path.append(edge)
                dfs(edge.dst, path, visited)
                path.pop()
                visited.remove(edge.dst)

    dfs(src, [], {src})
    return results


def nodes_in_path(path: list[Edge]) -> set[Node]:
    """All nodes appearing in a path (source, intermediates, destination)."""
    nodes = set()
    for e in path:
        nodes.add(e.src)
        nodes.add(e.dst)
    return nodes


def path_groups(path: list[Edge], topo: Topology) -> set[str]:
    """All shared groups touched by a path."""
    return {topo._edge_group[e] for e in path}


# =============================================================================
# Solver
# =============================================================================

@dataclass
class Solution:
    makespan: float
    # flow -> list of (path, bytes_on_path)
    assignments: dict[Flow, list[tuple[list[Edge], float]]]


def solve(topo: Topology, flows: list[Flow], max_hops: int = 6, verbose: bool = False) -> Solution:
    # Enumerate paths per flow
    paths: dict[Flow, list[list[Edge]]] = {}
    for f in flows:
        ps = find_paths(topo, f.src, f.dst, max_hops=max_hops)
        if not ps:
            raise ValueError(f"No path from {f.src.name} to {f.dst.name}")
        paths[f] = ps
        if verbose:
            print(f"  {f.src.name}->{f.dst.name}: {len(ps)} paths")

    # Precompute: for each group, which (flow_idx, path_idx) pairs touch it
    group_users: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for fi, f in enumerate(flows):
        for pi, p in enumerate(paths[f]):
            for g in path_groups(p, topo):
                group_users[g].append((fi, pi))

    # Precompute: for each node, which (flow_idx, path_idx) pairs include it
    node_users: dict[Node, list[tuple[int, int]]] = defaultdict(list)
    for fi, f in enumerate(flows):
        for pi, p in enumerate(paths[f]):
            for n in nodes_in_path(p):
                node_users[n].append((fi, pi))

    # Build LP
    prob = pulp.LpProblem("min_makespan", pulp.LpMinimize)

    T = pulp.LpVariable("T", lowBound=0)
    y: dict[tuple[int, int], pulp.LpVariable] = {}
    for fi, f in enumerate(flows):
        for pi in range(len(paths[f])):
            y[(fi, pi)] = pulp.LpVariable(f"y_{fi}_{pi}", lowBound=0)

    # Objective
    prob += T

    # Constraint 1: Demand
    for fi, f in enumerate(flows):
        prob += (
            pulp.lpSum(y[(fi, pi)] for pi in range(len(paths[f]))) == f.size,
            f"demand_{fi}",
        )

    # Constraint 2: Bandwidth (per shared group)
    for g, members in group_users.items():
        if members:
            prob += (
                pulp.lpSum(y[(fi, pi)] for fi, pi in members) <= topo.group_bandwidth[g] * T,
                f"bw_{g}",
            )

    # Constraint 3: Storage (per node)
    for n in topo.nodes:
        if topo.capacity[n] < math.inf and n in node_users:
            members = node_users[n]
            if members:
                prob += (
                    pulp.lpSum(y[(fi, pi)] for fi, pi in members) <= topo.capacity[n],
                    f"storage_{n.name}",
                )

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=verbose)
    status = prob.solve(solver)

    if status != pulp.constants.LpStatusOptimal:
        raise RuntimeError(f"LP not optimal: {pulp.LpStatus[status]}")

    # Extract solution
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


# =============================================================================
# Pretty printer
# =============================================================================

def print_solution(sol: Solution) -> None:
    print(f"\nMakespan: {sol.makespan:.6f}")
    for flow, used in sol.assignments.items():
        print(f"\n  {flow.src.name} -> {flow.dst.name} ({flow.size} bytes):")
        for path, bytes_on in used:
            route = " -> ".join([path[0].src.name] + [e.dst.name for e in path])
            pct = 100 * bytes_on / flow.size
            print(f"    {route}: {bytes_on:.2f} bytes ({pct:.1f}%)")


# =============================================================================
# Visualizer (pyvis, drag-to-arrange)
# =============================================================================

def show_topology(topo: Topology, path: str = "topology.html") -> str:
    from pyvis.network import Network
    net = _make_net()
    _add_nodes(net, topo)
    _add_base_edges(net, topo)
    net.write_html(path, open_browser=False, notebook=False)
    return path


def show_routing(topo: Topology, sol: Solution, path: str = "routing.html") -> str:
    from pyvis.network import Network
    COLORS = ["#E24B4A", "#378ADD", "#1D9E75", "#EF9F27",
              "#7F77DD", "#D85A30", "#D4537E", "#639922"]

    net = _make_net()
    _add_nodes(net, topo)
    _add_base_edges(net, topo)

    for fi, (flow, used) in enumerate(sol.assignments.items()):
        color = COLORS[fi % len(COLORS)]
        bytes_per_pair: dict[tuple[int, int], float] = defaultdict(float)
        for p, bop in used:
            for e in p:
                bytes_per_pair[(e.src.id, e.dst.id)] += bop
        for (u, v), b in bytes_per_pair.items():
            pct = 100 * b / flow.size
            net.add_edge(
                u, v,
                label=f"{flow.src.name}→{flow.dst.name}: {b:.0f}",
                title=f"{b:.1f} bytes ({pct:.0f}% of flow)",
                color=color,
                width=2 + 3 * (b / flow.size),
                smooth={"type": "curvedCW", "roundness": 0.2 + 0.1 * fi},
            )

    net.write_html(path, open_browser=False, notebook=False)
    header = (
        f"<div style='font:14px Arial;padding:10px;background:#f5f5f5;"
        f"border-bottom:1px solid #ddd'>"
        f"<b>Routing</b> — makespan = {sol.makespan:.4f}, "
        f"{len(sol.assignments)} flow(s)</div>"
    )
    with open(path, "r") as f:
        html = f.read()
    with open(path, "w") as f:
        f.write(html.replace("<body>", f"<body>{header}", 1))
    return path


def _make_net():
    from pyvis.network import Network
    net = Network(height="800px", width="100%", directed=True,
                  notebook=False, cdn_resources="in_line")
    net.set_options("""
    {
      "nodes": {"shape":"dot","size":24,"borderWidth":2,
        "color":{"background":"#E6F1FB","border":"#185FA5"},
        "font":{"size":14,"face":"Arial"}},
      "edges": {"smooth":{"type":"curvedCW","roundness":0.12},
        "arrows":{"to":{"enabled":true,"scaleFactor":0.6}},
        "font":{"size":10,"strokeWidth":3,"strokeColor":"#ffffff","align":"middle"},
        "color":{"color":"#888780"}},
      "physics":{"enabled":false},
      "interaction":{"hover":true,"dragNodes":true,"dragView":true,"zoomView":true}
    }""")
    return net


def _add_nodes(net, topo: Topology):
    import math as m
    cols = max(1, int(m.ceil(m.sqrt(len(topo.nodes)))))
    for i, n in enumerate(topo.nodes):
        row, col = divmod(i, cols)
        cap_str = f"cap={topo.capacity[n]}" if topo.capacity[n] < m.inf else "cap=∞"
        net.add_node(n.id, label=n.name, title=f"{n.name} ({cap_str})",
                     x=col * 200, y=row * 200)


def _add_base_edges(net, topo: Topology):
    # Collapse edges by (src, dst) for cleaner display
    pair_groups: dict[tuple[int, int], list[str]] = defaultdict(list)
    for e in topo.edges:
        g = topo._edge_group[e]
        pair_groups[(e.src.id, e.dst.id)].append(g)

    for (u, v), groups in pair_groups.items():
        bws = [topo.group_bandwidth[g] for g in groups]
        label = ", ".join(f"{g}:{topo.group_bandwidth[g]:g}" for g in set(groups))
        total_bw = sum(bws)
        net.add_edge(u, v, label=label,
                     title=f"groups: {', '.join(set(groups))}, total bw: {total_bw:g}",
                     width=1.5, color="#B4B2A9")


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":

    # Test 1: single edge
    print("=== Test 1: single edge ===")
    t = Topology()
    a = t.add_node("A")
    b = t.add_node("B")
    t.add_edge(a, b, group="ab", bandwidth=100.0)
    s = solve(t, [Flow(a, b, 1.0)])
    print_solution(s)
    assert abs(s.makespan - 0.01) < 1e-6
    print("PASS\n")

    # Test 2: two independent edges (no contention)
    print("=== Test 2: independent edges ===")
    t = Topology()
    a = t.add_node("A"); b = t.add_node("B"); c = t.add_node("C")
    t.add_edge(a, c, group="ac", bandwidth=100.0)
    t.add_edge(b, c, group="bc", bandwidth=100.0)
    s = solve(t, [Flow(a, c, 1.0), Flow(b, c, 1.0)])
    print_solution(s)
    assert abs(s.makespan - 0.01) < 1e-6
    print("PASS\n")

    # Test 3: shared group (contention)
    print("=== Test 3: shared group ===")
    t = Topology()
    a = t.add_node("A"); b = t.add_node("B"); c = t.add_node("C")
    t.add_edge(a, c, group="shared", bandwidth=100.0)
    t.add_edge(b, c, group="shared")  # same group, bandwidth already set
    s = solve(t, [Flow(a, c, 1.0), Flow(b, c, 1.0)])
    print_solution(s)
    assert abs(s.makespan - 0.02) < 1e-6  # 2 bytes through 100 bw
    print("PASS\n")

    # Test 4: splittable flow
    print("=== Test 4: split across two paths ===")
    t = Topology()
    a = t.add_node("A"); b = t.add_node("B"); c = t.add_node("C")
    t.add_edge(a, b, group="ab", bandwidth=50.0)
    t.add_edge(a, c, group="ac", bandwidth=100.0)
    t.add_edge(c, b, group="cb", bandwidth=100.0)
    s = solve(t, [Flow(a, b, 150.0)])
    print_solution(s)
    assert abs(s.makespan - 1.0) < 1e-6
    print("PASS\n")

    # Test 5: storage constraint
    print("=== Test 5: storage limits relay ===")
    t = Topology()
    a = t.add_node("A")
    b = t.add_node("B", capacity=5.0)  # can only hold 5 bytes
    c = t.add_node("C")
    t.add_edge(a, b, group="ab", bandwidth=100.0)
    t.add_edge(b, c, group="bc", bandwidth=100.0)
    t.add_edge(a, c, group="ac", bandwidth=10.0)  # slow direct path
    # Flow of 10 bytes. Path A->B->C can only carry 5 (storage limit on B).
    # Remaining 5 must go A->C directly at 10 bw.
    s = solve(t, [Flow(a, c, 10.0)])
    print_solution(s)
    # Direct: 5 bytes at 10 bw = 0.5s. Relay: 5 bytes at 100 bw = 0.05s.
    # Makespan = 0.5s (bottleneck is the direct path).
    assert abs(s.makespan - 0.5) < 1e-6
    print("PASS\n")

    # Test 6: NVSwitch-like topology
    print("=== Test 6: NVSwitch 4 GPUs ===")
    t = Topology()
    gpus = [t.add_node(f"G{i}", capacity=80_000.0) for i in range(4)]  # 80 GB
    switches = [t.add_node(f"S{i}") for i in range(6)]  # infinite storage
    for gi, g in enumerate(gpus):
        for si, sw in enumerate(switches):
            # Each GPU-switch pair is its own independent group (2 links @ 25 = 50)
            t.add_biedge(g, sw,
                         group_ab=f"g{gi}_s{si}",
                         group_ba=f"s{si}_g{gi}",
                         bandwidth=50.0)
    s = solve(t, [
        Flow(gpus[0], gpus[1], 100.0),
        Flow(gpus[2], gpus[3], 100.0),
    ])
    print_solution(s)
    # Each flow has 300 GB/s available (6 switches × 50), independent.
    # Makespan = 100/300 = 0.333
    assert abs(s.makespan - 1/3) < 1e-4
    print("PASS\n")

    print("All tests passed.")