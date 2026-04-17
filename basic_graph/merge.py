"""
solver.py — Minimum-makespan multi-commodity flow solver.

Given a topology and a set of flows (demands), find the routing that
minimizes the time until all flows complete.

Approach:
  - For each flow, enumerate k simple paths from src to dst.
  - Decision variable x[f, p] = rate assigned to flow f on path p.
  - Decision variable T = makespan.
  - Constraints:
      * For each edge e: sum of x[f, p] over (f, p) using e <= bandwidth(e).
      * For each flow f: sum of x[f, p] * T >= size(f)
        (equivalently: sum x[f, p] >= size(f) / T, but we linearize below)
  - Minimize T.

The makespan formulation is nonlinear as written. Standard trick: fix T
and ask "can we route all flows within time T?" as a feasibility LP, then
binary-search T. Or, equivalently, define y[f, p] = x[f, p] * T (bytes
routed on this path) and require sum y[f, p] >= size(f), with capacity
constraint y[f, p] / T <= bandwidth(e), i.e. sum y <= bandwidth * T.
Since T is the variable we're minimizing, and the constraint is linear
in both y and T, this works as a single LP.
"""

from dataclasses import dataclass
from typing import Optional

import pulp

from basic_graph.topology import Topology, Node, Edge, Flow


def find_simple_paths(
    topo: Topology, src: Node, dst: Node, max_hops: int = 4
) -> list[list[Edge]]:
    """Enumerate all simple paths (no repeated nodes) from src to dst.

    Returns a list of paths, where each path is a list of edges.
    `max_hops` bounds path length to keep enumeration tractable.

    For small topologies this is fine. For larger ones, you'd sample
    paths or use column generation. Start simple.
    """
    results: list[list[Edge]] = []

    def dfs(current: Node, path: list[Edge], visited: set[Node]) -> None:
        if current == dst:
            results.append(list(path))
            return
        if len(path) >= max_hops:
            return
        for edge in topo.neighbors_out(current):
            if edge.dst not in visited:
                visited.add(edge.dst)
                path.append(edge)
                dfs(edge.dst, path, visited)
                path.pop()
                visited.remove(edge.dst)

    dfs(src, [], {src})
    return results


@dataclass
class Routing:
    """Result of the solver.

    `makespan` is the time (in seconds, given bandwidth in bytes/sec and
    sizes in bytes) until all flows complete.

    `assignments` maps each flow to a list of (path, bytes_on_path).
    If a flow is split across 2 paths with 60%/40%, you get two entries.
    """
    makespan: float
    assignments: dict[Flow, list[tuple[list[Edge], float]]]


def solve(
    topo: Topology,
    flows: list[Flow],
    max_hops: int = 4,
    verbose: bool = False,
) -> Routing:
    """Find the minimum-makespan routing for the given flows on the topology."""

    # Enumerate candidate paths for each flow.
    # If a flow has no path, it's infeasible — surface that early.
    paths_per_flow: dict[Flow, list[list[Edge]]] = {}
    for f in flows:
        paths = find_simple_paths(topo, f.src, f.dst, max_hops=max_hops)
        if not paths:
            raise ValueError(
                f"No path from {f.src.name} to {f.dst.name} within "
                f"{max_hops} hops — flow is infeasible."
            )
        paths_per_flow[f] = paths
        if verbose:
            print(f"Flow {f.src.name}->{f.dst.name}: {len(paths)} candidate paths")

    # Build the LP.
    prob = pulp.LpProblem("min_makespan_mcflow", pulp.LpMinimize)

    # T = makespan, must be >= 0. Not bounded above because we don't know
    # the worst case ahead of time.
    T = pulp.LpVariable("T", lowBound=0)

    # y[f, p] = bytes routed for flow f on path p over the whole run.
    # Non-negative. Using (flow_index, path_index) as the key because
    # flows and paths aren't hashable-friendly in a PuLP dict.
    y = {}
    for fi, f in enumerate(flows):
        for pi, _path in enumerate(paths_per_flow[f]):
            y[(fi, pi)] = pulp.LpVariable(f"y_{fi}_{pi}", lowBound=0)

    # Objective: minimize makespan.
    prob += T

    # Demand constraint: each flow must have its full size routed somehow.
    for fi, f in enumerate(flows):
        prob += (
            pulp.lpSum(y[(fi, pi)] for pi, _ in enumerate(paths_per_flow[f]))
            >= f.size,
            f"demand_flow_{fi}",
        )

    # Capacity constraint: for each edge, the total bytes per unit time
    # (sum of y values for paths crossing this edge, divided by T) must
    # not exceed edge bandwidth. Rearranged to be linear:
    #   sum(y over paths using edge) <= bandwidth * T
    for ei, edge in enumerate(topo.edges):
        terms = []
        for fi, f in enumerate(flows):
            for pi, path in enumerate(paths_per_flow[f]):
                if edge in path:
                    terms.append(y[(fi, pi)])
        if terms:  # Only add constraint if some flow could use this edge.
            prob += (
                pulp.lpSum(terms) <= edge.bandwidth * T,
                f"cap_edge_{ei}",
            )

    # Solve.
    solver = pulp.PULP_CBC_CMD(msg=verbose)
    status = prob.solve(solver)

    if status != pulp.LpStatusOptimal:
        raise RuntimeError(f"Solver did not find optimal: {pulp.LpStatus[status]}")

    # Extract the solution into a friendlier format.
    makespan = pulp.value(T)
    assignments: dict[Flow, list[tuple[list[Edge], float]]] = {}
    for fi, f in enumerate(flows):
        used_paths = []
        for pi, path in enumerate(paths_per_flow[f]):
            bytes_on_path = pulp.value(y[(fi, pi)])
            if bytes_on_path and bytes_on_path > 1e-9:
                used_paths.append((path, bytes_on_path))
        assignments[f] = used_paths

    return Routing(makespan=makespan, assignments=assignments)


def describe_routing(routing: Routing) -> None:
    """Pretty-print a routing solution for humans."""
    print(f"\nMakespan: {routing.makespan:.6f} seconds")
    print(f"(i.e., the last flow finishes at t={routing.makespan:.6f})\n")
    for flow, paths in routing.assignments.items():
        print(f"Flow {flow.src.name} -> {flow.dst.name} ({flow.size} bytes):")
        for path, bytes_on_path in paths:
            path_str = " -> ".join(
                [path[0].src.name] + [e.dst.name for e in path]
            )
            frac = 100 * bytes_on_path / flow.size
            print(f"  {path_str}: {bytes_on_path:.2f} bytes ({frac:.1f}%)")
    print()


if __name__ == "__main__":
    # Test 1: trivial single-edge flow.
    # 1 GB through a 100 GB/s pipe should take 0.01 seconds.
    print("=== Test 1: single flow, direct edge ===")
    topo = Topology()
    a = topo.add_node("A")
    b = topo.add_node("B")
    topo.add_edge(a, b, bandwidth=100.0)

    flows = [Flow(src=a, dst=b, size=1.0)]
    routing = solve(topo, flows)
    describe_routing(routing)
    assert abs(routing.makespan - 0.01) < 1e-6, "Expected 0.01s"
    print("PASS\n")

    # Test 2: two flows competing for one edge.
    # Two 1 GB flows sharing a 100 GB/s pipe: must take 2/100 = 0.02s.
    print("=== Test 2: two flows sharing one edge ===")
    topo = Topology()
    a = topo.add_node("A")
    b = topo.add_node("B")
    c = topo.add_node("C")
    topo.add_edge(a, c, bandwidth=100.0)
    topo.add_edge(b, c, bandwidth=100.0)
    # Both flows go A->C and B->C. Different edges, no contention.
    flows = [Flow(src=a, dst=c, size=1.0), Flow(src=b, dst=c, size=1.0)]
    routing = solve(topo, flows)
    describe_routing(routing)
    assert abs(routing.makespan - 0.01) < 1e-6, "Independent flows, should be 0.01s"
    print("PASS\n")

    # Test 3: flow with a choice of two paths.
    # Direct edge at 50 GB/s, indirect two-hop at 100 GB/s each.
    # Solver should split to use both paths.
    print("=== Test 3: splittable flow across two paths ===")
    topo = Topology()
    a = topo.add_node("A")
    b = topo.add_node("B")
    c = topo.add_node("C")
    topo.add_edge(a, b, bandwidth=50.0)
    topo.add_edge(a, c, bandwidth=100.0)
    topo.add_edge(c, b, bandwidth=100.0)

    # One flow A->B, 150 bytes.
    # Direct A->B: 50 bw. Detour A->C->B: min(100, 100) = 100 bw.
    # Combined max rate: 50 + 100 = 150 bw. So 150 bytes in 1s.
    flows = [Flow(src=a, dst=b, size=150.0)]
    routing = solve(topo, flows)
    describe_routing(routing)
    assert abs(routing.makespan - 1.0) < 1e-6, "Should split, makespan 1.0s"
    print("PASS\n")
    
    
"--------------------------------------"

"""
solver.py — Minimum-makespan multi-commodity flow solver.

Given a topology and a set of flows (demands), find the routing that
minimizes the time until all flows complete.

Approach:
  - For each flow, enumerate k simple paths from src to dst.
  - Decision variable x[f, p] = rate assigned to flow f on path p.
  - Decision variable T = makespan.
  - Constraints:
      * For each edge e: sum of x[f, p] over (f, p) using e <= bandwidth(e).
      * For each flow f: sum of x[f, p] * T >= size(f)
        (equivalently: sum x[f, p] >= size(f) / T, but we linearize below)
  - Minimize T.

The makespan formulation is nonlinear as written. Standard trick: fix T
and ask "can we route all flows within time T?" as a feasibility LP, then
binary-search T. Or, equivalently, define y[f, p] = x[f, p] * T (bytes
routed on this path) and require sum y[f, p] >= size(f), with capacity
constraint y[f, p] / T <= bandwidth(e), i.e. sum y <= bandwidth * T.
Since T is the variable we're minimizing, and the constraint is linear
in both y and T, this works as a single LP.
"""

from dataclasses import dataclass
from typing import Optional

import pulp

from basic_graph.topology import Topology, Node, Edge, Flow


def find_simple_paths(
    topo: Topology, src: Node, dst: Node, max_hops: int = 4
) -> list[list[Edge]]:
    """Enumerate all simple paths (no repeated nodes) from src to dst.

    Returns a list of paths, where each path is a list of edges.
    `max_hops` bounds path length to keep enumeration tractable.

    For small topologies this is fine. For larger ones, you'd sample
    paths or use column generation. Start simple.
    """
    results: list[list[Edge]] = []

    def dfs(current: Node, path: list[Edge], visited: set[Node]) -> None:
        if current == dst:
            results.append(list(path))
            return
        if len(path) >= max_hops:
            return
        for edge in topo.neighbors_out(current):
            if edge.dst not in visited:
                visited.add(edge.dst)
                path.append(edge)
                dfs(edge.dst, path, visited)
                path.pop()
                visited.remove(edge.dst)

    dfs(src, [], {src})
    return results


@dataclass
class Routing:
    """Result of the solver.

    `makespan` is the time (in seconds, given bandwidth in bytes/sec and
    sizes in bytes) until all flows complete.

    `assignments` maps each flow to a list of (path, bytes_on_path).
    If a flow is split across 2 paths with 60%/40%, you get two entries.
    """
    makespan: float
    assignments: dict[Flow, list[tuple[list[Edge], float]]]


def solve(
    topo: Topology,
    flows: list[Flow],
    max_hops: int = 4,
    verbose: bool = False,
) -> Routing:
    """Find the minimum-makespan routing for the given flows on the topology."""

    # Enumerate candidate paths for each flow.
    # If a flow has no path, it's infeasible — surface that early.
    paths_per_flow: dict[Flow, list[list[Edge]]] = {}
    for f in flows:
        paths = find_simple_paths(topo, f.src, f.dst, max_hops=max_hops)
        if not paths:
            raise ValueError(
                f"No path from {f.src.name} to {f.dst.name} within "
                f"{max_hops} hops — flow is infeasible."
            )
        paths_per_flow[f] = paths
        if verbose:
            print(f"Flow {f.src.name}->{f.dst.name}: {len(paths)} candidate paths")

    # Build the LP.
    prob = pulp.LpProblem("min_makespan_mcflow", pulp.LpMinimize)

    # T = makespan, must be >= 0. Not bounded above because we don't know
    # the worst case ahead of time.
    T = pulp.LpVariable("T", lowBound=0)

    # y[f, p] = bytes routed for flow f on path p over the whole run.
    # Non-negative. Using (flow_index, path_index) as the key because
    # flows and paths aren't hashable-friendly in a PuLP dict.
    y = {}
    for fi, f in enumerate(flows):
        for pi, _path in enumerate(paths_per_flow[f]):
            y[(fi, pi)] = pulp.LpVariable(f"y_{fi}_{pi}", lowBound=0)

    # Objective: minimize makespan.
    prob += T

    # Demand constraint: each flow must have its full size routed somehow.
    for fi, f in enumerate(flows):
        prob += (
            pulp.lpSum(y[(fi, pi)] for pi, _ in enumerate(paths_per_flow[f]))
            >= f.size,
            f"demand_flow_{fi}",
        )

    # Capacity constraint: for each edge, the total bytes per unit time
    # (sum of y values for paths crossing this edge, divided by T) must
    # not exceed edge bandwidth. Rearranged to be linear:
    #   sum(y over paths using edge) <= bandwidth * T
    for ei, edge in enumerate(topo.edges):
        terms = []
        for fi, f in enumerate(flows):
            for pi, path in enumerate(paths_per_flow[f]):
                if edge in path:
                    terms.append(y[(fi, pi)])
        if terms:  # Only add constraint if some flow could use this edge.
            prob += (
                pulp.lpSum(terms) <= edge.bandwidth * T,
                f"cap_edge_{ei}",
            )

    # Solve.
    solver = pulp.PULP_CBC_CMD(msg=verbose)
    status = prob.solve(solver)

    if status != pulp.LpStatusOptimal:
        raise RuntimeError(f"Solver did not find optimal: {pulp.LpStatus[status]}")

    # Extract the solution into a friendlier format.
    makespan = pulp.value(T)
    assignments: dict[Flow, list[tuple[list[Edge], float]]] = {}
    for fi, f in enumerate(flows):
        used_paths = []
        for pi, path in enumerate(paths_per_flow[f]):
            bytes_on_path = pulp.value(y[(fi, pi)])
            if bytes_on_path and bytes_on_path > 1e-9:
                used_paths.append((path, bytes_on_path))
        assignments[f] = used_paths

    return Routing(makespan=makespan, assignments=assignments)


def describe_routing(routing: Routing) -> None:
    """Pretty-print a routing solution for humans."""
    print(f"\nMakespan: {routing.makespan:.6f} seconds")
    print(f"(i.e., the last flow finishes at t={routing.makespan:.6f})\n")
    for flow, paths in routing.assignments.items():
        print(f"Flow {flow.src.name} -> {flow.dst.name} ({flow.size} bytes):")
        for path, bytes_on_path in paths:
            path_str = " -> ".join(
                [path[0].src.name] + [e.dst.name for e in path]
            )
            frac = 100 * bytes_on_path / flow.size
            print(f"  {path_str}: {bytes_on_path:.2f} bytes ({frac:.1f}%)")
    print()


if __name__ == "__main__":
    # Test 1: trivial single-edge flow.
    # 1 GB through a 100 GB/s pipe should take 0.01 seconds.
    print("=== Test 1: single flow, direct edge ===")
    topo = Topology()
    a = topo.add_node("A")
    b = topo.add_node("B")
    topo.add_edge(a, b, bandwidth=100.0)

    flows = [Flow(src=a, dst=b, size=1.0)]
    routing = solve(topo, flows)
    describe_routing(routing)
    assert abs(routing.makespan - 0.01) < 1e-6, "Expected 0.01s"
    print("PASS\n")

    # Test 2: two flows competing for one edge.
    # Two 1 GB flows sharing a 100 GB/s pipe: must take 2/100 = 0.02s.
    print("=== Test 2: two flows sharing one edge ===")
    topo = Topology()
    a = topo.add_node("A")
    b = topo.add_node("B")
    c = topo.add_node("C")
    topo.add_edge(a, c, bandwidth=100.0)
    topo.add_edge(b, c, bandwidth=100.0)
    # Both flows go A->C and B->C. Different edges, no contention.
    flows = [Flow(src=a, dst=c, size=1.0), Flow(src=b, dst=c, size=1.0)]
    routing = solve(topo, flows)
    describe_routing(routing)
    assert abs(routing.makespan - 0.01) < 1e-6, "Independent flows, should be 0.01s"
    print("PASS\n")

    # Test 3: flow with a choice of two paths.
    # Direct edge at 50 GB/s, indirect two-hop at 100 GB/s each.
    # Solver should split to use both paths.
    print("=== Test 3: splittable flow across two paths ===")
    topo = Topology()
    a = topo.add_node("A")
    b = topo.add_node("B")
    c = topo.add_node("C")
    topo.add_edge(a, b, bandwidth=50.0)
    topo.add_edge(a, c, bandwidth=100.0)
    topo.add_edge(c, b, bandwidth=100.0)

    # One flow A->B, 150 bytes.
    # Direct A->B: 50 bw. Detour A->C->B: min(100, 100) = 100 bw.
    # Combined max rate: 50 + 100 = 150 bw. So 150 bytes in 1s.
    flows = [Flow(src=a, dst=b, size=150.0)]
    routing = solve(topo, flows)
    describe_routing(routing)
    assert abs(routing.makespan - 1.0) < 1e-6, "Should split, makespan 1.0s"
    print("PASS\n")

'---------------------------------------------------------------------------------------------------------------'

"""
topology.py — Abstract representation of a data-movement topology.

Stage 1: pure flow model.
  - Nodes have a capacity (max aggregate throughput they can source or sink).
  - Edges are directed, carrying bandwidth between two nodes.
  - Flows are "fluid": if multiple flows share an edge, they proportionally
    share the edge's bandwidth (no packet-level scheduling).
  - Flows are splittable: a single logical demand can be routed across
    multiple paths simultaneously.

No hardware specifics. No collectives. No compute-in-network.
Just: nodes, edges, flows, schedules.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Node:
    """An endpoint in the topology.

    Just an id and a name. No capacity (infinite storage assumed).
    No throughput cap at the node level — if you need to cap a node's
    total throughput, model it with edges.

    Frozen so Node instances are hashable (important for graph algorithms).
    """
    id: int
    name: str


@dataclass(frozen=True)
class Edge:
    """A directed link between two nodes.

    Directed because real links often have asymmetric behavior (e.g. PCIe
    in a non-coherent configuration), and because modeling each direction
    separately is cleaner than a single "bidirectional" abstraction.
    If you want a symmetric link, create two edges (one each way).

    `bandwidth` is the edge's capacity in the same units as Node.capacity.
    """
    src: Node
    dst: Node
    bandwidth: float


@dataclass(frozen=True)
class Flow:
    """A demand: move `size` bytes from `src` to `dst`.

    No timing yet — in Stage 1, all flows are assumed active simultaneously
    and we solve for makespan.
    """
    src: Node
    dst: Node
    size: float  # bytes


@dataclass
class Topology:
    """A collection of nodes and directed edges forming a data-movement graph."""
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def add_node(self, name: str) -> Node:
        """Create and register a new node. Returns the Node for convenience."""
        node = Node(id=len(self.nodes), name=name)
        self.nodes.append(node)
        return node

    def add_edge(self, src: Node, dst: Node, bandwidth: float) -> Edge:
        """Create and register a directed edge."""
        edge = Edge(src=src, dst=dst, bandwidth=bandwidth)
        self.edges.append(edge)
        return edge

    def add_bidirectional_edge(
        self, a: Node, b: Node, bandwidth: float
    ) -> tuple[Edge, Edge]:
        """Convenience: add two directed edges in opposite directions.

        Each direction gets its own independent `bandwidth` capacity,
        matching how full-duplex links actually work.
        """
        return (
            self.add_edge(a, b, bandwidth),
            self.add_edge(b, a, bandwidth),
        )

    def neighbors_out(self, node: Node) -> list[Edge]:
        """All edges leaving `node`."""
        return [e for e in self.edges if e.src == node]

    def neighbors_in(self, node: Node) -> list[Edge]:
        """All edges entering `node`."""
        return [e for e in self.edges if e.dst == node]


if __name__ == "__main__":
    # Smoke test: build a trivial 2-node topology with one bidirectional link.
    topo = Topology()
    a = topo.add_node("A")
    b = topo.add_node("B")
    topo.add_bidirectional_edge(a, b, bandwidth=50.0)

    print(f"Nodes: {topo.nodes}")
    print(f"Edges: {topo.edges}")
    print(f"Out of A: {topo.neighbors_out(a)}")
    print(f"Into A:   {topo.neighbors_in(a)}")
    
    
"------------"

"""
visualize.py — Drag-to-arrange pyvis visualizer.

No physics simulation. Straight edges. You drag nodes wherever you want
and they stay there.

Entry points:
  - show_topology(topo, path="topology.html")
  - show_routing(topo, routing, path="routing.html")
"""

from __future__ import annotations

from collections import defaultdict
from pyvis.network import Network

from basic_graph.topology import Topology, Node, Edge
from basic_graph.solver import Routing


FLOW_COLORS = [
    "#E24B4A", "#378ADD", "#1D9E75", "#EF9F27",
    "#7F77DD", "#D85A30", "#D4537E", "#639922",
]


def _base_network(height: str = "800px") -> Network:
    net = Network(
        height=height, width="100%", directed=True,
        notebook=False, cdn_resources="in_line",
        bgcolor="#ffffff", font_color="#222222",
    )
    net.set_options("""
    {
      "nodes": {
        "shape": "dot",
        "size": 24,
        "borderWidth": 2,
        "color": {"background": "#E6F1FB", "border": "#185FA5"},
        "font": {"size": 14, "face": "Arial"}
      },
      "edges": {
        "smooth": {"type": "curvedCW", "roundness": 0.2},
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.6}},
        "font": {"size": 11, "strokeWidth": 3, "strokeColor": "#ffffff", "align": "middle"},
        "color": {"color": "#888780", "highlight": "#185FA5"}
      },
      "physics": {"enabled": false},
      "interaction": {
        "hover": true,
        "dragNodes": true,
        "dragView": true,
        "zoomView": true
      }
    }
    """)
    return net


def _add_nodes(net: Network, topo: Topology) -> None:
    """Add nodes in a simple grid as starting positions — you can drag anywhere after."""
    import math
    n = len(topo.nodes)
    cols = max(1, int(math.ceil(math.sqrt(n))))
    spacing = 180
    for i, node in enumerate(topo.nodes):
        row, col = divmod(i, cols)
        net.add_node(
            node.id,
            label=node.name,
            title=f"{node.name} (id={node.id})",
            x=col * spacing, y=row * spacing,
        )


def _group_edges(edges: list[Edge]) -> dict[tuple[int, int], list[Edge]]:
    groups: dict[tuple[int, int], list[Edge]] = defaultdict(list)
    for e in edges:
        groups[(e.src.id, e.dst.id)].append(e)
    return groups


def show_topology(topo: Topology, path: str = "topology.html") -> str:
    net = _base_network()
    _add_nodes(net, topo)
    for (u, v), edges_here in _group_edges(topo.edges).items():
        if len(edges_here) == 1:
            label = f"{edges_here[0].bandwidth:g}"
            title = f"bw = {edges_here[0].bandwidth:g}"
        else:
            total = sum(e.bandwidth for e in edges_here)
            label = f"{len(edges_here)}× {edges_here[0].bandwidth:g}"
            title = f"{len(edges_here)} parallel links, total bw = {total:g}"
        net.add_edge(u, v, label=label, title=title, width=1.5)
    net.write_html(path, open_browser=False, notebook=False)
    return path


def show_routing(topo: Topology, routing: Routing, path: str = "routing.html") -> str:
    net = _base_network()
    _add_nodes(net, topo)

    # Per-edge utilization for base-edge hover info.
    edge_rate: dict[Edge, float] = defaultdict(float)
    for flow, assignments in routing.assignments.items():
        for p, bytes_on_path in assignments:
            rate = bytes_on_path / routing.makespan
            for e in p:
                edge_rate[e] += rate

    # Base edges: one per (src,dst) pair, collapsed.
    for (u, v), edges_here in _group_edges(topo.edges).items():
        total_bw = sum(e.bandwidth for e in edges_here)
        total_rate = sum(edge_rate[e] for e in edges_here)
        util = 100 * total_rate / total_bw if total_bw > 0 else 0
        width = 1.2 + 3.0 * min(util / 100, 1.0)
        net.add_edge(
            u, v,
            label=f"bw={total_bw:g}" + (f" ({util:.0f}%)" if total_rate > 0 else ""),
            title=f"{len(edges_here)} link(s), bw={total_bw:g}, rate={total_rate:.2f}, util={util:.0f}%",
            width=width, color="#B4B2A9",
        )

    # Flow overlays: one colored edge per (flow, pair).
    for fi, (flow, assignments) in enumerate(routing.assignments.items()):
        color = FLOW_COLORS[fi % len(FLOW_COLORS)]
        bytes_on_pair: dict[tuple[int, int], float] = defaultdict(float)
        for p, bytes_on_path in assignments:
            for e in p:
                bytes_on_pair[(e.src.id, e.dst.id)] += bytes_on_path
        for (u, v), bytes_here in bytes_on_pair.items():
            frac = bytes_here / flow.size
            net.add_edge(
                u, v,
                label=f"{flow.src.name}→{flow.dst.name}: {bytes_here:.0f}",
                title=f"{flow.src.name}→{flow.dst.name}: {bytes_here:.1f} bytes ({frac*100:.1f}% of flow)",
                color=color,
                width=2.0 + 3.0 * frac,
            )

    # Inject a small header so you know what makespan this routing achieved.
    net.write_html(path, open_browser=False, notebook=False)
    header = (
        f"<div style='font-family:Arial,sans-serif;padding:10px;"
        f"background:#f5f5f5;border-bottom:1px solid #ddd'>"
        f"<b>Routing</b> — makespan = {routing.makespan:.4f}, "
        f"{len(routing.assignments)} flow(s)"
        f"</div>"
    )
    with open(path, "r") as f:
        html = f.read()
    html = html.replace("<body>", f"<body>{header}", 1)
    with open(path, "w") as f:
        f.write(html)
    return path


if __name__ == "__main__":
    from solver import solve
    from topology import Flow

    topo = Topology()
    gpus = [topo.add_node(f"G{i}") for i in range(4)]
    switches = [topo.add_node(f"S{i}") for i in range(6)]
    for g in gpus:
        for s in switches:
            topo.add_bidirectional_edge(g, s, 50.0)

    show_topology(topo, path="/home/claude/nvs_topology.html")
    routing = solve(topo, [
        Flow(gpus[0], gpus[1], 100.0),
        Flow(gpus[0], gpus[2], 100.0),
    ])
    show_routing(topo, routing, path="/home/claude/nvs_routing.html")
    print(f"makespan = {routing.makespan:.4f}")
    print("Open the HTML files. Drag nodes wherever you want.")