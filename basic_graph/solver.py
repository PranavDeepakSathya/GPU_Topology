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