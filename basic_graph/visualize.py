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