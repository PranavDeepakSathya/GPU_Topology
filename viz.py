"""
viz.py

show_topology(topo, path="topology.html")
show_routing(topo, sol, path="routing.html")
"""

from __future__ import annotations
from collections import defaultdict
import math
from pyvis.network import Network
from routing import Topology, Solution


def _net():
    net = Network(height="800px", width="100%", directed=True,
                  notebook=False, cdn_resources="in_line")
    net.set_options("""
    {
      "nodes": {"shape":"dot","size":24,"borderWidth":2,
        "color":{"background":"#E6F1FB","border":"#185FA5"},
        "font":{"size":14,"face":"Arial"}},
      "edges": {"smooth":{"type":"curvedCW","roundness":0.15},
        "arrows":{"to":{"enabled":true,"scaleFactor":0.6}},
        "font":{"size":10,"strokeWidth":3,"strokeColor":"#ffffff","align":"middle"},
        "color":{"color":"#888780"}},
      "physics":{"enabled":false},
      "interaction":{"hover":true,"dragNodes":true,"dragView":true,"zoomView":true}
    }""")
    return net


def _add_nodes(net: Network, topo: Topology):
    cols = max(1, int(math.ceil(math.sqrt(len(topo.nodes)))))
    for i, n in enumerate(topo.nodes):
        row, col = divmod(i, cols)
        cap = f"{n.capacity:g}" if n.capacity < math.inf else "∞"
        net.add_node(id(n), label=n.name, title=f"{n.name} (cap={cap})",
                     x=col * 200, y=row * 200)


def show_topology(topo: Topology, path: str = "topology.html") -> str:
    net = _net()
    _add_nodes(net, topo)

    # collapse parallel edges by (src, dst)
    pairs: dict[tuple, list] = defaultdict(list)
    for e in topo.edges:
        pairs[(id(e.src), id(e.dst))].append(e)

    for (u, v), edges in pairs.items():
        groups = set(e.group.name for e in edges)
        bws = [e.group.bandwidth for e in edges]
        label = ", ".join(f"{e.group.name}:{e.group.bandwidth:g}" for e in edges)
        net.add_edge(u, v, label=label, title=label, width=1.5)

    net.write_html(path, open_browser=False, notebook=False)
    return path


COLORS = ["#E24B4A", "#378ADD", "#1D9E75", "#EF9F27",
          "#7F77DD", "#D85A30", "#D4537E", "#639922"]


def show_routing(topo: Topology, sol: Solution, path: str = "routing.html") -> str:
    net = _net()
    _add_nodes(net, topo)

    # base edges dimmed
    pairs: dict[tuple, list] = defaultdict(list)
    for e in topo.edges:
        pairs[(id(e.src), id(e.dst))].append(e)
    for (u, v), edges in pairs.items():
        label = ", ".join(f"{e.group.name}:{e.group.bandwidth:g}" for e in edges)
        net.add_edge(u, v, label=label, title=label, width=1.5, color="#B4B2A9")

    # flow overlays
    for fi, (flow, used) in enumerate(sol.assignments.items()):
        color = COLORS[fi % len(COLORS)]
        bytes_per_pair: dict[tuple, float] = defaultdict(float)
        for p, bop in used:
            for e in p:
                bytes_per_pair[(id(e.src), id(e.dst))] += bop
        for (u, v), b in bytes_per_pair.items():
            pct = 100 * b / flow.size
            net.add_edge(u, v,
                         label=f"{flow.src.name}→{flow.dst.name}: {b:.0f}",
                         title=f"{b:.1f} bytes ({pct:.0f}%)",
                         color=color,
                         width=2 + 3 * (b / flow.size),
                         smooth={"type": "curvedCW", "roundness": 0.2 + 0.1 * fi})

    net.write_html(path, open_browser=False, notebook=False)
    # inject header
    with open(path, "r") as f:
        html = f.read()
    header = (f"<div style='font:14px Arial;padding:10px;background:#f5f5f5;"
              f"border-bottom:1px solid #ddd'><b>Routing</b> — "
              f"makespan={sol.makespan:.4f}, {len(sol.assignments)} flow(s)</div>")
    with open(path, "w") as f:
        f.write(html.replace("<body>", f"<body>{header}", 1))
    return path