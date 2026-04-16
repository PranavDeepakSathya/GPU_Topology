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