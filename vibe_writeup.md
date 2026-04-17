# GPU Data Movement Routing — Project Log

## 1. The problem

Given a multi-GPU system, tensors need to move between GPUs. The question:
what's the fastest way to schedule those transfers, given the physical
constraints of the interconnect?

This matters because distributed training spends significant time on
communication (AllReduce, AllGather, etc.), and the optimal schedule depends
on the specific hardware topology. NCCL uses heuristics. We want to find
provably optimal schedules.

---

## 2. Hardware we studied

### GPU memory (HBM/VRAM)
Each GPU has its own high-bandwidth memory (80 GB on A100). Data must
physically reside in a GPU's memory before that GPU can compute on it.
Moving data between GPUs means writing into the destination's VRAM.

### NVLink
- Private point-to-point interconnect between GPUs (or GPU ↔ NVSwitch).
- Each link: 25 GB/s per direction (full duplex).
- A100 has 12 links per GPU → up to 300 GB/s bidirectional.
- Completely separate wiring from PCIe. NVLink traffic does not touch
  the CPU or PCIe bus at all.
- **Not a shared resource.** Each link is dedicated.

### NVSwitch
- A crossbar chip that sits between GPUs.
- Each GPU connects its 12 NVLinks to 6 NVSwitches (2 links per switch).
- The switch routes any input port to any output port — so any GPU can
  talk to any other GPU at full bandwidth simultaneously.
- On A100 systems: 2nd-gen NVSwitch. Can do multicast (send once,
  replicate to multiple destinations) and in-network reduction (SHARP —
  sum tensors inside the switch without involving a GPU).
- NVSwitch is NOT visible in `nvidia-smi topo -m`. It's not directly
  programmable. The hardware handles per-link routing automatically.
- What we control: which GPUs talk to which, how much data, in what order.
  What the hardware controls: which specific switch a given byte traverses.

### PCIe
- Standard bus connecting GPUs, NICs, SSDs to the CPU.
- PCIe Gen4 x16: ~31.5 GB/s per direction.
- **Tree topology** — devices connect to switches, switches connect to
  the CPU's root complex.
- **Shared resource.** This is the critical difference from NVLink.
  Two GPUs behind the same PCIe switch share the switch's upstream link.
  Multiple transfers through the same part of the tree compete for bandwidth.

### PCIe connection types (from `nvidia-smi topo -m`)

| Label | Path | Shared? | Typical BW |
|-------|------|---------|------------|
| NV12  | Direct NVLink (12 links bonded) | No — dedicated | ~300 GB/s |
| PIX   | One PCIe switch | Minimal — switch internal crossbar | ~25-32 GB/s |
| PXB   | Multiple PCIe switches, same root complex | Yes — upstream links | ~25-32 GB/s |
| PHB   | Through CPU host bridge | Yes — CPU root complex | ~15-28 GB/s |
| NODE  | Across root complexes within same socket | Yes — CPU internal | Lower |
| SYS   | Across CPU sockets (QPI/UPI/Infinity Fabric) | Yes — inter-socket link | ~10-20 GB/s |

Ranking (best → worst): NVLink >>> PIX ≥ PXB > PHB > NODE > SYS

Key insight: **PCIe paths are not independent edges.** They are shared
resources with contention. NVLink is the only truly independent connection.

### NIC (Network Interface Card)
- Mellanox ConnectX, plugged into a PCIe slot.
- How the machine talks to other machines (inter-node communication).
- NIC0 on the same socket as GPUs → PIX/PXB path.
- NIC1 on a different socket → SYS path (crosses inter-socket link).
- For multi-node training, GPU↔NIC bandwidth matters because one GPU
  typically relays data to/from the network for its peers.

### NUMA (Non-Uniform Memory Access)
- Each CPU socket has its own local RAM. Accessing local RAM is fast;
  accessing the other socket's RAM crosses the inter-socket link (slow).
- GPUs are physically wired to a specific socket's PCIe.
- NUMA affects CPU↔GPU transfers (data loading). Does not affect
  GPU↔GPU NVLink transfers.

---

## 3. Communication patterns in distributed training

### AllReduce
Every GPU has a gradient tensor. After the operation, every GPU has the
sum (or average) of all tensors. Used in data parallelism — every training
step. Both a reduction and a broadcast combined.

### Reduce-Scatter
Every GPU has a full tensor. They reduce (sum) and each GPU keeps one
shard of the result. Used in ZeRO-style optimizers and tensor parallelism.

### AllGather
Every GPU has a shard. After the operation, every GPU has all shards
concatenated. Inverse of Reduce-Scatter. Used in tensor parallelism
when gathering split weight matrices.

### Broadcast
One GPU has data, all GPUs need it. One-to-many.

### Point-to-point send/receive
GPU A sends activations to GPU B. Used in pipeline parallelism where
consecutive layers live on different GPUs. Asymmetric — not all GPUs
participate.

### All-to-All
Every GPU sends a different chunk to every other GPU. Used in
mixture-of-experts (MoE) routing and sequence parallelism. Most
bandwidth-hungry pattern.

---

## 4. What we can and cannot control programmatically

### What we control
- Which GPUs participate in an operation.
- What operation (send/recv, AllReduce, AllGather, etc.).
- Which tensor, how many bytes.
- When operations start (CUDA streams, events for ordering).
- Algorithm hints (ring vs tree vs NVLS, via NCCL env vars).
- Chunking — decompose one large operation into smaller overlapped ones.
- Subcommunicator structure (which GPUs form a group).

### What the hardware controls for us
- Which specific NVSwitch a byte traverses.
- How the 12 NVLinks from a GPU are load-balanced.
- PCIe packet-level scheduling and switch routing.
- Whether multicast/SHARP hardware is used (NCCL decides based on hints).
- Flow control, retries, error correction at the link layer.

### Implication for modeling
We don't model per-switch routing — the hardware does that deterministically.
We model the decisions we actually make (flow identities, timing, algorithm
choice) and treat the hardware's behavior as a known function of our inputs
(bandwidth constraints at group/endpoint level).

---

## 5. Model v1 — naive edge bandwidths

### Structure
- Nodes with no properties.
- Directed edges, each with an independent bandwidth.
- Flows: (source, destination, size).
- All flows simultaneous, splittable across multiple paths.

### LP formulation
Minimize makespan T subject to:
- Each flow fully routed.
- Each edge: total bytes through it ≤ edge bandwidth × T.

### What it got right
- Basic multi-commodity flow structure.
- Flow splitting across paths (the solver finds optimal splits).
- Worked perfectly for NVSwitch topologies (uniform, symmetric).

### What it got wrong
- **No shared resources.** Every PCIe edge was independent. In reality,
  PXB edges share upstream bandwidth. Two flows on different PXB edges
  should compete, but didn't in this model.
- **No storage limits.** Nodes had no VRAM capacity. Any node could relay
  unlimited data, which isn't physical.
- **No node properties at all.** GPUs, switches, NICs were all identical
  featureless nodes.

---

## 6. Model v2 — shared groups + storage

### Structure
- **Node**: name + storage capacity (bytes, possibly infinite).
- **Edge**: directed, (src, dst) + belongs to one Group.
- **Group**: name + bandwidth. All edges in a group share this bandwidth.
  Independent links are groups of size 1.
- **Topology**: list of nodes + list of edges. Groups are reachable
  through the edges.
- **Flow**: (source, destination, size in bytes).

### LP formulation
Variables: T (makespan), y(f,p) (bytes for flow f on path p).

Minimize T subject to:

1. **Demand:** for each flow f, sum of y(f,p) across paths = flow size.
2. **Bandwidth:** for each group g, sum of y(f,p) across all flows/paths
   that touch any edge in g ≤ group bandwidth × T.
3. **Storage:** for each node v, sum of y(f,p) across all flows/paths
   that pass through v ≤ node capacity.

### What it captures
- **Shared PCIe bandwidth.** Multiple PXB edges in one group → flows
  through different edges still compete. Correctly models the upstream
  bottleneck without knowing the internal PCIe tree structure.
- **NVLink independence.** Each NVLink direction is its own group of
  size 1 → no false contention.
- **VRAM limits.** Relay through a GPU is bounded by available memory.
- **Heterogeneous topologies.** NVLink pairs + PCIe cross-pairs + NIC
  links, all with different bandwidths and sharing rules.

### Known limitations

**Fluid approximation.** The LP computes average rate over T, not
instantaneous rate. If a pipe allows 5 GB/s and makespan is 2 seconds,
the LP permits sending 10 GB in the first second and 0 in the second.
The real pipe would be oversubscribed in the first second. This is
accurate when flows are similar in size, pessimistic when they differ.

**Tensors are not water.** The LP freely splits a flow across paths in
arbitrary proportions. In reality, a tensor may need to arrive whole at
the destination before computation can proceed. If half goes via fast
NVLink and half goes via slow PCIe, the tensor is only "ready" when the
slow half arrives. The makespan objective partially accounts for this
(it minimizes the time the last byte arrives), but the free splitting
assumption may produce schedules that aren't implementable if the tensor
can't be sharded arbitrarily.

**No temporal ordering.** All flows run simultaneously. Real collective
algorithms have phases — reduce-scatter then all-gather, pipelining
across stages. Extending to phased schedules (sum of per-phase makespans)
is straightforward but not yet implemented.

**No compute-in-network.** NVSwitch can do reductions and multicast
inside the switch. Our model treats every intermediate node as a dumb
relay. Modeling node capabilities (reduce, broadcast) is a natural
extension.

**PCIe shared bandwidth is estimated.** The group bandwidth for PXB/SYS
is not directly queryable from `nvidia-smi`. It can be measured with
`nvbandwidth` or estimated from specs. The topology matrix tells you
the connection type but not the exact shared capacity.

---

## 7. A real topology we modeled

From a 4× H 100 machine:

```
GPU0 ↔ GPU1:  NV12   (own group, 300 GB/s)
GPU2 ↔ GPU3:  NV12   (own group, 300 GB/s)
GPU0 ↔ GPU2:  PXB    (shared group "pxb", 31.5 GB/s)
GPU0 ↔ GPU3:  PXB    (shared group "pxb")
GPU1 ↔ GPU2:  PXB    (shared group "pxb")
GPU1 ↔ GPU3:  PXB    (shared group "pxb")
GPU0 ↔ NIC0:  PIX    (own group, 31.5 GB/s)
All  ↔ NIC1:  SYS    (shared group "sys", ~20 GB/s est.)
```

This is interesting because it's asymmetric. A ring flow
(GPU0→1→2→3→0) has two fast NVLink hops and two slow shared PCIe hops.
The solver can potentially find better routes by relaying through
NVLink-connected peers to avoid PCIe contention.

---

## 8. Open questions

1. Can collective algorithms be cleanly decomposed into sequences of
   point-to-point flows? (Ring AllReduce can. What about tree, NVLS,
   pipeline parallelism?)

2. How much does the fluid approximation matter in practice? Is the
   gap between LP makespan and real measured time significant?

3. For the "tensors aren't water" problem — can we add constraints
   that prevent splitting below a minimum chunk size? Or require all
   chunks to arrive within a time window?

4. What does P2P memory access (as opposed to DMA copies) mean for
   our model? P2P allows custom kernels to read/write remote GPU
   memory directly in any pattern, without launching separate
   communication kernels. Does this give finer-grained control than
   our model assumes?

5. Is there value in comparing our LP solution against NCCL's actual
   behavior (measured via NCCL debug logs + nccl-tests) to quantify
   the gap?

---

## 9. Related work

- **TACCL** (Microsoft, NSDI 2023): synthesizes optimal collective
  schedules for given topologies. Closest prior work.
- **SCCL / MSCCL**: language + runtime for custom topology-aware
  collectives.
- **Blink** (UC Berkeley): optimizes collectives for heterogeneous
  GPU topologies.
- **NCCL**: NVIDIA's production library. Heuristic algorithm selection.
  The baseline we'd compare against.

---

## 10. Code

Two files, ~210 lines total:

- `routing.py`: data model (Node, Edge, Group, Flow, Topology),
  path enumeration, LP solver, pretty printer.
- `viz.py`: pyvis-based interactive HTML visualizer.

```python
from routing import Node, Edge, Group, Flow, Topology, solve, print_solution
from viz import show_topology, show_routing
```