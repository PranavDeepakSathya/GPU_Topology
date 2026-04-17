okay, let me fucking brain dump 


1. understand the full topology collected on the 4x A100s (including non NVLINK stuff, and the rules of commuication etc etc etc)

2. understand all the communication schemes and why/when they are used. 

3. Collect MORE topologies for thinking against

4.  Is the goal of the project to route alogrithms/communication (communication schemes) or is it to simply answer, hey these are my tensors, they are sitting on these gpus, they need to go to these gpus, and more importantly can algorithms be reduced to a bunch of flows in some nice way

5. is the topology itself reducible (to some extent) to the naive graph problem we currently have? 

6.  what are we doing?

7. Programmatic control: Even if we think of exact routing in the topology, can we program that? or if the hardware takes away some control from us, can we deterministically model what would happen on the hardware if we deployed a particular protocol at the finest granulairty of programmatic control we have? 

8. we have a very naive water flow based modelling, how much of that actually holds with hardware? 
what is the linear programming alogrithm doing? what are the constrians of the water flow? 

https://blog.dailydoseofds.com/p/all-reduce-and-ring-reduce-for-model read this 



I guess we're sorta like thinking about making a desicion right now 

Starting point: 

Static specification: 
Give the topology, give the specification this tensor needs to go from this gpu to this gpu, this is a static specification, and we just need to schedule it. 

So the first pass is restricted to data movement ONLY, and particularly INTER-GPU, so GPU capacities, edge bandwidths, limits this that, I guess we need to model and solve a general problem where its like 
say I decide oh this node can broadcast data, (like an NVSWITCH broadvcasting without any contention) that should also be statable, or this node has some other limit or this or that, basically. 


So I guess NVswitch is capable of (newly) doing arithmetic as well, like reduction inside the nswitch and some crazy shit like that, I can look into that, but for now, I guess anything that isn't MOVE A TO B as a static spec is out of the model. 

I guess for the second pass, we can sorta think about refining the notion of "tensor" and also I guess thinking about programmatic granularity and control, hmm maybe we should think about that right now. 

okay, after collecting some data for h100 link, we saw some rather interesting things, first of all, since the nvswitch or whatever the fuck is not part of the topology matrix I guess its not programmatically controlled? we can read more about nvlink itself. 

second of all, maybe extact per lane shit is not needed, but rather we only need the throughput between two gpus, or from pcie to gpu or this and that. 

Okay so this means that firstly, we have to start reading up on hardware. 


IN the basic MODEL we assume that you can just split the flows however you please 
this is NOT A GOOD IDEA, like if I want to transfer 1 tensor, It might be the case that I need to use that tensor in some operation. And also my layouts and stuff might not permit 
the kind of sharding I want to do. 

As much as I want it to be, TENSORS ARE NOT WATER!  

for example, let us say half a tensor is running through slow PCIE and the other half is 
running though NVLINK fast, the time it takes for the tensor to materialize and be ready 
in the desination gpu is bound by the PCIE.

I guess to some extent, the makespan case which we are doing, which is minimize the time 
it takes for the last flow to complete, is covering for that. 

The other, more important issue, is that our sharding might not be ideal towards the way that the water based splitting of transfers happens, and I am not sure if you can like push two loads into the same pipe and this and that, so I guess ACTUALLY READING HARDWARE AND LIKE WRITING SOME DATA TRANSFERS FOR NVIDIA if possible and modelling off of that might be a good idea. Or at least enumerate possiblities and allow for flexibility in the model. 


Need to read up on the components more, example, PCIE bandwidth is more of a shared resource AFAIK so just edge capacities isn't going to work, its more like all edges connected should be such that that the sum of the rates flowing through is less that the total bandwidth 
(not too sure about this) 

Obviously limits exist. (node capacities) 


also p2p path is different topology, I guess Idk what exactly p2p does, but our boy gau 
nernest https://gau-nernst.github.io/amd-a2a/#single-gpu-moe says 
"Before talking about custom HIP kernels, let’s discuss Peer-to-Peer (P2P) and Symmetric memory, the fundamental building blocks of multi-GPU communications. P2P memory access can be broadly understood as the ability for devices to read from and write to memory of other devices. This is very powerful as we can write custom kernels that perform remote memory access directly, in any patterns we want, without launching separate communication kernels or issuing Direct Memory Access (DMA) commands. Ironically, I read CUDA C++ documentation to understand P2P usage on MI300X, though it also means that AMD’s strategy of mirroring CUDA API in HIP has some benefits." 

so does that mean non p2p path requires DMA commands and kernel launches? 



# GPU Interconnect + Topology Reading List

## Phase 0 — Basics
- PCIe (overview)
  https://en.wikipedia.org/wiki/PCI_Express

## Phase 1 — GPU Interconnects
- NVLink intro (NVIDIA blog)
  https://developer.nvidia.com/blog/how-nvlink-will-enable-faster-easier-multi-gpu-computing/

- PCIe vs NVLink (practical perspective)
  https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/#NVLink

## Phase 2 — Topology (important)
- nvidia-smi topology docs
  https://docs.nvidia.com/deploy/nvidia-smi/index.html#topology

- NCCL topology guide
  https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/topology.html

## Phase 3 — Communication Algorithms
- AllReduce / Ring explanation
  https://blog.dailydoseofds.com/p/all-reduce-and-ring-reduce-for-model

- NCCL internals (collectives)
  https://developer.nvidia.com/blog/fast-multi-gpu-collectives-nccl/

## Phase 4 — NUMA (light)
- NUMA overview
  https://en.wikipedia.org/wiki/Non-uniform_memory_access

## Phase 5 — NVLink Fabric (inter-node)
- NVLink overview / switch system
  https://www.nvidia.com/en-us/data-center/nvlink/

- Hopper architecture (NVLink network)
  https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/


AllReduce       → sync everything
Reduce-Scatter  → reduce + shard
AllGather       → collect shards
Broadcast       → one → all


all to all moe 

(single gpu case) 
suppose there are M tokens, as a tensor TOK = (M,dim) (this is the out of an attention head )
top_k one hots (M, top_k) showing which experts are selected for each token 
PCIe / NVLink Topology Labels — Practical Summary

Core idea:
These labels describe HOW data travels between GPUs.
More hops + worse components → lower bandwidth, higher latency, more contention.

--------------------------------------------------

NVLINK
- Path: GPU ↔ GPU (direct link)
- No PCIe, no CPU
- Bandwidth: VERY HIGH (~300 GB/s)
- Contention: minimal
- Interpretation: best possible connection

--------------------------------------------------

PIX (best PCIe case)
- Path: GPU → PCIe switch → GPU
- Single PCIe bridge
- No CPU involvement
- Bandwidth: high (~25–32 GB/s)
- Contention: shared switch
- Interpretation: “same switch, clean path”

--------------------------------------------------

PXB
- Path: GPU → PCIe switch → PCIe switch → GPU
- Multiple PCIe bridges
- No CPU
- Bandwidth: similar peak (~25–32 GB/s) but more overhead
- Contention: shared fabric (worse than PIX)
- Interpretation: “longer PCIe path”

--------------------------------------------------

PHB
- Path: GPU → PCIe → CPU (host bridge) → PCIe → GPU
- Goes through CPU root complex
- Bandwidth: lower (~15–28 GB/s)
- Contention: CPU + PCIe shared
- Interpretation: “CPU involved”

--------------------------------------------------

NODE
- Path: GPU → PCIe → CPU interconnect (within same socket) → PCIe → GPU
- Crosses internal CPU fabric
- Bandwidth: lower than PHB
- Contention: CPU internal interconnect
- Interpretation: “inside CPU, but still costly”

--------------------------------------------------

SYS (worst case)
- Path: GPU → PCIe → CPU0 → NUMA interconnect (QPI/UPI) → CPU1 → PCIe → GPU
- Crosses CPU sockets
- Bandwidth: lowest (~10–20 GB/s)
- Contention: heavy (NUMA + PCIe)
- Interpretation: “cross-socket = pain”

--------------------------------------------------

Ranking (best → worst):

NVLINK >>> PIX ≥ PXB > PHB > NODE > SYS

--------------------------------------------------

What matters for modeling:

1. NVLINK:
   - high bandwidth
   - dedicated edge

2. PCIe (PIX/PXB):
   - medium bandwidth
   - shared resource (IMPORTANT)

3. CPU / NUMA (PHB/NODE/SYS):
   - lower bandwidth
   - heavy contention

--------------------------------------------------

Key takeaway:

PCIe paths are NOT independent edges.
They are shared resources with contention.
NVLink is the only truly “clean” connection.


the LP forumulation 

y[f,p] in R is the scalar quantity, that 



List of problems (that we will tackle): 
1. GPU/CPU nodes need capacity. (VRAM/RAM) 
2. Shared Resource modelling, (independent edges with bandwidths are not sufficent) Understand hardware better.
3. Sharding thing, tensors are not water (not sure what to do about it) 


2. For each edge e: sum y[f,p] where e in p <= e.bandwidth * T

this one bothers me a bit, there may not be a positive flow throught the duration T

**if a pipe allows 5 liters per second, and the makespan was 2 seconds, and in this pipe, I sent 10 liters in 1 second, and did nothing for the next second, in reality the pipe would burst, but our model allows it **

You're right. The constraint assumes every flow runs for the full duration T. But a small flow might finish early, freeing up bandwidth on that edge for the remaining time.

This is the "pessimistic" assumption I mentioned earlier. The LP says "all flows are active for T seconds, sharing bandwidth the entire time." In reality, small flows finish early and the remaining flows speed up — like the water pipe example we discussed.

For a first pass this is fine because it gives you a lower bound on makespan that's actually achievable (the LP might slightly overestimate for asymmetric flow sizes, but it won't underestimate). The LP is still finding the optimal routing — it's just evaluating each routing slightly pessimistically.

To model the "flows finish at different times and bandwidth gets redistributed" behavior correctly, you'd need time-indexed variables or a simulation. That's a meaningful jump in complexity.

Worth it now, or save for later?

https://www.exxactcorp.com/blog/HPC/exploring-the-complexities-of-pcie-connectivity-and-peer-to-peer-communication