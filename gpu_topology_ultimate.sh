#!/bin/bash
################################################################################
# GPU TOPOLOGY & INTERCONNECT DISCOVERY - ULTIMATE EDITION
#
# Captures EVERYTHING queryable about NVIDIA GPU topology, NVLink, PCIe,
# NUMA affinity, bandwidth, error counters, fabric health, and more.
#
# Usage: sudo bash gpu_topology_ultimate.sh
# (root/sudo recommended for full data; runs degraded without it)
#
# Output: gpu_topology_<hostname>_<timestamp>.txt
#
# References:
#   - nvidia-smi docs:     https://docs.nvidia.com/deploy/nvidia-smi/index.html
#   - DCGM User Guide:     https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/
#   - Fabric Manager:       https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/
#   - NVBandwidth:          https://github.com/NVIDIA/nvbandwidth
#   - NCCL docs:            https://docs.nvidia.com/deeplearning/nccl/user-guide/
#   - NVLink wikipedia:     https://en.wikipedia.org/wiki/NVLink
#   - cuda-samples P2P:     https://github.com/NVIDIA/cuda-samples
################################################################################

set -euo pipefail

OUT="gpu_topology_$(hostname)_$(date +%Y%m%d_%H%M%S).txt"
DIVIDER_WIDTH=80

# Colors for terminal (not written to file)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

divider() {
    echo "" >> "$OUT"
    printf '=%.0s' $(seq 1 $DIVIDER_WIDTH) >> "$OUT"
    echo "" >> "$OUT"
    echo "  $1" >> "$OUT"
    printf '=%.0s' $(seq 1 $DIVIDER_WIDTH) >> "$OUT"
    echo -e "\n" >> "$OUT"
}

subdiv() {
    echo "" >> "$OUT"
    echo "--- $1 ---" >> "$OUT"
    echo "" >> "$OUT"
}

run_cmd() {
    # $1 = description, $2... = command
    local desc="$1"; shift
    subdiv "$desc"
    if eval "$@" >> "$OUT" 2>&1; then
        echo -e "  ${GREEN}✓${NC} $desc"
    else
        echo "[command failed or not available]" >> "$OUT"
        echo -e "  ${YELLOW}⚠${NC} $desc (skipped or failed)"
    fi
}

safe_cmd() {
    # Like run_cmd but completely silent on failure
    local desc="$1"; shift
    subdiv "$desc"
    eval "$@" >> "$OUT" 2>/dev/null || echo "[not available on this system]" >> "$OUT"
}

echo "=============================================="
echo " GPU TOPOLOGY ULTIMATE DISCOVERY"
echo " Output: $OUT"
echo "=============================================="
echo ""

# Check nvidia-smi exists
if ! command -v nvidia-smi &>/dev/null; then
    echo -e "${RED}ERROR: nvidia-smi not found. NVIDIA driver not installed.${NC}"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
echo -e "Detected ${GREEN}${GPU_COUNT}${NC} GPU(s). Collecting data...\n"

# Write header
{
    echo "GPU TOPOLOGY & INTERCONNECT DISCOVERY REPORT"
    echo "Generated: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "Hostname:  $(hostname)"
    echo "User:      $(whoami)"
    echo "GPUs:      $GPU_COUNT"
} >> "$OUT"


################################################################################
divider "1. SYSTEM INFORMATION"
################################################################################

run_cmd "Kernel & OS" \
    "uname -a"

run_cmd "OS Release" \
    "cat /etc/os-release 2>/dev/null || cat /etc/redhat-release 2>/dev/null"

run_cmd "CPU Model, Sockets, Cores, Threads, NUMA" \
    "lscpu | grep -E 'Model name|Socket|Core|Thread|NUMA|Architecture|CPU\(s\)|MHz|cache'"

run_cmd "IOMMU Status (important for GPU passthrough)" \
    "dmesg 2>/dev/null | grep -i iommu | head -20 || echo 'dmesg not accessible'"

run_cmd "Kernel Command Line (check iommu, pci settings)" \
    "cat /proc/cmdline"

run_cmd "Loaded NVIDIA Kernel Modules" \
    "lsmod | grep -i nvidia"

run_cmd "NVIDIA Driver Version & CUDA Version" \
    "nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 && echo 'CUDA:' && nvidia-smi --query-gpu=cuda_version --format=csv,noheader | head -1 2>/dev/null || true"


################################################################################
divider "2. GPU OVERVIEW (nvidia-smi default)"
################################################################################

run_cmd "nvidia-smi (standard output)" \
    "nvidia-smi"


################################################################################
divider "3. FULL GPU QUERY - EVERY QUERYABLE FIELD (CSV)"
################################################################################

# Core fields that are stable across driver versions
run_cmd "Core GPU Properties (CSV)" \
    "nvidia-smi --query-gpu=\
timestamp,\
driver_version,\
count,\
name,\
serial,\
uuid,\
pci.bus_id,\
pci.domain,\
pci.bus,\
pci.device,\
pci.device_id,\
pci.sub_device_id,\
pcie.link.gen.current,\
pcie.link.gen.max,\
pcie.link.width.current,\
pcie.link.width.max,\
index,\
display_active,\
persistence_mode,\
accounting.mode,\
vbios_version,\
inforom.image,\
inforom.oem,\
inforom.ecc,\
fan.speed,\
pstate,\
clocks_event_reasons.active,\
clocks_event_reasons.gpu_idle,\
clocks_event_reasons.sw_power_cap,\
clocks_event_reasons.hw_slowdown,\
clocks_event_reasons.hw_thermal_slowdown,\
clocks_event_reasons.hw_power_brake_slowdown,\
clocks_event_reasons.sw_thermal_slowdown,\
memory.total,\
memory.reserved,\
memory.used,\
memory.free,\
compute_mode,\
compute_cap,\
utilization.gpu,\
utilization.memory,\
ecc.mode.current,\
ecc.mode.pending,\
temperature.gpu,\
temperature.memory,\
power.draw.instant,\
power.limit,\
power.default_limit,\
power.min_limit,\
power.max_limit,\
clocks.current.graphics,\
clocks.current.sm,\
clocks.current.memory,\
clocks.current.video,\
clocks.applications.graphics,\
clocks.applications.memory,\
clocks.max.graphics,\
clocks.max.sm,\
clocks.max.memory,\
mig.mode.current,\
mig.mode.pending \
--format=csv 2>&1 || echo 'Some fields not supported on this driver version'"

# Dynamically discover ALL valid --query-gpu fields and query each one
# This ensures we never miss a field regardless of driver version
run_cmd "Auto-discovered valid fields (driver-version-safe)" \
    "ALL_FIELDS=\$(nvidia-smi --help-query-gpu 2>&1 | grep -oP '\"[a-z][a-z0-9_.]+\"' | tr -d '\"' | sort -u)
    VALID_FIELDS=''
    for f in \$ALL_FIELDS; do
        if nvidia-smi --query-gpu=\$f --format=csv,noheader -i 0 &>/dev/null; then
            if [ -z \"\$VALID_FIELDS\" ]; then
                VALID_FIELDS=\"\$f\"
            else
                VALID_FIELDS=\"\$VALID_FIELDS,\$f\"
            fi
        fi
    done
    echo \"Valid queryable fields on this driver (\$(echo \$VALID_FIELDS | tr ',' '\n' | wc -l) fields):\"
    echo \"\$VALID_FIELDS\" | tr ',' '\n' | sort
    echo ''
    echo '--- Full CSV dump of all valid fields ---'
    nvidia-smi --query-gpu=\$VALID_FIELDS --format=csv 2>&1"


################################################################################
divider "4. DETAILED GPU QUERY SECTIONS (nvidia-smi -q -d)"
################################################################################

# All documented -d display filters
for section in MEMORY UTILIZATION ECC TEMPERATURE POWER CLOCK COMPUTE PIDS \
               PERFORMANCE SUPPORTED_CLOCKS PAGE_RETIREMENT ACCOUNTING \
               ENCODER_STATS SUPPORTED_GPU_TARGET_TEMP VOLTAGE FBC_STATS \
               ROW_REMAPPER GSP_FIRMWARE_VERSION POWER_SMOOTHING POWER_PROFILES; do
    run_cmd "Query Section: $section" \
        "nvidia-smi -q -d $section 2>&1 || true"
done


################################################################################
divider "5. FULL XML DUMP (for machine parsing)"
################################################################################

run_cmd "Full XML Query (truncated to key sections)" \
    "nvidia-smi -q -x 2>/dev/null | head -500 || echo 'XML output not available'"


################################################################################
divider "6. TOPOLOGY MATRIX & P2P CONNECTIVITY"
################################################################################

run_cmd "Topology Matrix (nvidia-smi topo -m)" \
    "nvidia-smi topo -m"

run_cmd "Topology Matrix - P2P Access (nvidia-smi topo -mp)" \
    "nvidia-smi topo -mp"

subdiv "Topology Legend"
cat >> "$OUT" << 'LEGEND'
  X   = Self
  SYS = Traverses PCIe + inter-NUMA SMP interconnect (QPI/UPI)
  NODE= Traverses PCIe + interconnect between PCIe Host Bridges within a NUMA node
  PHB = Traverses PCIe + a PCIe Host Bridge (typically the CPU)
  PXB = Traverses multiple PCIe bridges (no Host Bridge crossing)
  PIX = Traverses at most a single PCIe bridge
  NV# = Traverses a bonded set of # NVLinks
LEGEND
echo -e "  ${GREEN}✓${NC} Topology Legend"


################################################################################
divider "7. NVLINK - STATUS, CAPABILITIES, BANDWIDTH PER LINK"
################################################################################

for i in $(seq 0 $((GPU_COUNT-1))); do
    subdiv "GPU $i - NVLink Status (active/inactive per link)"
    nvidia-smi nvlink --status -i "$i" >> "$OUT" 2>&1 || echo "  [not supported]" >> "$OUT"

    subdiv "GPU $i - NVLink Capabilities (P2P, atomics, SLI per link)"
    nvidia-smi nvlink --capabilities -i "$i" >> "$OUT" 2>&1 || echo "  [not supported]" >> "$OUT"

    # Remote device info for each link (which GPU is on the other end)
    subdiv "GPU $i - NVLink Remote Device Info"
    nvidia-smi nvlink -i "$i" -p 2>&1 >> "$OUT" || echo "  [not supported]" >> "$OUT"

    echo -e "  ${GREEN}✓${NC} GPU $i NVLink status/capabilities"
done


################################################################################
divider "8. NVLINK - UTILIZATION COUNTERS (THROUGHPUT)"
################################################################################

subdiv "NVLink Data Throughput Counters (Rx/Tx per link)"
echo "These show cumulative bytes transferred since driver load." >> "$OUT"
echo "Compare two snapshots to get throughput rate." >> "$OUT"
echo "" >> "$OUT"

subdiv "Available nvidia-smi nvlink options on this driver"
nvidia-smi nvlink -h >> "$OUT" 2>&1 || echo "  [nvlink help not available]" >> "$OUT"
echo "" >> "$OUT"

for i in $(seq 0 $((GPU_COUNT-1))); do
    subdiv "GPU $i - NVLink Throughput Counters (Counter Set 0)"
    # Try new-style flag first (--getcounters 0), fall back to old-style (-g 0)
    nvidia-smi nvlink --getcounters 0 -i "$i" >> "$OUT" 2>&1 || \
    nvidia-smi nvlink -g 0 -i "$i" >> "$OUT" 2>&1 || \
    echo "  [throughput counters not supported on this driver]" >> "$OUT"

    subdiv "GPU $i - NVLink Throughput Counters (Counter Set 1)"
    nvidia-smi nvlink --getcounters 1 -i "$i" >> "$OUT" 2>&1 || \
    nvidia-smi nvlink -g 1 -i "$i" >> "$OUT" 2>&1 || \
    echo "  [not supported]" >> "$OUT"

    subdiv "GPU $i - NVLink Counter Control Info (Counter 0)"
    nvidia-smi nvlink --getcontrol 0 -i "$i" >> "$OUT" 2>&1 || \
    nvidia-smi nvlink -gc 0 -i "$i" >> "$OUT" 2>&1 || \
    echo "  [not supported]" >> "$OUT"

    subdiv "GPU $i - NVLink Counter Control Info (Counter 1)"
    nvidia-smi nvlink --getcontrol 1 -i "$i" >> "$OUT" 2>&1 || \
    nvidia-smi nvlink -gc 1 -i "$i" >> "$OUT" 2>&1 || \
    echo "  [not supported]" >> "$OUT"

    # Also try the throughput subcommand (available on some driver versions)
    subdiv "GPU $i - NVLink Throughput (--throughput)"
    nvidia-smi nvlink --throughput -i "$i" >> "$OUT" 2>&1 || \
    echo "  [--throughput not available on this driver]" >> "$OUT"

    echo -e "  ${GREEN}✓${NC} GPU $i NVLink throughput counters"
done

subdiv "NVLink Counter Encoding Reference"
cat >> "$OUT" << 'COUNTERREF'
  Counter set string format: <set><mode><traffic_types>
    Set:   0 = counter 0, 1 = counter 1
    Mode:  c = count cycles, p = count packets, b = count bytes
    Traffic types (combinable):
      n = nop, r = read, w = write
      x = reduction atomic requests
      y = non-reduction atomic requests
      f = flush, d = responses with data
      o = responses with no data, z = all traffic

  Example: To set counter 0 to count bytes for reads+writes:
    nvidia-smi nvlink -sc 0brw -i <gpu>
  Then read with:
    nvidia-smi nvlink -g 0 -i <gpu>
COUNTERREF


################################################################################
divider "9. NVLINK - ERROR COUNTERS & CRC ERRORS"
################################################################################

for i in $(seq 0 $((GPU_COUNT-1))); do
    subdiv "GPU $i - NVLink Error Counters (replay, recovery, CRC, etc)"
    nvidia-smi nvlink --errorcounters -i "$i" >> "$OUT" 2>&1 || echo "  [not supported]" >> "$OUT"

    subdiv "GPU $i - NVLink Per-Lane CRC Error Counters"
    nvidia-smi nvlink -ec -i "$i" >> "$OUT" 2>&1 || echo "  [not supported]" >> "$OUT"

    echo -e "  ${GREEN}✓${NC} GPU $i NVLink error counters"
done

subdiv "NVLink Error Types Reference"
cat >> "$OUT" << 'ERRREF'
  CRC FLIT Error:   Data link receive flow control digit CRC error
  CRC Data Error:   Data link receive data CRC error
  Replay Error:     Transmit replay error (packet retransmission)
  Recovery Error:   Transmit recovery error (link recovery events)

  Non-zero replay/recovery errors may indicate cable/connection issues.
  Occasional CRC errors can be normal; sustained errors indicate problems.
ERRREF


################################################################################
divider "10. NVLINK - FABRIC STATUS (NVSwitch systems)"
################################################################################

run_cmd "GPU Fabric Status (Healthy/Unhealthy/Limited for NVSwitch)" \
    "nvidia-smi -q 2>/dev/null | grep -A 20 'GPU Fabric' || echo 'No GPU Fabric info (not an NVSwitch system or unsupported)'"

run_cmd "NVSwitch devices visible via nvidia-smi" \
    "nvidia-smi nvswitch -ls 2>&1 || echo 'No NVSwitch CLI support (nvidia-smi nvswitch not available)'"

run_cmd "Fabric Manager Status" \
    "systemctl status nvidia-fabricmanager 2>/dev/null || echo 'Fabric Manager not installed or not a systemd system'"

run_cmd "Fabric Manager Logs (last 30 lines)" \
    "journalctl -u nvidia-fabricmanager --no-pager -n 30 2>/dev/null || echo 'No fabric manager logs found'"


################################################################################
divider "11. PCIe DETAILS"
################################################################################

run_cmd "PCIe Topology Tree (GPU devices)" \
    "lspci -tv 2>/dev/null | grep -B2 -A2 -E 'VGA|3D|Display|NVIDIA' || echo 'lspci not available'"

run_cmd "PCIe Detailed Info per GPU" \
    "lspci -vvv 2>/dev/null | grep -A 40 'NVIDIA' | head -200 || echo 'lspci -vvv not available (try with sudo)'"

run_cmd "PCIe Link Speed & Width per GPU (lspci)" \
    "for dev in \$(lspci 2>/dev/null | grep -i nvidia | awk '{print \$1}'); do echo \"Device \$dev:\"; lspci -s \"\$dev\" -vv 2>/dev/null | grep -E 'LnkCap|LnkSta|LnkCtl|MaxPayload|MaxReadReq|ASPM|Width|Speed' || true; echo; done"

run_cmd "PCIe ACS (Access Control Services) - affects P2P" \
    "for dev in \$(lspci 2>/dev/null | grep -i nvidia | awk '{print \$1}'); do echo \"Device \$dev:\"; setpci -s \"\$dev\" ECAP_ACS+6.w 2>/dev/null && echo '(non-zero = ACS enabled, may block P2P)' || echo 'Cannot read ACS'; done"

run_cmd "PCIe AER (Advanced Error Reporting)" \
    "for dev in \$(lspci 2>/dev/null | grep -i nvidia | awk '{print \$1}'); do echo \"Device \$dev:\"; lspci -s \"\$dev\" -vv 2>/dev/null | grep -A 15 'Advanced Error' || true; echo; done"

run_cmd "nvidia-smi dmon PCIe Tx/Rx Throughput (1 sample)" \
    "nvidia-smi dmon -c 1 -s t 2>&1 || echo 'dmon PCIe throughput not supported'"


################################################################################
divider "12. NUMA TOPOLOGY & CPU AFFINITY"
################################################################################

run_cmd "NUMA Hardware" \
    "numactl --hardware 2>/dev/null || echo 'numactl not installed'"

run_cmd "NUMA CPU Lists per Node" \
    "for f in /sys/devices/system/node/node*/cpulist; do echo \"\$f: \$(cat \$f 2>/dev/null)\"; done"

run_cmd "NUMA Memory Info per Node" \
    "cat /proc/buddyinfo 2>/dev/null || true; echo; numastat 2>/dev/null || true"

run_cmd "GPU to NUMA Affinity (from sysfs)" \
    "for gpu_dir in /sys/bus/pci/devices/*/; do
        if [ -f \"\${gpu_dir}class\" ] && grep -q '0x030' \"\${gpu_dir}class\" 2>/dev/null; then
            dev=\$(basename \$gpu_dir)
            numa=\$(cat \${gpu_dir}numa_node 2>/dev/null || echo 'N/A')
            local_cpus=\$(cat \${gpu_dir}local_cpulist 2>/dev/null || echo 'N/A')
            echo \"PCI \$dev => NUMA node: \$numa, local CPUs: \$local_cpus\"
        fi
    done"

run_cmd "GPU NUMA node from nvidia-smi topology" \
    "nvidia-smi topo -m 2>/dev/null | grep -E 'GPU|NUMA|CPU' || true"


################################################################################
divider "13. DCGM (Data Center GPU Manager) TOPOLOGY"
################################################################################

run_cmd "DCGM Discovery - GPUs and NVSwitches" \
    "dcgmi discovery -l 2>&1 || echo 'DCGM (dcgmi) not installed. Install: https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/getting-started.html'"

run_cmd "DCGM Topology" \
    "dcgmi topo --gpuid 0 2>&1 || echo 'dcgmi topo not available'"

run_cmd "DCGM NVLink Status" \
    "dcgmi nvlink --status 2>&1 || echo 'dcgmi nvlink not available'"

run_cmd "DCGM Health Check (non-invasive)" \
    "dcgmi health -c 2>&1 || echo 'dcgmi health not available'"

run_cmd "DCGM Diagnostics Level 1 (quick)" \
    "dcgmi diag -r 1 2>&1 || echo 'dcgmi diag not available'"

run_cmd "DCGM NVLink Error Counters" \
    "dcgmi nvlink -e 2>&1 || echo 'dcgmi nvlink errors not available'"

run_cmd "DCGM Field Values - NVLink Throughput" \
    "dcgmi dmon -e 700,701,702,703,704,705 -c 1 2>&1 || echo 'dcgmi dmon not available (fields 700-705 = NVLink Tx/Rx)'"


################################################################################
divider "14. PROCESS MONITORING (who's using the GPUs)"
################################################################################

run_cmd "nvidia-smi pmon (1 sample)" \
    "nvidia-smi pmon -c 1 2>&1 || echo 'pmon not supported'"

run_cmd "Compute processes per GPU" \
    "nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv 2>&1 || echo 'No compute processes'"

run_cmd "GPU Accounting Data" \
    "nvidia-smi --query-accounted-apps=pid,gpu_name,gpu_bus_id,time,max_memory_usage --format=csv 2>&1 || echo 'Accounting not enabled or no data'"


################################################################################
divider "15. MIG (Multi-Instance GPU) CONFIGURATION"
################################################################################

run_cmd "MIG Mode Status" \
    "nvidia-smi mig -lgi 2>&1 || echo 'MIG not supported or not enabled'"

run_cmd "MIG GPU Instance Profiles" \
    "nvidia-smi mig -lgip 2>&1 || echo 'MIG profiles not available'"

run_cmd "MIG Compute Instance Profiles" \
    "nvidia-smi mig -lcip 2>&1 || echo 'MIG CI profiles not available'"


################################################################################
divider "16. DEVICE MONITORING SNAPSHOT"
################################################################################

run_cmd "Device Monitor - All Metrics (3 samples)" \
    "nvidia-smi dmon -c 3 -s pucvmet 2>&1 || echo 'dmon not fully supported'"

subdiv "dmon Metric Keys"
cat >> "$OUT" << 'DMONREF'
  p = Power Usage and Temperature
  u = Utilization (SM, Memory, Encoder, Decoder)
  c = Proc and Memory Clocks
  v = Power and Thermal Violations
  m = FB and Bar1 Memory
  e = ECC Errors and PCIe Replay errors
  t = PCIe Rx and Tx Throughput
DMONREF


################################################################################
divider "17. P2P ACCESS MATRIX (PyTorch)"
################################################################################

run_cmd "PyTorch P2P Access & Device Properties" \
    "python3 - << 'PYEOF'
import sys
try:
    import torch
    n = torch.cuda.device_count()
    print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
    print(f'PyTorch sees {n} GPU(s)')
    print(f'CUDA version: {torch.version.cuda}')
    print(f'cuDNN version: {torch.backends.cudnn.version()}')
    print()

    # Device properties
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name}')
        print(f'  Compute Capability: {props.major}.{props.minor}')
        print(f'  Total Memory: {props.total_mem / 1024**3:.1f} GB')
        print(f'  SM Count: {props.multi_processor_count}')
        print(f'  Max Threads/SM: {props.max_threads_per_multi_processor}')
        print()

    # P2P Matrix
    if n > 1:
        print('P2P Access Matrix (1=enabled, 0=disabled):')
        header = '     ' + '  '.join([f'GPU{j}' for j in range(n)])
        print(header)
        for i in range(n):
            row = f'GPU{i} '
            for j in range(n):
                if i == j:
                    row += '  -  '
                else:
                    can = torch.cuda.can_device_access_peer(i, j)
                    row += f\"  {'1' if can else '0'}  \"
            print(row)
    else:
        print('Only 1 GPU - P2P matrix not applicable')

except ImportError:
    print('PyTorch not installed - skipping')
except Exception as e:
    print(f'Error: {e}')
PYEOF"


################################################################################
divider "18. P2P ACCESS MATRIX (CUDA/ctypes - no PyTorch needed)"
################################################################################

run_cmd "CUDA Runtime P2P Check (via ctypes)" \
    "python3 - << 'CUDAEOF'
import ctypes, sys

# Try to load CUDA runtime
for libname in ['libcudart.so', 'libcudart.so.12', 'libcudart.so.11.0']:
    try:
        cuda = ctypes.CDLL(libname)
        break
    except OSError:
        cuda = None

if cuda is None:
    print('Cannot load libcudart.so - CUDA runtime not found')
    sys.exit(0)

count = ctypes.c_int()
cuda.cudaGetDeviceCount(ctypes.byref(count))
n = count.value
print(f'CUDA Runtime sees {n} device(s)')
print()

if n > 1:
    can_access = ctypes.c_int()
    print('P2P CanAccessPeer Matrix:')
    header = '     ' + '  '.join([f'GPU{j}' for j in range(n)])
    print(header)
    for i in range(n):
        row = f'GPU{i} '
        for j in range(n):
            if i == j:
                row += '  -  '
            else:
                cuda.cudaDeviceCanAccessPeer(ctypes.byref(can_access), i, j)
                row += f'  {can_access.value}  '
        print(row)
CUDAEOF"


################################################################################
divider "19. NCCL & COMMUNICATION ENVIRONMENT"
################################################################################

run_cmd "NCCL, CUDA, GPU Environment Variables" \
    "env | grep -E 'NCCL|CUDA|GPU|NVIDIA|OMPI|MPI|UCX|HCOLL|SHARP|RDMA|IB_' | sort || echo 'No relevant env vars set'"

run_cmd "NCCL Library Version" \
    "python3 -c 'import torch; print(\"NCCL version:\", torch.cuda.nccl.version())' 2>/dev/null || \
     ldconfig -p 2>/dev/null | grep nccl || \
     echo 'NCCL not found'"

run_cmd "NCCL Debug Tip" \
    "echo 'To enable NCCL debug logging, set:
  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=ALL
Key NCCL topology env vars:
  NCCL_TOPO_FILE          - Custom topology XML
  NCCL_TOPO_DUMP_FILE     - Dump detected topology to file
  NCCL_P2P_LEVEL          - P2P transport level (LOC/NVL/PIX/PXB/PHB/SYS)
  NCCL_P2P_DISABLE        - Disable P2P (0/1)
  NCCL_SHM_DISABLE        - Disable shared memory (0/1)
  NCCL_NET_GDR_LEVEL      - GPUDirect RDMA level
  NCCL_CROSS_NIC          - Cross-NIC communication (0/1/2)
  NCCL_IB_DISABLE         - Disable InfiniBand (0/1)
  NCCL_SOCKET_IFNAME      - Network interface for socket transport
  NCCL_NVLS_ENABLE        - NVLink SHARP (Hopper+) (0/1)
  NCCL_ALGO               - Algorithm selection (Ring/Tree/CollnetDirect/CollnetChain/NVLS/NVLSTree)
  NCCL_PROTO              - Protocol selection (LL/LL128/Simple)'"


################################################################################
divider "20. INFINIBAND / RoCE / GPUDirect RDMA"
################################################################################

run_cmd "InfiniBand Devices" \
    "ibstat 2>/dev/null || echo 'ibstat not available (no IB stack)'"

run_cmd "IB Device List" \
    "ibv_devinfo 2>/dev/null | head -60 || echo 'ibv_devinfo not available'"

run_cmd "RDMA Devices (rdma tool)" \
    "rdma link show 2>/dev/null || echo 'rdma tool not available'"

run_cmd "GPUDirect RDMA - nvidia_peermem module" \
    "lsmod | grep -i peermem 2>/dev/null || echo 'nvidia_peermem not loaded (GPUDirect RDMA not active)'"

run_cmd "GPUDirect Storage support" \
    "lsmod | grep -i gds 2>/dev/null; ls /dev/nvidia-fs* 2>/dev/null || echo 'GPUDirect Storage not detected'"


################################################################################
divider "21. SYSFS GPU RAW DATA"
################################################################################

run_cmd "GPU sysfs entries (power, link speed, NUMA)" \
    "for gpu_dir in /sys/bus/pci/devices/*/; do
        if [ -f \"\${gpu_dir}class\" ] && grep -q '0x030' \"\${gpu_dir}class\" 2>/dev/null; then
            dev=\$(basename \$gpu_dir)
            echo \"=== PCI Device: \$dev ===\"
            echo \"  Class:          \$(cat \${gpu_dir}class 2>/dev/null)\"
            echo \"  Vendor:         \$(cat \${gpu_dir}vendor 2>/dev/null)\"
            echo \"  Device:         \$(cat \${gpu_dir}device 2>/dev/null)\"
            echo \"  NUMA Node:      \$(cat \${gpu_dir}numa_node 2>/dev/null)\"
            echo \"  Local CPUs:     \$(cat \${gpu_dir}local_cpulist 2>/dev/null)\"
            echo \"  Current Link Speed: \$(cat \${gpu_dir}current_link_speed 2>/dev/null)\"
            echo \"  Current Link Width: \$(cat \${gpu_dir}current_link_width 2>/dev/null)\"
            echo \"  Max Link Speed: \$(cat \${gpu_dir}max_link_speed 2>/dev/null)\"
            echo \"  Max Link Width: \$(cat \${gpu_dir}max_link_width 2>/dev/null)\"
            echo \"  D3cold Allowed: \$(cat \${gpu_dir}d3cold_allowed 2>/dev/null)\"
            echo \"  Enable:         \$(cat \${gpu_dir}enable 2>/dev/null)\"
            echo \"  IRQ:            \$(cat \${gpu_dir}irq 2>/dev/null)\"
            echo \"  Driver:         \$(readlink -f \${gpu_dir}driver 2>/dev/null | xargs basename 2>/dev/null)\"
            echo \"  IOMMU Group:    \$(readlink -f \${gpu_dir}iommu_group 2>/dev/null | xargs basename 2>/dev/null)\"
            # Power management from sysfs
            if [ -d \"\${gpu_dir}power\" ]; then
                echo \"  Runtime Status: \$(cat \${gpu_dir}power/runtime_status 2>/dev/null)\"
                echo \"  Control:        \$(cat \${gpu_dir}power/control 2>/dev/null)\"
            fi
            echo
        fi
    done"


################################################################################
divider "22. NVBandwidth / cuda-samples P2P BANDWIDTH TEST"
################################################################################

run_cmd "NVBandwidth (if installed)" \
    "which nvbandwidth &>/dev/null && nvbandwidth --help 2>&1 | head -5 && echo '' && \
     echo 'Available test types:' && nvbandwidth --list 2>&1 || \
     echo 'nvbandwidth not installed.
Install from: https://github.com/NVIDIA/nvbandwidth
Build:
  git clone https://github.com/NVIDIA/nvbandwidth.git
  cd nvbandwidth && cmake -B build && cmake --build build
  ./build/nvbandwidth
Key tests:
  nvbandwidth                                   # Run all tests
  nvbandwidth -t device_to_device_memcpy_read_ce  # D2D copy engine bandwidth
  nvbandwidth -t device_to_device_memcpy_read_sm  # D2D SM bandwidth
  nvbandwidth -t host_to_device_memcpy_ce         # H2D copy engine
  nvbandwidth -b 1024 -i 10                       # 1GB buffer, 10 iterations
  nvbandwidth -j                                  # JSON output'"

run_cmd "CUDA p2pBandwidthLatencyTest (if compiled)" \
    "for path in /usr/local/cuda/samples/Samples/5_Domain_Specific/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest \
                 /usr/local/cuda-samples/Samples/5_Domain_Specific/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest \
                 ./p2pBandwidthLatencyTest; do
        if [ -x \"\$path\" ]; then
            echo \"Found at: \$path\"
            \"\$path\" 2>&1
            break
        fi
    done || echo 'p2pBandwidthLatencyTest not found.
Build from: https://github.com/NVIDIA/cuda-samples
  git clone https://github.com/NVIDIA/cuda-samples.git
  cd cuda-samples/Samples/5_Domain_Specific/p2pBandwidthLatencyTest
  make
  ./p2pBandwidthLatencyTest'"


################################################################################
divider "23. NVIDIA TOPO FILE DUMP (for NCCL)"
################################################################################

run_cmd "Dump NCCL topology detection" \
    "echo 'To capture NCCL detected topology at runtime:
  export NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo.xml
  <run your training script>
  cat /tmp/nccl_topo.xml

NCCL Topology XML contains:
  - GPU PCI paths and NVLink connections
  - NVSwitch configuration
  - Network interfaces and their NUMA/PCI affinity
  - CPU topology and thread mapping'"


################################################################################
divider "24. CLOCK & POWER CONFIGURATION STATE"
################################################################################

run_cmd "Supported Clock Frequencies" \
    "nvidia-smi -q -d SUPPORTED_CLOCKS 2>&1 | head -100 || echo 'Not supported'"

run_cmd "Application Clock Settings" \
    "nvidia-smi --query-gpu=index,clocks.applications.graphics,clocks.applications.memory,clocks.default_applications.graphics,clocks.default_applications.memory --format=csv 2>&1"

run_cmd "Power Limits" \
    "nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,power.min_limit,power.max_limit,power.draw --format=csv 2>&1"

run_cmd "GPU Lock Status (clocks)" \
    "nvidia-smi --query-gpu=index,clocks.current.graphics,clocks.max.graphics,clocks.current.memory,clocks.max.memory --format=csv 2>&1"


################################################################################
divider "25. ROW REMAPPER & RETIRED PAGES (memory health)"
################################################################################

run_cmd "Row Remapper Status" \
    "nvidia-smi -q -d ROW_REMAPPER 2>&1 || echo 'Not supported'"

run_cmd "Retired Pages" \
    "nvidia-smi --query-gpu=index,retired_pages.single_bit_ecc.count,retired_pages.double_bit.count,retired_pages.pending --format=csv 2>&1 || echo 'Not supported'"

run_cmd "Remapped Rows (from nvidia-smi query)" \
    "nvidia-smi --query-remapped-rows=gpu_bus_id,gpu_uuid,remapped_rows.correctable,remapped_rows.uncorrectable,remapped_rows.pending,remapped_rows.failure --format=csv 2>&1 || echo 'Not supported'"


################################################################################
divider "26. VGPU & VIRTUALIZATION"
################################################################################

run_cmd "GPU Virtualization Mode" \
    "nvidia-smi --query-gpu=index,gpu_operation_mode.current,gsp_firmware_version --format=csv 2>&1 || true"

run_cmd "vGPU Info" \
    "nvidia-smi vgpu 2>&1 || echo 'vGPU not configured'"


################################################################################
divider "27. SOFTWARE VERSIONS SUMMARY"
################################################################################

run_cmd "All NVIDIA Software Versions" \
    "{
    echo 'nvidia-smi version:'; nvidia-smi --version 2>/dev/null || nvidia-smi | head -1
    echo ''
    echo 'CUDA toolkit:'; nvcc --version 2>/dev/null || echo 'nvcc not in PATH'
    echo ''
    echo 'cuDNN:'; find / -name 'libcudnn*' -type f 2>/dev/null | head -5 || echo 'cuDNN not found'
    echo ''; dpkg -l 2>/dev/null | grep -i cudnn | head -5 || true
    echo ''
    echo 'NCCL:'; dpkg -l 2>/dev/null | grep -i nccl || rpm -qa 2>/dev/null | grep -i nccl || echo 'NCCL package not found'
    echo ''
    echo 'Fabric Manager:'; dpkg -l 2>/dev/null | grep -i fabricmanager || rpm -qa 2>/dev/null | grep -i fabricmanager || echo 'FM not installed'
    echo ''
    echo 'DCGM:'; dcgmi --version 2>/dev/null || echo 'DCGM not installed'
    echo ''
    echo 'GDRCopy:'; gdrcopy_sanity 2>/dev/null && echo 'GDRCopy OK' || echo 'GDRCopy not installed'
}"


################################################################################
divider "28. QUICK HEALTH SUMMARY"
################################################################################

subdiv "Automated Health Checks"
{
    echo "=== Quick Health Assessment ==="
    echo ""

    # Check for ECC errors
    echo "ECC Errors (uncorrected aggregate):"
    nvidia-smi --query-gpu=index,name,ecc.errors.uncorrected.aggregate.total --format=csv,noheader 2>/dev/null | while read line; do
        echo "  $line"
    done
    echo ""

    # Check for retired pages
    echo "Retired Pages:"
    nvidia-smi --query-gpu=index,retired_pages.double_bit.count,retired_pages.pending --format=csv,noheader 2>/dev/null | while read line; do
        echo "  $line"
    done
    echo ""

    # Check for throttling
    echo "Throttle Reasons (active):"
    nvidia-smi --query-gpu=index,clocks_event_reasons.hw_slowdown,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_power_cap --format=csv,noheader 2>/dev/null | while read line; do
        echo "  $line"
    done
    echo ""

    # Check PCIe gen
    echo "PCIe Generation (current vs max):"
    nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv,noheader 2>/dev/null | while read line; do
        echo "  $line"
    done
    echo ""

    # Check temperatures
    echo "Temperatures:"
    nvidia-smi --query-gpu=index,temperature.gpu,temperature.memory --format=csv,noheader 2>/dev/null | while read line; do
        echo "  $line"
    done
    echo ""

    # NVLink error check
    echo "NVLink Errors (non-zero = investigate):"
    for i in $(seq 0 $((GPU_COUNT-1))); do
        errors=$(nvidia-smi nvlink -e -i "$i" 2>/dev/null | grep -v "^$" | grep -v "GPU\|Link\|^$" | awk '{sum += $NF} END {print sum+0}')
        echo "  GPU $i: total NVLink errors = $errors"
    done

} >> "$OUT" 2>&1
echo -e "  ${GREEN}✓${NC} Health summary"


################################################################################
divider "END OF REPORT"
################################################################################

{
    echo "Report complete: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "Output file: $OUT"
    echo ""
    echo "=== ADDITIONAL TOOLS TO EXPLORE ==="
    echo ""
    echo "NVBandwidth (real bandwidth measurement):"
    echo "  https://github.com/NVIDIA/nvbandwidth"
    echo ""
    echo "CUDA Samples P2P tests:"
    echo "  https://github.com/NVIDIA/cuda-samples"
    echo ""
    echo "DCGM (datacenter GPU management):"
    echo "  https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/"
    echo ""
    echo "NCCL Tests (collective communication benchmarks):"
    echo "  https://github.com/NVIDIA/nccl-tests"
    echo "  Run: ./build/all_reduce_perf -b 8 -e 256M -f 2 -g <num_gpus>"
    echo ""
    echo "Fabric Manager (NVSwitch systems):"
    echo "  https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/"
    echo ""
    echo "nvidia-smi full reference:"
    echo "  https://docs.nvidia.com/deploy/nvidia-smi/index.html"
    echo "  nvidia-smi --help-query-gpu  (all queryable fields)"
    echo ""
    echo "NCCL topology debugging:"
    echo "  export NCCL_DEBUG=INFO NCCL_TOPO_DUMP_FILE=/tmp/topo.xml"
} >> "$OUT"

echo ""
echo "=============================================="
echo -e " ${GREEN}DONE!${NC} Report written to: ${GREEN}${OUT}${NC}"
LINES=$(wc -l < "$OUT")
SIZE=$(du -h "$OUT" | cut -f1)
echo " Size: $SIZE ($LINES lines)"
echo "=============================================="