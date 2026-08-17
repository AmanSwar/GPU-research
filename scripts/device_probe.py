#!/usr/bin/env python3
"""Ground-truth device properties, straight from the driver.

Deliberately uses ctypes against libcuda.so rather than torch or the CUDA
runtime: the box's torch install is broken and the runtime is pinned per-project,
but libcuda is always there because the driver is. This script therefore runs
anywhere, which is the point -- it is the one source in this corpus that is
measured on the machine rather than read off a spec sheet.

    python3 scripts/device_probe.py [--json]
"""

import ctypes
import json
import sys

# CUdevice_attribute values from cuda.h. Kept as an explicit table so the
# numbers in the corpus can be traced back to a named driver attribute.
ATTRS = {
    "MAX_THREADS_PER_BLOCK": 1,
    "MAX_SHARED_MEMORY_PER_BLOCK": 8,
    "TOTAL_CONSTANT_MEMORY": 9,
    "WARP_SIZE": 10,
    "MAX_PITCH": 11,
    "MAX_REGISTERS_PER_BLOCK": 12,
    "CLOCK_RATE_KHZ": 13,
    "MULTIPROCESSOR_COUNT": 16,
    "INTEGRATED": 18,
    "CAN_MAP_HOST_MEMORY": 19,
    "MEMORY_CLOCK_RATE_KHZ": 36,
    "GLOBAL_MEMORY_BUS_WIDTH": 37,
    "L2_CACHE_SIZE": 38,
    "MAX_THREADS_PER_MULTIPROCESSOR": 39,
    "ASYNC_ENGINE_COUNT": 40,
    "UNIFIED_ADDRESSING": 41,
    "COMPUTE_CAPABILITY_MAJOR": 75,
    "COMPUTE_CAPABILITY_MINOR": 76,
    "GLOBAL_L1_CACHE_SUPPORTED": 79,
    "LOCAL_L1_CACHE_SUPPORTED": 80,
    "MAX_SHARED_MEMORY_PER_MULTIPROCESSOR": 81,
    "MAX_REGISTERS_PER_MULTIPROCESSOR": 82,
    "MANAGED_MEMORY": 83,
    "MULTI_GPU_BOARD": 84,
    "HOST_NATIVE_ATOMIC_SUPPORTED": 86,
    "PAGEABLE_MEMORY_ACCESS": 88,
    "CONCURRENT_MANAGED_ACCESS": 89,
    "COMPUTE_PREEMPTION_SUPPORTED": 90,
    "MAX_SHARED_MEMORY_PER_BLOCK_OPTIN": 97,
    "PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES": 98,
    "DIRECT_MANAGED_MEM_ACCESS_FROM_HOST": 99,
    "MAX_PERSISTING_L2_CACHE_SIZE": 108,
    "MAX_BLOCKS_PER_MULTIPROCESSOR": 106,
    "MAX_ACCESS_POLICY_WINDOW_SIZE": 109,
    "RESERVED_SHARED_MEMORY_PER_BLOCK": 111,
    "SPARSE_CUDA_ARRAY_SUPPORTED": 112,
    "MEMORY_POOLS_SUPPORTED": 115,
    "GPU_DIRECT_RDMA_SUPPORTED": 116,
    "CLUSTER_LAUNCH": 120,
    "CAN_USE_64_BIT_STREAM_MEM_OPS": 121,
    "DMA_BUF_SUPPORTED": 124,
    "MULTICAST_SUPPORTED": 132,
}

# Attributes worth reporting in bytes-with-units rather than raw integers.
BYTE_ATTRS = {
    "L2_CACHE_SIZE",
    "MAX_SHARED_MEMORY_PER_BLOCK",
    "MAX_SHARED_MEMORY_PER_MULTIPROCESSOR",
    "MAX_SHARED_MEMORY_PER_BLOCK_OPTIN",
    "MAX_PERSISTING_L2_CACHE_SIZE",
    "MAX_ACCESS_POLICY_WINDOW_SIZE",
    "RESERVED_SHARED_MEMORY_PER_BLOCK",
    "TOTAL_CONSTANT_MEMORY",
}


def human_bytes(n):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or unit == "GiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n / 1:.2f} {unit}"
        n /= 1024
    return str(n)


def main():
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError as e:
        sys.exit(f"cannot load libcuda.so.1: {e}")

    if cuda.cuInit(0) != 0:
        sys.exit("cuInit failed")

    count = ctypes.c_int()
    cuda.cuDeviceGetCount(ctypes.byref(count))

    version = ctypes.c_int()
    cuda.cuDriverGetVersion(ctypes.byref(version))

    devices = []
    for dev_index in range(count.value):
        dev = ctypes.c_int()
        cuda.cuDeviceGet(ctypes.byref(dev), dev_index)

        name = ctypes.create_string_buffer(256)
        cuda.cuDeviceGetName(name, 256, dev)

        total_mem = ctypes.c_size_t()
        cuda.cuDeviceTotalMem_v2(ctypes.byref(total_mem), dev)

        info = {
            "index": dev_index,
            "name": name.value.decode(),
            "total_memory_bytes": total_mem.value,
        }
        for attr_name, attr_id in ATTRS.items():
            val = ctypes.c_int()
            if cuda.cuDeviceGetAttribute(ctypes.byref(val), attr_id, dev) == 0:
                info[attr_name] = val.value
        devices.append(info)

    # Peer access matrix: which GPUs can address each other's memory directly,
    # and the driver's own ranking of the link between them.
    peer = []
    for i in range(count.value):
        di = ctypes.c_int()
        cuda.cuDeviceGet(ctypes.byref(di), i)
        row = []
        for j in range(count.value):
            if i == j:
                row.append("self")
                continue
            dj = ctypes.c_int()
            cuda.cuDeviceGet(ctypes.byref(dj), j)
            can = ctypes.c_int()
            # CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = 3
            cuda.cuDeviceGetP2PAttribute(ctypes.byref(can), 3, di, dj)
            perf = ctypes.c_int()
            # CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = 1
            cuda.cuDeviceGetP2PAttribute(ctypes.byref(perf), 1, di, dj)
            row.append(f"{'yes' if can.value else 'no'}(rank{perf.value})")
        peer.append(row)

    out = {"driver_version": version.value, "device_count": count.value,
           "devices": devices, "p2p": peer}

    if "--json" in sys.argv:
        print(json.dumps(out, indent=2))
        return

    d = devices[0]
    print(f"driver {version.value}, {count.value}x {d['name']}")
    print(f"compute capability sm_{d['COMPUTE_CAPABILITY_MAJOR']}{d['COMPUTE_CAPABILITY_MINOR']}")
    print(f"total memory        {d['total_memory_bytes'] / 2**30:.2f} GiB")
    print()
    for k in sorted(k for k in d if k not in ("index", "name", "total_memory_bytes")):
        v = d[k]
        shown = human_bytes(v) if k in BYTE_ATTRS else f"{v:,}"
        print(f"  {k:<50s} {shown}")
    print()
    print("p2p access matrix (row=src, col=dst):")
    for i, row in enumerate(peer):
        print(f"  gpu{i}: " + "  ".join(f"{c:>12s}" for c in row))

    # Flag any device whose properties differ from device 0 -- heterogeneity in
    # a supposedly uniform node is worth knowing about before it shows up as
    # unexplained rank skew.
    diffs = []
    for other in devices[1:]:
        for k, v in d.items():
            if k in ("index",):
                continue
            if other.get(k) != v:
                diffs.append(f"gpu{other['index']}.{k} = {other.get(k)} (gpu0: {v})")
    print()
    print("heterogeneity vs gpu0: " + ("none" if not diffs else ""))
    for line in diffs:
        print("  " + line)


if __name__ == "__main__":
    main()
