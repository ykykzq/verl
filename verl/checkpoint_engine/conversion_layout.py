# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pointer-free FSDP export metadata and deterministic snapshot bucket planning."""

import math

import torch


def trainer_source_layout(engine, dtype):
    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()
    local_layout, aliases = {}, {}
    for name, param in engine.module.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(param), []).append(name)
        mesh = getattr(param, "device_mesh", None)
        local = param.to_local() if hasattr(param, "to_local") else param
        local_layout[name] = {
            "rank": rank,
            "shape": list(param.shape),
            "local_shape": list(local.shape),
            "dtype": str(param.dtype),
            "placements": [str(p) for p in getattr(param, "placements", ())],
            "mesh": mesh.mesh.tolist() if mesh is not None else None,
            "coordinate": mesh.get_coordinate() if mesh is not None else None,
        }
    shared = {name: sorted(names) for names in aliases.values() for name in names}
    generator, peft = engine.get_per_tensor_param()
    if peft is not None:
        raise ValueError("conversion plans require merged full-parameter export")
    parameters = []
    for name, tensor in generator:
        if name not in local_layout:
            raise ValueError(f"source {name} needs an explicit FSDP-to-HF layout mapping")
        parameters.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(dtype if tensor.is_floating_point() else tensor.dtype),
                "aliases": shared[name],
                "fsdp_shard": local_layout[name],
            }
        )
    # No tensor or DTensor address is returned or retained by the plan.
    torch.cuda.synchronize()
    return {
        "rank": rank,
        "world_size": world,
        "export": "full_tensor_all_gather_on_all_trainer_ranks",
        "parameters": parameters,
    }


def merge_source_layouts(layouts):
    layouts = sorted(layouts, key=lambda x: x["rank"])
    if not layouts or [v["rank"] for v in layouts] != list(range(len(layouts))):
        raise ValueError("source layouts must cover each trainer rank exactly once")
    if any(v["world_size"] != len(layouts) for v in layouts):
        raise ValueError("trainer world-size mismatch")
    sources = []
    for index, value in enumerate(layouts[0]["parameters"]):
        description = {k: v for k, v in value.items() if k != "fsdp_shard"}
        shards = []
        for layout in layouts:
            if len(layout["parameters"]) != len(layouts[0]["parameters"]):
                raise ValueError("source parameter counts differ between ranks")
            other = layout["parameters"][index]
            if {k: v for k, v in other.items() if k != "fsdp_shard"} != description:
                raise ValueError("source parameter order/layout differs between trainer ranks")
            shards.append(other["fsdp_shard"])
        sources.append(
            {
                **description,
                "fsdp_shards": shards,
                "producer_ranks": list(range(len(layouts))),
                "export": layouts[0]["export"],
            }
        )
    return sources


def plan_source_buckets(sources, bucket_bytes, max_tensor_bytes):
    buckets, entries, used = [], [], 0
    for source in sources:
        dtype = getattr(torch, source["dtype"].removeprefix("torch."))
        size = math.prod(source["shape"]) * dtype.itemsize
        if size > max_tensor_bytes:
            raise ValueError(f"source tensor exceeds input budget: {source['name']}")
        offset = (used + 15) // 16 * 16
        if entries and offset + size > bucket_bytes:
            buckets.append(entries)
            entries, offset = [], 0
        entries.append({"name": source["name"], "offset": offset, "bytes": size})
        used = offset + size
    if entries:
        buckets.append(entries)
    return buckets
