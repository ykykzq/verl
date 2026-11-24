# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");

"""Exercise client-side backpressure with more producers than Ray actor slots."""

import asyncio
import subprocess
import sys
from pathlib import Path

import ray

from verl.experimental.fully_async_policy.message_queue import MessageQueue, MessageQueueClient


async def exercise(actor):
    client = MessageQueueClient(actor)
    producers = [asyncio.create_task(client.put_sample_wait(index)) for index in range(32)]
    try:
        while (await client.get_queue_size()) < 2:
            await asyncio.sleep(0.01)
        # Let other producers attempt a full queue before scheduling consumers.
        await asyncio.sleep(0.3)
        stats = await client.get_statistics()
        assert stats["queue_size"] == 2 and stats["dropped_samples"] == 0
        received = [(await client.get_sample())[0] for _ in range(32)]
        assert all(await asyncio.gather(*producers))
        assert sorted(received) == list(range(32))
        await client.put_sample_wait(None)
        assert (await client.get_sample())[0] is None
        stats = await client.get_statistics()
        assert stats["total_produced"] == stats["total_consumed"] == 33
        assert stats["queue_size"] == stats["dropped_samples"] == 0
    finally:
        for task in producers:
            task.cancel()
        await asyncio.gather(*producers, return_exceptions=True)


def run_probe():
    # Isolate this probe from any existing training Ray cluster.
    ray.init(address="local", num_cpus=1, include_dashboard=False, object_store_memory=100_000_000)
    actor = None
    try:
        actor = MessageQueue.options(num_cpus=0, max_concurrency=20).remote(None, 2)
        asyncio.run(asyncio.wait_for(exercise(actor), 30))
        print("32 producers, capacity 2, 33 messages consumed without drops", flush=True)
    finally:
        if actor is not None:
            ray.kill(actor)
        ray.shutdown()


def test_producers_do_not_starve_consumers_of_ray_actor_slots():
    result = subprocess.run(
        [sys.executable, "-m", "tests.experimental.fully_async_policy.test_message_queue_backpressure_on_cpu"],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=90,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "33 messages consumed without drops" in output, output


if __name__ == "__main__":
    run_probe()
