"""vLLM side of the existing trainer-produced immutable snapshot protocol."""

from verl.checkpoint_engine.hf_snapshot_plan import HFSnapshotPlan, describe_hf_layout


class JointSnapshotMixin:
    async def abort_weight_update_from_ipc(self):
        """Close admission during quarantine; never claim an active DMA was canceled."""
        self._submission_paused = self._rejecting = True
        self._resume_event.clear()
        pending = getattr(self, "_joint_weight_install_tasks", set())
        if any(not task.done() for task in pending):
            raise RuntimeError("vLLM native weight installation is still active; retain owners and restart the engine")
        return {"quarantined": True, "active_native_installations": 0}

    def _require_hf_snapshot(self):
        cfg = self.config.weight_publication
        if not cfg.enabled or cfg.transport != "nccl_gpu":
            raise ValueError("vLLM HF snapshots require enabled nccl_gpu publication")
        if self.config.tensor_model_parallel_size != 1 or self.config.quantization is not None:
            raise ValueError("vLLM HF snapshots currently require unquantized TP1")
        if cfg.gpu_snapshot.conversion_location != "trainer" or not cfg.gpu_snapshot.cache_conversion_plan:
            raise ValueError("vLLM HF snapshots require a validated trainer-side plan")

    async def describe_gpu_snapshot(self):
        self._require_hf_snapshot()
        if not hasattr(self, "_hf_snapshot_description"):
            description = describe_hf_layout(self.model_config.local_path, self.config.dtype)
            # The shared GPU pool consumes named metadata records; the identity
            # conversion plan stores canonical triples internally.
            description["layout"] = [
                dict(name=name, shape=shape, dtype=dtype) for name, shape, dtype in description["layout"]
            ]
            self._hf_snapshot_description = description
        return self._hf_snapshot_description

    async def prepare_conversion_plan(self, source_layout, transport):
        description = await self.describe_gpu_snapshot()
        target = [(item["name"], item["shape"], item["dtype"]) for item in description["layout"]]
        self._hf_snapshot_plan = HFSnapshotPlan(source_layout, target, transport)
        plan = self._hf_snapshot_plan
        return {"plan_hash": plan.plan_hash, "manifest": plan.manifest, "rule_construction_s": plan.build_seconds}

    async def export_conversion_plan(self):
        self._require_hf_snapshot()
        plan = self._hf_snapshot_plan
        return {"kind": plan.kind, "plan_hash": plan.plan_hash, "payload": plan.export()}
