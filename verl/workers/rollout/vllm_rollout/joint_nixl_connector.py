"""NIXL pull for a generated prefix, including its uncomputed final token."""

from verl.workers.rollout.vllm_rollout.native_libraries import load_nixl_dependencies

load_nixl_dependencies()

from vllm.distributed.kv_transfer.kv_connector.v1.nixl import NixlConnector


class JointNixlConnector(NixlConnector):
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        params = request.kv_transfer_params or {}
        if params.get("v17_trajectory_migration") and params.get("do_remote_prefill"):
            # Ordinary PD assumes all prompt KV exists remotely. After decode,
            # the latest sampled token has not itself passed through the model.
            # Import only the source's materialized frontier; compute the tail.
            frontier = params.get("remote_num_tokens")
            if type(frontier) is not int or not 0 < frontier < request.num_prompt_tokens:
                raise ValueError("trajectory transfer lacks a valid materialized KV frontier")
            count = max(0, frontier - num_computed_tokens)
            return count, bool(count)
        return super().get_num_new_matched_tokens(request, num_computed_tokens)
