"""Request identity records for joining RTP native prefill timing.

Only sizes, versions and request identifiers are recorded. Native request-phase
and batch-forward events provide the timing; first-output latency is not a
substitute for prefill compute time. Profile I/O failure is fatal when enabled.
"""

import json
import math
import os
import socket
import threading
import time
from pathlib import Path

_AUX_FIELDS = {"first_token_cost_time", "wait_time", "cost_time", "input_len", "prefix_len", "reuse_len"}


class RePrefillProfileError(RuntimeError):
    """An enabled profile cannot be recorded completely."""


def validate_profile_context(context, prefix_tokens):
    if context is None:
        return {"context_missing": True}
    required = {
        "logical_request_id",
        "attempt",
        "original_prompt_tokens",
        "generated_before",
        "previous_weight_version",
    }
    if not isinstance(context, dict) or set(context) != required:
        raise RePrefillProfileError("invalid re-prefill profile context fields")
    if not isinstance(context["logical_request_id"], str) or not context["logical_request_id"]:
        raise RePrefillProfileError("profile logical request ID must be a nonempty string")
    for key in ("attempt", "original_prompt_tokens", "generated_before"):
        if type(context[key]) is not int or context[key] < (1 if key == "attempt" else 0):
            raise RePrefillProfileError(f"invalid profile integer: {key}")
    if context["original_prompt_tokens"] + context["generated_before"] != prefix_tokens:
        raise RePrefillProfileError("profile prefix length differs from generation input")
    version = context["previous_weight_version"]
    if version is not None and type(version) not in (int, str):
        raise RePrefillProfileError("invalid previous weight version")
    return {**context, "context_missing": False}


def first_output_aux(aux_info):
    """Whitelist backend numeric fields, retaining their original field names."""
    result = {}
    for key in sorted(_AUX_FIELDS):
        value = getattr(aux_info, key, None)
        if type(value) in (int, float) and math.isfinite(value):
            result[key] = value
    return result


class RequestProfileWriter:
    EVENTS = {"request_rejected", "request_admitted", "request_first_output", "request_terminal"}
    FIELDS = {
        "native_request_id",
        "backend_request_id",
        "logical_request_id",
        "attempt",
        "original_prompt_tokens",
        "generated_before",
        "prefix_tokens",
        "previous_weight_version",
        "installed_weight_version",
        "ticket_mode",
        "context_missing",
        "replica_rank",
        "tp_rank",
        "cache_read_policy",
        "reuse_cache",
        "enable_remote_cache",
        "recompute_candidate",
        "rejection_reason",
        "output_tokens",
        "remote_reuse_tokens",
        "aborted",
        "error_type",
        "aux_info",
    }

    def __init__(self, directory):
        if not isinstance(directory, str) or not Path(directory).is_absolute():
            raise RePrefillProfileError("re-prefill profile directory must be an absolute path")
        self.directory = Path(directory)
        self.host, self.pid = socket.gethostname(), os.getpid()
        self.path = self.directory / f"verl-request-profile-{self.pid}-{time.time_ns()}.jsonl"
        self.lock = threading.Lock()

    def emit(self, event, **fields):
        if event not in self.EVENTS or set(fields) - self.FIELDS:
            raise RePrefillProfileError("unrecognized re-prefill profile event or field")
        if "aux_info" in fields:
            aux = fields["aux_info"]
            if (
                not isinstance(aux, dict)
                or set(aux) - _AUX_FIELDS
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in aux.values())
            ):
                raise RePrefillProfileError("invalid numeric auxiliary profile fields")
        record = dict(
            schema_version=1,
            event=event,
            host=self.host,
            pid=self.pid,
            record_us=time.time_ns() // 1000,
            monotonic_ns=time.monotonic_ns(),
            **fields,
        )
        try:
            data = (json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n").encode()
            with self.lock:
                self.directory.mkdir(parents=True, exist_ok=True)
                fd = os.open(self.path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
                try:
                    offset = 0
                    while offset < len(data):
                        written = os.write(fd, data[offset:])
                        if written <= 0:
                            raise OSError("profile write made no progress")
                        offset += written
                finally:
                    os.close(fd)
        except (OSError, TypeError, ValueError) as error:
            raise RePrefillProfileError("failed to append re-prefill request profile") from error


def writer_for_server(server, directory):
    if directory is None:
        return None
    writer = getattr(server, "_rtp_request_profile_writer", None)
    if writer is None:
        writer = RequestProfileWriter(directory)
        server._rtp_request_profile_writer = writer
    elif str(writer.directory) != directory:
        raise RePrefillProfileError("profile directory changed while server was active")
    return writer
