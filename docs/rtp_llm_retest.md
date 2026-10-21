# TP=4 集群复测清单

本清单对应 2026-10-21 两节点 L20C / Python 3.10.20 / Torch 2.11.0+cu130 /
Ray 2.59.0 / Mooncake 0.3.13.post1 的反馈。TP=4 生成及 NCCL 同步成功，不能替代
RDMA 发布、KVCM 迁移或有效学习的验收。本地开发机器无法连接该验证集群，也没有 CUDA/RDMA。

## 1. RDMA 发布

适配层存在两个可以从源码确认的风险：

- 每 bucket 注册/注销 MR，而 Mooncake 缓存 peer segment。相同地址重新注册后可能继续命中
  旧 rkey。现在使用固定注册的 CPU arena，释放 bucket 只归还区间。
- `transfer_sync_read` 的内部计时器可能在原生操作仍活动时返回失败。现在使用单请求的异步
  提交与 `transfer_check_status`，只把 `1/-1` 当作终态。

依据精确版本的 [Python binding](https://github.com/kvcache-ai/Mooncake/blob/v0.3.13.post1/mooncake-integration/transfer_engine/transfer_engine_py.cpp)、
[RDMA 状态查询](https://github.com/kvcache-ai/Mooncake/blob/v0.3.13.post1/mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp) 和
[peer metadata cache](https://github.com/kvcache-ai/Mooncake/blob/v0.3.13.post1/mooncake-transfer-engine/src/transfer_metadata.cpp)。
这些风险与现场错误相容，但尚不能证明它们是 `remote access error` / `local access violation`
的唯一原因。多 NIC 连通、GID、注册权限和 native transport 的故障路径仍需实机复测。

先跑默认的小包、多版本 smoke，再跑 512 MiB 大包。以下压力配置每版本产生 64 GiB 权重，
要求每个节点的 Ray object store、CPU 内存和 spill 空间足够。初次可将 bucket 数量改为 8。

```bash
VERL_RUN_RDMA_RELAY_TEST=1 RAY_ADDRESS=auto \
VERL_RDMA_TEST_BUCKET_BYTES=536870912 \
VERL_RDMA_TEST_BUCKET_COUNT=128 \
VERL_RDMA_TEST_VERSIONS=3 \
VERL_RDMA_TEST_STAGING_BYTES=2147483648 \
VERL_RDMA_TEST_TIMEOUT_S=600 \
python -m pytest tests/checkpoint_engine/test_rdma_relay_cluster.py -q -s
```

所有 Ray 节点使用相同的新代码。先设置 `VERL_RDMA_DEVICE_NAME` 为两端已验证可互通的单 NIC，
再用空字符串进行多 NIC 对照；这是测试驱动设置，训练配置对应
`actor_rollout_ref.rollout.weight_publication.relay.device_name`。
通过 Ray worker 的启动环境设置 `VERL_LOGGING_LEVEL=INFO`，收集注册/注销时间、session、
bucket ID、地址、字节数、读完成状态和耗时，并与 native NIC/rkey 日志对齐。
分发器日志提供版本、bucket 序号与 source bucket ID 的映射。

Mooncake 该版本优先读取 `MC_GID_INDEX`，不存在时读取 `NCCL_IB_GID_INDEX`，因此原来的
`NCCL_IB_GID_INDEX=3` 不能直接认定为无效。应检查每个 Ray actor 实际收到的环境变量，
以及选定 NIC 的 GID/路由。[配置源码](https://github.com/kvcache-ai/Mooncake/blob/v0.3.13.post1/mooncake-transfer-engine/src/config.cpp)

正常空闲时 `active_bytes=0`、`registered_buckets=0`，但 `registered_bytes` 仍等于 arena
预算，直到 relay 关闭。这不是 bucket 泄漏，也没有增加 GPU 权重副本。发布到 Ray 前会克隆
紧凑的 tensor storage，避免将整个 arena 重复序列化；临时 CPU 内存需计入容量规划。

若 `pending_read.requires_process_restart=True`，说明完成状态不可信。保持两端租约是有意的，
此时不能原地重试、强行注销或复用源区间。停止受影响发布任务的 relay 进程，确认进程退出后
重新创建分发。测试失败时会停止自己创建的 relay；整个压力测试仍应由集群任务平台设置外层
运行时限，防止网络分区导致控制 RPC 永远不能返回。`0/-2` 状态仅表示等待/超时，不能证明完成。

单/多 NIC、大包、多版本均通过后，再恢复 Qwen 27B 全量权重发布；记录每个 replica 的安装
版本、发布耗时和更新前后权重差值。当前本地测试不构成硬件故障已经修复的证据。

## 2. Python 3.10

```bash
python -m pytest \
  tests/workers/rollout/rollout_rtp_llm/test_async_server_kv_cache.py \
  tests/workers/rollout/rollout_rtp_llm/test_tensor_parallel_on_cpu.py \
  tests/workers/rollout/rollout_rtp_llm/test_build_capabilities_on_cpu.py -q
```

异步等待与主动 flush 超时统一使用 `asyncio.TimeoutError`，abort 捕获它及内置
`TimeoutError`，返回 `aborted=False`。测试覆盖取消清理尚未完成时的返回值，以及启动超时后
所有已创建 actor、地址和句柄的清理。本地 Python 3.12 另用独立异常类型模拟 3.10 的分离语义；
仍需在复现环境用实际 Python 3.10 执行上述命令。

## 3. KVCM wheel 与服务

同时安装 RTP-LLM 的新源码与重新构建的原生 wheel。只更新 Python 文件不能补齐 native 能力。
RTP 仓库的 `docs/backend/remote_cache_build.md` 提供 x86 CUDA13 的
`--config=cuda13_remote_cache` 构建命令和库依赖检查。ARM 另外需要架构匹配的 3FS 依赖，
不能直接使用当前 x86_64 RPM。

在每个 rollout 节点、源码目录之外检查安装的库：

```python
from rtp_llm.ops import get_build_capabilities
assert get_build_capabilities()["remote_kv_cache"] is True
import libth_transformer
print(libth_transformer.__file__)
```

verl 在模型配置/权重加载之前执行该检查。旧库缺少查询接口、库加载失败或能力为 false 时
明确拒绝 remote-prefix 启动，不降级为 token-only 迁移。此检查仅证明编译能力，不验证服务。
还需配置可达 KVCM 服务及一致的 domain、namespace，确认相同权重/TP 布局下迁移命中远程 KV，
并覆盖迁移失败回退；随后启用联合调度，再单独测跨版本 token-only re-prefill。

## 4. 有效学习

两步训练退出码为 0、参数版本到 2，只证明流程可推进。若组内 reward 恒为 1，则 GRPO
advantage 可以全为 0，`pg_loss=0`、`grad_norm=0` 与无有效 policy 更新相符。

验收应使用真实评分能够产生组内奖励差异的 prompt/样本和随机采样设置，保持 `G > 1`：

1. 保存每个 prompt 组的 reward 与有效 token mask，确认部分组 reward 方差非零。
2. 检查 advantage 有限且部分非零，loss 有限，实际 optimizer step 的梯度范数有限且非零。
3. 在 optimizer step 前后，对同一组可训练参数或全部分片记录真实数值差值，确认非零且有限；
   排除仅版本计数变化、合成权重测试或 weight decay 引起的变化。
4. 确认更新后的快照安装到 rollout ranks，记录其版本和对应参数变化；评估学习效果仍需更多训练步。

`hendrycks_math` 改为已支持的 `math` 属于数据标签修正，不需要修改 TP 实现或人为制造奖励。

## 5. OSS 产物 + Mooncake KV 数据通路

模型路径、checkpoint、日志继续使用原来的 OSS 挂载目录。KV 的存储选择属于 KVCM
instance group 的服务端配置，与 `MODEL_PATH`、`CKPTS_DIR` 无关。
先在 KVCM 服务端为独立 instance group 配置 Mooncake storage，并部署与原生 SDK
内嵌客户端协议匹配的 master。`KVCM_SERVER_ADDRESS` 填 KVCM 的 metadata RPC 地址，
不能填 Mooncake master 地址。只修改客户端的 `type=mooncake` 不会替换服务端的 file/NFS storage。

当前 SDK RPM 固定为 `2026_09_15_17_40-cuda130.x86_64`。查证
[KVCM 上游依赖](https://github.com/alibaba/tair-kvcache/blob/b079c1adceb67f858d7d447360e685187e4b9771/open_source/deps/git.bzl#L173)
后，master 已固定为 `openanolis/Mooncake@211b75742b6d1fee739ad9a486f2ae9ce2695847`。
其 `PutStart` 四参数签名与该 RPM 的二进制符号一致；现场 master `v0.3.13.post1` 的
接口增加了 `tenant_id`，这是明确的协议差异。SDK 精确源码提交仍不可核实，不能把公开
依赖参考提交当作 RPM 构建提交，也不能把静态协议对齐当成端到端验证通过。
RTP 的 `deps/kvcm_versions.bzl` 记录 artifact/hash/master pin/参考提交，并写入 native
build info。`tools/build_kvcm_mooncake.py` 校验源码 SHA256 后构建配套 master 和 CPU
storage client；详细命令和证据见 RTP 的 `docs/backend/remote_cache_build.md`。
后续必须通过下面的原生测试确认，Python 客户端成功不能替代它。

在 RTP 仓库执行真实跨节点 GPU 测试，两端安装相同的新 wheel：

```bash
RTP_RUN_NATIVE_KV_TEST=1 RAY_ADDRESS=auto \
KVCM_SERVER_ADDRESS=kvcm-host:rpc-port KVCM_INSTANCE_GROUP=mooncake-probes \
python -m pytest rtp_llm/test/smoke/test_native_kv_probe_cluster.py -q -s
```

测试在不同 IP 的两个 GPU 节点上，通过实际链接的 `NativeKVProbe` 执行分配、put/get，
逐字节比较 64 KiB 和 64 MiB GPU buffer，各重复两轮。输出双方 build info、节点、字节数
和 SHA256。源客户端保持存活直至读端完成；超时后停止测试 actor 进程，不提前复用 GPU buffer。
另可用 `python -m rtp_llm.utils.kvcm_probe --address ... --instance-group ...` 做单机启动预检。
该预检会真正调用 SDK 的分配、写入、读取路径，FUSE 的 SDK 105 错误不会被忽略。
启动预检在独立子进程执行，失败或超时后以进程退出结束 native buffer 生命周期；
Mooncake SDK 的 soft timeout 不能视作 DMA 已结束后直接复用 buffer。

配置完成后使用新增入口（模型及产物路径仍由环境变量控制）：

```bash
KVCM_SERVER_ADDRESS=kvcm-host:rpc-port KVCM_INSTANCE_GROUP=mooncake-rollout \
MODEL_PATH=/mnt/fuse/Qwen/Qwen3.8-27B CKPTS_DIR=/mnt/fuse/rl/checkpoints \
bash examples/grpo_trainer/run_qwen3_5_27b_mooncake_kv.sh \
  trainer.nnodes=1 trainer.n_gpus_per_node=4 \
  rollout.nnodes=3 rollout.n_gpus_per_node=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4
```

启动预检默认开启：`trajectory_migration.storage_preflight=True`。当前入口支持直接
KVCM RPC 配置；使用自定义 `kvcm_client_config` 或 VIP discovery 时，需显式关闭该预检，
并独立运行与实际配置等价的原生测试。关闭预检不代表存储通过验收。

## 6. 严格验收与诊断

原生 `wait_remote_cache_status()` 返回 `status/error_code/message`，区分 `success`、
`timeout`、`write_failed`、`unavailable`。SDK 错误经 TP RPC 保留，日志包含失败 rank。
旧 bool API 的 false 只能解释为未知失败，不能归类为超时。
例外是实际 SDK 数据传输返回 `ER_SDK_TIMEOUT=100`：因原生 DMA 完成状态不确定，
该 rank 在释放 KV block 前以退出码 70 结束，现有 TP 故障隔离将该副本置为不可用，需重启。
这与“等待 flush 达到时限、写入仍在执行”的普通等待超时不同。

两个 TP4 副本的长前缀迁移专用验收启用：

```text
++actor_rollout_ref.rollout.trajectory_migration.strict_remote_hit=True
```

该模式要求每条 trajectory 至少完成一次跨副本 remote-prefix 迁移并实际远端命中；
普通完成、未迁移、token-only re-prefill、零远端命中均不能通过。选择足够长、不会在首个
checkpoint 前结束的输入，并使用现有迁移 scorer 确保产生迁移决策。训练输出保留
`trajectory_migrations`（含命中与回退原因）、`trajectory_migration_failures`；聚合指标为
`rollout/{remote_reuse_len,trajectory_migration_attempts,trajectory_remote_hits,trajectory_migration_fallbacks}/{sum,mean}`。
`remote_reuse_len` 是各段实际复用量之和，可能重复包含同一个前缀，不是唯一 token 数。
严格迁移模式仅用于专用验收，不适合要求所有短轨迹都迁移的日常训练。

有效学习的短程验收增加以下覆盖项（`G > 1`，使用真实 `math` reward）：

```text
++trainer.learning_acceptance=True
++actor_rollout_ref.actor.optim.verify_parameter_updates=True
++actor_rollout_ref.rollout.verify_weight_updates=True
actor_rollout_ref.actor.optim.weight_decay=0
actor_rollout_ref.actor.optim.lr_warmup_steps=0
```

训练检查组内奖励方差和有效 advantage；FSDP optimizer 在每步前后每个本地参数分片
最多抽取 4096 个元素，跨 rank 归并检查有限、非零梯度及真实参数差值，输出
`actor/parameter_delta_max` 和 `actor/parameter_probe_elements_per_rank_max`。
这是参数变化的抽样证据，不是完整模型逐字节审计。RTP 在每个转换后的权重安装时对比
真实目标 tensor 与传入数据，失败即中止，成功日志含 `verified_weights`。
这些检查会增加 CPU 拷贝和同步开销，仅用于验收。

RDMA 导出现在按 `min(bucket_bytes, floor(staging_bytes / pipeline_depth / 16) * 16)`
切分超大 tensor，并保存名称、原始形状和 offset 供安装前重组；单个 tensor 不再绕过 staging
容量约束。CPU arena 使用固定槽位，避免不同 bucket 大小与并发释放造成碎片。
日志含 `weight_snapshot export_s/cpu_copy_s/serialization_s`、每 bucket 的
`weight_transfer transfer_s` 和每 rank 的 `weight_install install_s/h2d_s`。
分片限制 relay arena 占用，不限制 Ray object store 或重组完整张量所需 CPU 内存。

验收顺序保持：原生跨节点字节一致 → 两个 TP4 长前缀真实命中 → 4+3×TP4 非零参数更新
→ RDMA 与迁移联合运行。本地 CPU/语法测试不能代替以上四项硬件验收。
