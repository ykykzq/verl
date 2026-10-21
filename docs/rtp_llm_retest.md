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
