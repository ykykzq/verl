# Recipe: Fully Async Policy Trainer

**Author:**  `https://github.com/meituan-search`

Last updated: 05/25/2026.

本文档介绍了完全异步PPO训练系统，该系统实现了 Trainer 和 Rollouter 的完全解耦，支持异步样本生成和训练。
在该系统下，我们使用128卡训练qwen2.5-7B模型取得了2.35x-2.67x的性能提升,同时效果没有显著受到影响。

## Introduction

### Background

rollout和train分离架构相较于colocate的架构能够更加灵活地分配资源，设计更加灵活的训练逻辑，从而处理长尾等问题带来的GPU利用率低，训练效率低的问题。
one_step_off_policy通过分离架构的设计并进行rollout和train一轮异步的训练方法，缓解了rollout时间过长的问题，并在训练效率上取得了一些收益，
但其强制使用一轮异步的数据，存在不够灵活等问题，而且并不能完全去除长尾对训练效率带来的的影响；在其他框架如areal、Magistral、streamrl、asyncflow上，
已经基于分离架构实现了异步训练、流式训练，并取得了收益；我们借鉴其方法，在verl上进行了实现。fully_async_policy支持异步、流式、partial
rollout的训练， 通过合理设置资源分配情况、参数同步频率等参数，fully_async_policy能够显著提高训练效率。

> Magistral https://arxiv.org/abs/2506.10910
>
> AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language
> Reasoning https://arxiv.org/abs/2505.24298
>
> StreamRL: Scalable, Heterogeneous, and Elastic RL for LLMs with Disaggregated Stream
> Generation https://arxiv.org/abs/2504.15930
>
> AsyncFlow: An Asynchronous Streaming RL Framework for Efficient LLM Post-Training https://arxiv.org/abs/2507.01663
>

### 核心贡献

* **资源隔离**：与使用hybrid_engine不同，Rollouter和Trainer使用分离的计算资源，需要分别指定所占用的资源。
* **生成与训练并行**：Trainer在训练的同时，Rollouter在生成新的样本。
* **多步异步**: 相比 one step off policy 支持0.x步到多步的异步设定，异步方案更加灵活。
* **nccl参数同步**：基于nccl通信原语，参考[checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine)实现Rollouter与Trainer间的高效参数同步。
* **Stream推理与训练**：Rollouter逐样本生成数据，同时数据传输以单个sample为最小传输单位。
* **异步训练与新鲜度控制**：通过设置参数async_training.staleness_threshold，支持使用旧参数生成的样本进行训练。
* **PartialRollout**: Rollouter推理过程支持partial rollout逻辑，通过参数同步时，添加`sleep()`和`resume()`
  逻辑，保存进行中的rollout的样本，并在下一次rollout中继续使用，减少参数同步等待进行中的任务结束时间。

目前支持使用模式包括 megatron/fsdp+vllm，以及本分支的 FSDP/FSDP2 + standalone RTP-LLM 适配。
vllm必须使用基于AgentLoop的server模式。

## 设计

fully_async_policy的整体架构如下图所示，fully_async_policy主要由Rollouter、MessageQueue、Trainer、ParameterSynchronizer四部分组成。

![fully_async_policy_structure](
https://github.com/ArronHZG/verl-community/blob/main/docs/fully_async_policy_structure.svg?raw=true)

1. Rollouter逐样本生成序列，并将生成的sample放入MessageQueue中，生产的速度受新鲜度控制。
2. MessageQueue用于暂存Rollouter生成的sample。
3. Trainer逐样本从MessageQueue中获取，获取到`require_batches*ppo_mini_batch_size`
   数量的样本后，就会进行训练，训练async_training.trigger_parameter_sync_step轮后，触发与Rollouter的一次参数同步。
4. ParameterSynchronizer 实现了Nccl的同步参数同步能力。

当前方案对比base的收益来源，在于colocate情况下，rollout使用更多的资源无法解决长尾样本带来的空闲，
当我们进行资源隔离后，rollout的时间和train的时间都可能相较于之前更长（因为使用的资源变少了），
但是相互之间的耗时overlap，端到端的耗时反而有所缩减。

![fully_async_policy_revenue](
https://github.com/ArronHZG/verl-community/blob/main/docs/fully_async_policy_revenue.svg?raw=true)

## RTP-LLM 独立权重发布

通过以下参数启用独立发布器。未开启时继续使用原有权重同步路径。

```yaml
actor_rollout_ref:
  hybrid_engine: false
  actor:
    strategy: fsdp2
  rollout:
    name: rtp_llm
    checkpoint_engine:
      backend: nccl
      update_weights_bucket_megabytes: 512
    fine_grained_weight_update:
      enabled: true
    weight_publication:
      enabled: true
      wait_timeout_s: 600
    trajectory_migration:
      enabled: false
async_training:
  partial_rollout: true
  use_trainer_do_validate: false
  use_dynamic_resource_scheduling: false
```

以上是单独验证权重发布的配置，不依赖外部 KVCM 服务。trajectory 迁移是可选功能；要验证迁移和
跨版本续跑，再开启 `trajectory_migration.enabled=true` 和 `allow_cross_version_recompute=true`。
默认 `remote_prefix` 迁移后端会启用 RTP remote cache，需要事先配置可用的 KVCM 服务及匹配的
domain / namespace。未配置该服务时，不应直接套用 remote-prefix 迁移配置进行发布器 smoke 测试。

常规训练步只等待完整快照导出和发布准入确认。所有 FSDP ranks 参与已有的参数导出 collective，
rank 0 将与训练参数脱离的 CPU tensor 按 bucket 写入 Ray object store，然后把包含对象引用的
manifest 交给独立 Ray actor。训练随后可以修改参数，不会改变已经发布的快照。

发布器一次更新一个 replica，优先处理版本最旧且未 pin 的正常 replica。新版本替换尚未开始的旧版本任务，
正在执行的单 replica 更新则使用原快照完成，包括重试。发布器保留最新快照和正在使用的快照，
训练导出期间还会暂时存在第三份。应为完整 CPU 权重、传输缓存及单个最大 tensor 预留足够内存；
object store 容量不足或溢写可能增加快照导出耗时。

目标 rollout worker 从 object store 逐 bucket 获取权重、搬到 GPU，再复用 RTP-LLM 的 IPC 安装流程。
这条路径不让训练 workers 参与逐 replica 的通信组或重试，也不需要额外的发布 GPU；数据传输使用
Ray object store，而不是原来的训练到 rollout NCCL 广播。其网络吞吐需要单独测量。

每个目标仍执行准入关闭、请求中断和排空、KV 清理、完整权重安装、版本校验和恢复服务。
最终失败的 replica 进入 `QUARANTINED`，其他 replica 继续运行。只有某版本第一次成功安装后，
发布器才通知 rollouter 重置现有 staleness 预算。样本队列与 staleness 的背压机制继续生效。

初始加载和显式验证等待目标版本发布完成；等待屏障跳过 pinned / quarantined replica，且至少需要一个
已达到目标版本的 `SERVING` replica。因此 pin 旧版本时，验证结果可能包含多个版本的行为。
关闭时会排空或取消发布任务并清理 actor。后台致命错误和所有 replica 均被隔离的情况会传播到训练任务。

可从 trainer 获取发布器 handle，独立控制 pin、恢复和查看状态：

```python
publisher = ray.get(trainer.get_weight_publisher.remote())
ray.get(publisher.pin_replica.remote(0, "finish long trajectories"))
ray.get(publisher.unpin_replica.remote(0))  # Catch up to the retained latest snapshot.
ray.get(publisher.recover_replica.remote(1))
status = ray.get(publisher.get_status.remote())
```

状态包含最新发布版本、当前传输版本和 replica、已提交版本、各 replica 状态以及后台错误。
训练侧分别记录 `timing_s/weight_snapshot_export` 和 `timing_s/weight_publication_admission`。
快照是内存中的临时发布数据，不是持久化训练 checkpoint；训练导出进程或发布 actor 故障后不保证恢复。

当前限定 FSDP/FSDP2 完整权重导出和 standalone RTP-LLM，不支持 hybrid / dynamic resource scheduling、
trainer-side validation 或未合并的 LoRA。被中断 trajectory 的续跑继续使用已有 partial rollout 和
跨版本 re-prefill 机制；独立发布不消除 off-policy，也不新增 staleness 剪枝算法。

上线前应在 GPU 集群验证：训练步与慢 replica 安装确实重叠；多版本连续发布时显存和 object store 占用有界；
pin/unpin 和失败恢复不误更新版本；续跑 trajectory 的版本元数据及 re-prefill 行为正确。
CPU 单测和本地多 actor Ray 测试不代表已完成 GPU 吞吐或大模型内存验收。
GPU smoke 必须检查实际完成的 optimizer steps，不能只依据退出码或初始权重发布屏障判断成功。

### 权重更新与 trajectory 联合调度

在上述 standalone RTP-LLM 发布配置上增加以下配置。该算法默认关闭；开启前需要部署可用的 KVCM。
`kv_capacity_tokens` 和 `max_batch_size` 必须根据实际 KV 容量和 decode roofline 测量填写，下面的
数值只是配置示例，不代表某个模型的推荐值。

```yaml
actor_rollout_ref:
  rollout:
    enable_prefix_caching: true
    trajectory_migration:
      enabled: true
      allow_cross_version_recompute: true
      checkpoint_tokens: 64
      kv_transfer_backend: remote_prefix
    weight_publication:
      enabled: true
      joint_scheduling:
        enabled: true
        kv_capacity_tokens: 65536
        kv_block_size_tokens: 16
        max_batch_size: 32
        update_cooldown_s: 5.0
        min_decode_window_s: 2.0
        min_decode_tokens: 64
        max_defer_s: 30.0
        evacuation_timeout_s: 5.0
        staleness_version_gap: 2
        tail_tokens: 32
        tail_kl_threshold: 0.1
        probe_interval_s: 5.0
        probe_timeout_s: 2.0
        max_concurrent_probes: 1
async_training:
  partial_rollout: true
```

联合调度要求 `async_training.partial_rollout=true`，使更新中断的 trajectory 保留前缀并续跑；
配置校验会拒绝关闭此选项，避免将更新中断误当作生成完成。

每次发布先在 router 执行预请求，关闭所有旧版本 replica 的新 trajectory 准入。完整逻辑请求在
checkpoint 间隙仍保持登记和容量占用，因此已有续跑不会被当作新请求。新请求只进入已安装最新
发布版本的 replica；所有 replica 暂不可用时在客户端等待，所有可用 replica 被 pin 时发布健康检查
会报错，提示 unpin/recover，避免训练静默等待数据。

后台优先选择逻辑 trajectory 较少的 replica 进行排空。checkpoint 上的同版本迁移优先填入已有负载
的兼容 replica，释放出的 replica 更新后恢复新请求准入。版本、模型、rollout 配置指纹、KVCM domain
和 namespace 必须匹配。达到排空期限、最长延期，或只剩少量且无法迁移的请求时，执行原地更新；
幸存请求保留 token ids，通过显式 recompute ticket 在新版本重新 prefill。高负载 replica 也可向同版本
低负载 replica 分流。失效源可在兼容的同版本健康 replica 上显式 recompute 恢复。

迁移目标与更新选择在 router 内预留，防止多条 trajectory 同时抢占剩余容量。容量按完整
`prompt + max_new_tokens` 向 KV block 大小取整保守预留，同时约束逻辑请求数；它不是引擎实时显存
使用率。并发上限还会受实际引擎上限约束。混合/线性注意力模型的实际 block 可能是 1024，必须对应
设置 `kv_block_size_tokens`；配置的记账粒度必须是实际 block 大小的整数倍，否则拒绝运行。
实际模型序列上限、权重和运行时工作区也必须纳入部署容量规划。

每次更新后保留 `update_cooldown_s`，每次迁移后保证 `min_decode_window_s` 的连续解码时间。
`min_decode_tokens` 提供额外的 token 进度保护，但最长延期到期后可放宽该 token 条件，防止停滞请求
无限阻塞更新；时间窗口仍然保留。连续发布只替换待更新的快照，不重置最早等待时间。即将到达更新
期限的旧 replica 不再接收迁入请求，避免尾部工作不断被转移到同一台 replica。

版本差达到 `staleness_version_gap` 后，在两个版本上异步 teacher-force 当前完整前缀，只返回最后
`tail_tokens` 个生成 token 的 raw logprob，计算非负 sampled k3 指标
`mean(expm1(log_p_new - log_p_old) - (log_p_new - log_p_old))`。超过阈值后在后续 checkpoint 尝试迁到
较新版本。这个指标是抽样 KL 代理，不是全词表精确 KL；temperature/top-p 等采样变换会影响其解释。
它衡量当前两个策略在所选前缀上的偏离，不会清除历史 token 的 off-policy 属性，也不剪枝 GRPO 组。

评分仍需要完整前缀前向，并非只计算尾部 token。两端都预留临时 KV 和一个请求槽；容量不足时跳过，
发布调度不等待评分。超时或过期版本的结果不触发迁移；临时预留保留至两个后端 RPC 都结束，避免
本地超时后后端仍计算却释放容量。后端另有 `VERL_RTP_LLM_TAIL_SCORE_TIMEOUT_S`（默认 5 秒）限制。
生成请求取消时，也要先确认对应后端 RPC 退出，再释放逻辑请求容量。

router 的 `get_status()` 增加 `joint_scheduling`，发布器状态增加 `scheduling_decision`。
生成输出保存 `trajectory_segments`、迁移记录及 `forced_prefill_tokens`，保留各段行为版本和 logprob。
默认使用 CPU snapshot/Ray object store 发布路径，也可开启下述 RDMA relay。两种方式都没有新增完整
GPU 权重副本。实际 KVCM 传输、teacher-forcing 支持和端到端收益仍需在 GPU 集群验收。

### CPU RDMA 权重 Relay

在独立权重发布配置上启用：

```yaml
actor_rollout_ref:
  rollout:
    weight_publication:
      enabled: true
      transport: mooncake_rdma
      relay:
        device_name: ""
        max_staging_bytes: 4294967296
        pipeline_depth: 2
```

数据路径为 `trainer CPU snapshot -> master relay -> node relay -> 本机 rollout worker -> GPU`。
master 固定在实际导出权重的训练 rank 0 所在节点，每个 rollout worker 节点只创建一个 CPU relay。
trainer 仍只等待快照导出与发布准入；打包、注册 CPU 内存、跨节点传输由后台执行。
节点 relay 通过 Mooncake `transfer_sync_read` 从前驱读取 bucket；控制 RPC 只传描述符和引用，
跨 relay 的权重字节走 RDMA。本机落地为不可变 Ray object-store bucket，再复用现有安装器。
TP/PP 分片仍由 RTP-LLM 的现有 named-tensor 安装路径处理。

`pipeline_depth` 个 bucket 可同时流过链路。任一节点收到完整版本后即可安装，无需等全部节点；
安装前先确认该 replica 的所有 worker 节点就绪，再中断生成、释放 KV 并切版权重。
中间节点失败时，下游尝试更早的前驱直到 master；同版本重试和显式恢复会重建失败的分发。
已确认死亡的 relay 按节点隔离；存活节点的注销失败会保留注册内存并报告错误。
CPU buffer 在所有依赖读取完成前不会注销，取消或关闭也要等待底层读取实际结束。
底层原生读取超时由 Mooncake 配置控制，Python 不用强行释放 buffer 来模拟超时。

内存需要按模型规模规划：训练侧保留 active/latest 的 CPU 快照；节点侧保留一个正在使用的完整
CPU 版本，并占用流水线暂存内存和序列化临时内存。`max_staging_bytes` 只限制每个 relay 的注册
暂存区，不限制 Ray object store。应至少覆盖流水深度乘以最大打包 bucket（含对齐填充）；
单个大 tensor 可能超过 checkpoint bucket 配置值。增大此值不会新增完整 GPU 权重缓冲。
连续发布合并为最新待处理版本，正在安装的版本持有租约，旧版本清理完成后才启动替代分发。

所有参与节点需安装提供 `mooncake.engine.TransferEngine` 的 Mooncake，并具备可用的 RDMA NIC、
网络连通和足够的内存注册权限。RDMA 初始化失败直接报错，不会静默退回 TCP/Ray 跨节点传权重。
接口依据 [Mooncake Transfer Engine Python API](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/api-reference/python/transfer-engine.md)。
默认 `transport: ray` 不要求安装 Mooncake。发布器 `get_status()` 的 `relay` 字段提供分发版本、
就绪节点、节点错误和清理错误。

真实 RDMA 集群验收（已有 Ray 集群至少两个存活节点）：

```bash
VERL_RUN_RDMA_RELAY_TEST=1 RAY_ADDRESS=auto \
  python -m pytest tests/checkpoint_engine/test_rdma_relay_cluster.py -q
```

可用 `VERL_RDMA_DEVICE_NAME` 指定 NIC，用 `VERL_RDMA_TEST_TIMEOUT_S` 调整测试等待时间。
该 smoke 检查混合 dtype、多 bucket 的逐值一致性及注册内存回收，不代表大模型吞吐验收。
本地 CPU/Ray 测试使用测试专用的字节拷贝后端，不代表已经验证 RDMA 硬件性能。

## 使用方式

### 参数说明

| super params                                                     | implication                                                     |
|------------------------------------------------------------------|-----------------------------------------------------------------|
| `trainer.nnodes`                                                 | Trainer的node数量                                                  |
| `trainer.n_gpus_per_node`                                        | Trainer每个node上gpu的数量                                            |
| `rollout.nnodes`                                                 | Rollouter的node数量                                                |
| `rollout.n_gpus_per_node`                                        | Rollouter每个node上gpu的数量                                          |
| `data.train_batch_size`                                          | 在fully async策略中，该值不生效（默认设置为0）                                   |
| `data.gen_batch_size`                                            | 在fully async策略中，使用流式的样本生产逻辑（默认设置为1)                             |
| `rollout.total_rollout_steps`                                    | 总的rollout的sample数量                                              |
| `rollout.test_freq`                                              | Rollouter每更新多少次参数，进行一次validation                                |
| `actor_rollout_ref.actor.ppo_mini_batch_size`                    | The ppo_mini_batch_size is a global num across all workers/gpus |
| `actor_rollout_ref.actor.use_rollout_log_probs=True`             | 使用rollout产生的log_probs                                           |
| `algorithm.rollout_correction.bypass_mode`                       | 是否在train阶段，使用train模型的参数计算token的 log_prob                        |
| `async_training.require_batches`                                 | FullyAsyncTrainer一次性获取的ppo_mini_batch_size的数量                   |
| `async_training.trigger_parameter_sync_step`                     | 表示FullyAsyncTrainer进行多少次本地更新后,进行一次参数同步                          |
| `async_training.staleness_threshold`                             | 新鲜度控制                                                           |
| `async_training.partial_rollout`                                 | 是否进行partial_rollout                                             |
| `async_training.use_trainer_do_validate`                         | 是否使用Trainer的do_validate方法进行validation，默认值False                  |

**进一步的解释：**

* `rollout.total_rollout_steps`

  与 colocate 相比，数量可以通过 train_batch_size 与 step 相乘对齐:
  `rollout.total_rollout_steps = data.train_batch_size * step`。

* `async_training.trigger_parameter_sync_step`

  在fully async策略中，表示Trainer进行多少次本地更新后（也就是获取多少次`require_batches * ppo_mini_batch_size`数量样本），
  与Rollouter之间进行一次参数同步。
  每两次Rollouter和Trainer参数同步之间，Trainer将会处理`trigger_parameter_sync_step* require_batches\
  ppo_mini_batch_size`份sample。
  如果为了与colocate在公平的情况下对比速度，trigger_parameter_sync_step应该设置为 `data.train_batch_size / (
  require_batches * ppo_mini_batch_size)`。

* `async_training.staleness_threshold`

  在fully async策略中，表示最大允许使用的staleness样本的比例。

    * staleness_threshold=0，表示同步训练。
      Rollouter两次参数更新之间将会生成固定数量的样本，样本数为：
      $$rollout\_num = (trigger\_parameter\_sync\_step*require\_batches*ppo\_mini\_batch\_size)$$
    * staleness_threshold>0，表示异步训练， 可以设置为小数，支持更灵活的异步调用。
      Rollouter两次参数更新之间将会最多生成的样本数为：
      $$rollout\_num = (1+staleness\_threshold)*(trigger\_parameter\_sync\_step*require\_batches*ppo\_mini\_batch\_size) - num\_staleness\_sample $$

  num_staleness_sample 表示上一次rollout多生成的陈旧样本数。

  由于是流式系统，rollout持续生成，trainer持续消费。如果rollouter较慢，trainer会更早触发参数同步，rollouter并不会实际生产rollout_num个样本。
  当rollout 足够快时，staleness_threshold设置为1，基本上等价于one_step_off policy。
  为了避免过期样本太多影响训练精度，建议该值设置小于1。

* `async_training.partial_rollout`

  partial_rollout只会在staleness_threshold>0时才实际上起作用。

* `actor_rollout_ref.actor.use_rollout_log_probs=True`

  在强化学习算法中，log_probs与参数版本，token都存在隐性的相关性。由于PPO/GRPO/DAPO等算法的设定，我们在计算重要性采样时，
  即 old_log_prob必须使用rollout参数及token所对应log_probs，才能保证算法的正确性。在fully
  async策略中，我们默认old_log_prob是由rollout所计算的，而不是由trainer所计算。

* `algorithm.rollout_correction.bypass_mode`
  algorithm.rollout_correction.bypass_mode 默认为 True, 直接使用rollout log prob。

  我们在训练过程中，观测到随着训练的进行，训练后期指标和response长度可能会出现不稳定的情况，
  这里我们可以使用 [Rollout Importance Sampling](https://verl.readthedocs.io/en/latest/advance/rollout_is.html) 的技术进行
  重要性采样，缓解这一问题。为了使用 `Rollout Importance Sampling` 我们需要使用训练引擎使用当前的参数版本计算old_log_prob，此开关需要打开。
  此外，在 mode d (async stream pipeline with partial rollout) 的情况下 `algorithm.rollout_correction.bypass_mode=False`
  以及
  `Rollout Importance Sampling` 后，我们的实现已近似Areal的 `Decoupled PPO`。

* `async_training.require_batches`

  在流式训练中，require_batches 应该设置为1，表示生产够ppo_mini_batch_size样本后，就进行训练。
  在实际测试中，我们发现，如果单次下发的样本较少，由于数据分发的顺序，会导致训练不稳定，response 长度变长。
  在这里，我们额外提供 require_batches 进行流式分发，单次参与训练的样本数量控制。

* `async_training.use_trainer_do_validate`

  控制是否使用trainer的`do_validate`方法进行validation。
  如果设置为True，trainer会在每次参数更新后，调用`do_validate`方法进行validation。
  如果设置为False，trainer不会调用`do_validate`方法。

### 模式支持

1. on policy pipeline:
    1. **trigger_parameter_sync_step=1，staleness_threshold=0**
    2. Rollouter一次生产`require_batches*ppo_mini_batch_size`
       的samples，Trainer获取这些samples后进行训练，训练完后Trainer和Rollouter之间进行一次参数同步;
    3. 在rollout阶段，如果存在长尾的样本，但是rollout样本数较少时，较短的样本无法填充到空闲的资源中，会造成一定的资源浪费。
    4. 如图a所示；

2. stream off policy pipeline:
    1. **trigger_parameter_sync_step>1，staleness_threshold=0**
    2. 将会进行同步的流式训练，Rollouter一次生产`require_batches*ppo_mini_batch_size*trigger_parameter_sync_step`
       的samples，Trainer每获取`require_batches*ppo_mini_batch_size`
       就进行一次本地训练，训练trigger_parameter_sync_step次后，Trainer和Rollouter之间进行一次参数同步;
    3. 相较于a，由于一次生成的样本更多，资源的空闲会更低。
    4. 在一次step训练中，会存在两次资源闲置的时间，分别是在第一次获取样本时，train等待`require_batches*ppo_mini_batch_size`
       个样本生产，以及最后一次参数更新时，rollout等待训练完成。
    5. 如图b所示；

3. async stream pipeline with staleness samples:
    1. **trigger_parameter_sync_step>=1，staleness_threshold>0，partial_rollout=Flase**
    2. Rollouter在每次参数更新后将计划最多生产rollout_num个样本（实际根据rollout速度，生成的样本可能会少与这个值）。
    3. 如果rollout过程比较快，Rollouter将会在参数同步前额外生成一部分样本num_stale_samples，用于参数同步后立即给Trainer使用。
       触发参数同步时，如果Rollouter有正在生产的任务，将会等待任务完成，同时不会添加新的任务；
    4. 相较于b，除第一次step训练外，后续的训练都不会有wait first batch rollout finish的时间，但是会有wait active task
       finish的时间。
    5. 如图c所示；

4. async stream pipeline with partial rollout:
    1. **trigger_parameter_sync_step>=1，staleness_threshold>0，partial_rollout=True**
    2. 相较于c，触发参数同步时，Rollouter如果有正在生产的sample，会打断rollout过程并进行参数同步，被中断的sample会在参数同步后继续生成。减少了wait
       active task finish的时间。
    3. 如图d所示；

![fully_async_policy_mode](
https://github.com/ArronHZG/verl-community/blob/main/docs/fully_async_policy_mode.svg?raw=true)

### 关键指标

| metrics                                        | implication                                               |
|------------------------------------------------|-----------------------------------------------------------|
| `trainer/idle_ratio`                           | Trainer闲置率                                                |
| `rollouter/idle_ratio`                         | Rollouter闲置率                                              |
| `fully_async/count/stale_samples_processed`    | 训练使用的旧sample总数                                            |
| `fully_async/count/stale_trajectory_processed` | 训练使用的旧trajectory总数(一个sample会生产rollout.n条trajectory)       |
| `fully_async/partial/total_partial_num`        | 两次trigger_parameter_sync_step之间Trainer处理的partial样本数       |
| `fully_async/partial/partial_ratio`            | 两次trigger_parameter_sync_step之间Trainer处理的partial样本的比例     |
| `fully_async/partial/max_partial_span`         | 两次trigger_parameter_sync_step之间Trainer处理的partial样本的最大参数跨度 |

### 调参建议

* 资源分配与调整:
    * 合理的资源分配是获得好的训练效率的前提。理想的资源分配情况应该是使得Rollout的时间和Train的时间接近，从而使得整个训练过程流水气泡最小，
      避免资源闲置，同时Trainer不会使用旧样本。在真实训练场景下，可以根据实际训练过程中rollout和train的空闲时间调整资源分配，
      可从rollouter/idle_ratio和trainer/idle_ratio获得，如果rollouter/idle_ratio较高trainer/idle_ratio较低，
      应该增多Trainer的资源减少Rollouter的资源，反之亦然。

* 关键参数：
    * staleness_threshold: 设置太大会导致较多的旧样本使用，影响模型效果，建议设置小于1。
    * require_batches：越接近1，越接近纯流式过程，训练过程中bubble越小，能够在速度上获得更快的加速效果，但会对样本的处理顺序产生影响；
    * trigger_parameter_sync_step: 设置的越小越接近on policy，但会导致频繁的参数同步，长尾样本浪费的资源无法被短样本填充，资源利用率低。
      设置的越大有更高的计算效率，但是精度上会受到off policy的影响。
    * rollout.test_freq: 会占用Rollouter资源，不建议设置太小。

* 模式选择：通过调整不同的参数，Fully Async架构支持不同程度上的优化加速，适用于不同场景的任务。
    * 对于小规模任务，需要保证训练的稳定性和 on-policy 性，对速度要求不高的场景，可以尝试使用on policy pipeline的模式（模式1）。
    * 对于需要提高训练吞吐量，但对 staleness 敏感的场景，可以尝试使用 stream off policy pipeline 的模式。即通过
      设置trigger_parameter_sync_step>1 ，提高 训练效率，但仍保持同步机制 (staleness_threshold=0 )（模式2）。
    * 对于大规模任务，对训练速度有较高要求，且可以容忍一定 off-policy 程度、staleness的场景，可以设置staleness_threshold>
      0、partial_rollout=True提高训练效率，使用 async stream pipeline 模式（模式 3 或 4）。

### 快速开始

```shell
rollout_mode="async"
rollout_name="vllm" # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

train_prompt_bsz=0
gen_prompt_bsz=1
n_resp_per_prompt=16
train_prompt_mini_bsz=32
total_rollout_steps=$(((512*400)))
test_freq=10
staleness_threshold=0
trigger_parameter_sync_step=16
partial_rollout=False


python -m verl.experimental.fully_async_policy.fully_async_main \
	train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.return_raw_chat=${return_raw_chat} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    trainer.nnodes="${NNODES_TRAIN}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    rollout.nnodes="${NNODES_ROLLOUT}" \
    rollout.n_gpus_per_node="${NGPUS_PER_NODE}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    rollout.test_freq="${test_freq}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.partial_rollout="${partial_rollout}"
```

## 实验

### 在7B模型上进行异步训练

我们使用 Qwen2.5-Math-7B 验证 fully async 策略在长候选下，多种资源下的收益情况。
使用`async stream pipeline with staleness samples` 策略，我们在32卡，64卡，128卡都取得2x左右的性能提升，同时没有显著影响实验效果。

* 机器：H20
* 模型：Qwen2.5-Math-7B
* rollout长度：max_response_length FSDP2: 28K tokens;
* 算法：DAPO
* 数据集： TRAIN_FILE: dapo-math-17k.parquet TEST_FILE: aime-2024.parquet
* engine: vllm+FSDP2
* rollout.n: 16
* ppo_mini_batch_size: 32
* test_freq: 20

* colocate sync:
    * step: 400
    * train_batch_size: 512

* fully_async_policy
    * total_rollout_steps: 512*400
    * require_batches: 4
    * trigger_parameter_sync_step: 4
    * staleness_threshold: 0.5
    * partial_rollout: True

|  training mode   	   | resource allocation 	 | step  	  |  gen  	  | old_log_prob 	 | update_actor 	 | total time<br>100 step 	 | total time<br>200 step 	 | total time<br>300 step 	 | total time<br>400 step 	 |      acc/mean@1          	      |
|:--------------------:|:---------------------:|:--------:|:--------:|:--------------:|:--------------:|:------------------------:|:------------------------:|:------------------------:|:------------------------:|:-------------------------------:|
| colocate sync      	 | 32                  	 | 790.10 	 | 357.41 	 | 107.71       	 | 269.80      	 | 13h 44m                	 | 1d 3h 43m              	 | 2d 9h 22m              	 | 3d 17h 5m              	 | max: 0.3313<br>last: 0.2448  	  |
| fully_async_policy 	 | 16:16               	 |  294.77  |  21.26   | \            	 |      313.81     |    7h 58m<br>(1.72x)     |    16h 21m<br>(1.70x)    |   1d 0h 53m<br>(2.31x)   |   1d 9h 26m<br>(2.66x)   | max: 0.3302<br>last: 0.2333   	 |
| colocate sync      	 | 64                  	 | 365.28 	 | 150.72 	 | 70.26        	 | 133.41       	 | 10h 22m                	 | 20h 45m                	 | 1d 7h 6m               	 | 1d 17h 32m             	 | max: 0.3365<br>last:  0.2333 	  |
| fully_async_policy 	 | 32:32               	 | 189.26 	 | 28.46  	 | \            	 | 156.98       	 | 4h 57m<br>(2.09x)      	 | 10h 14m<br>(2.03x)     	 | 16h 58m<br>(1.83x)     	 | 21h 40m<br>(1.92x)     	 | max: 0.3677<br>last: 0.3406  	  |
| colocate sync      	 | 128                 	 | 356.30 	 | 177.85 	 | 53.92        	 | 113.81       	 | 8h 36m                 	 | 17h 56m                	 | 1d 5h 6m               	 | 1d 16h 48m             	 | max: 0.3573<br>last: 0.2958  	  |
| fully_async_policy 	 | 64:64               	 | 150.63 	 | 33.14  	 | \            	 | 113.16       	 | 3h 13m<br>(2.67x)      	 | 6h 46m<br>(2.65x)      	 | 10h 53m<br>(2.67x)     	 | 17h 22m<br>(2.35x)     	 | max: 0.3521<br>last: 0.3094  	  |

> source data: https://wandb.ai/hou-zg-meituan/fully-async-policy-colocate_async?nw=nwuserhouzg

### 128卡 7B 异步模式实验

我们使用 Qwen2.5-Math-7B 验证 fully async 所支持的各个模式的效果。
我们可以看到 stream 带来的收益大约1.6x，叠加 staleness 和 partial_rollout 后，收益为2.35x。

|                             mode                                         	                              | step  	  |  gen  	  | old_log_prob 	 | update_actor 	 | total time<br>100 step 	 | total time<br>200 step 	 | total time<br>300 step 	 | total time<br>400 step 	 |      acc/mean@1         	      |
|:-------------------------------------------------------------------------------------------------------:|:--------:|:--------:|:--------------:|:--------------:|:------------------------:|:------------------------:|:------------------------:|:------------------------:|:------------------------------:|
|                                          colocate sync      	                                           | 356.30 	 | 177.85 	 | 53.92        	 | 113.81       	 | 8h 36m                 	 | 17h 56m                	 | 1d 5h 6m               	 | 1d 16h 48m             	 | max: 0.3573<br>last: 0.2958  	 |
| `stream off policy pipeline`<br>(+fully async: trigger_parameter_sync_step= 4,<br>require_batches= 4) 	 | 231.34 	 | 128.47 	 | \            	 | 98.77        	 | 4h 25m                 	 | 9h 41m                 	 | 15h 2m                 	 | 1d 1h 53m              	 | max: 0.2844<br>last: 0.2604 	  |
|        `async stream pipeline with staleness samples`<br>(+staleness_threshold=0.5)            	        |    	     |    	     |       	        |       	        |            	             |            	             |            	             |            	             |               	                |
|        `async stream pipeline with partial rollout`<br>(+partial_rollout=True)                 	        | 150.63 	 | 33.14  	 | \            	 | 113.16       	 | 3h 13m                 	 | 6h 46m                 	 | 10h 53m                	 | 17h 22m                	 | max: 0.3521<br>last: 0.3094 	  |

> source data: https://wandb.ai/hou-zg-meituan/fully-async-policy-stream_stale_partial?nw=nwuserhouzg

### 128卡 stale 消融实验

在 `async stream pipeline with partial rollout` 模式下，我们验证 staleness 的设置对于训练效率的影响。
我们可以发现，staleness 越大，最终取得的收益越明显。
同时我们也注意到 staleness 取 0.3 和 0.5 的时间比较接近，原因是随着训练步数的增量，response 长度变化较大，训练出现了不稳定的问题。
后续还需要针对该问题进行进一步的分析和优化。

| staleness_threshold 	 | step  	  |  gen  	  | old_log_prob 	 | update_actor 	 | total time<br>100 step 	 | total time<br>200 step 	 | total time<br>300 step 	 | total time<br>400 step 	 |     acc/mean@1         	      |
|:---------------------:|:--------:|:--------:|:--------------:|:--------------:|:------------------------:|:------------------------:|:------------------------:|:------------------------:|:-----------------------------:|
| 0                   	 | 231.34 	 | 128.47 	 | \            	 | 98.77        	 | 4h 25m                 	 | 9h 41m                 	 | 15h 2m                 	 | 1d 1h 53m              	 | max: 0.2844<br>last: 0.2604 	 |
| 0.1                 	 | 171.30 	 | 58.17  	 | \            	 | 109.12       	 | 3h 53m                 	 | 8h 37m                 	 | 14h 25m                	 | 19h 59m                	 | max: 0.3542<br>last: 0.2979 	 |
| 0.3                 	 | 146.11 	 | 38.88  	 | \            	 | 103.22       	 | 3h 18m                 	 | 6h 49m                 	 | 11h 40m                	 | 17h 20m                	 | max: 0.3469<br>last: 0.2865 	 |
| 0.5                 	 | 150.63 	 | 33.14  	 | \            	 | 113.16       	 | 3h 13m                 	 | 6h 46m                 	 | 10h 53m                	 | 17h 22m                	 | max: 0.3521<br>last: 0.3094 	 |

> source data: https://wandb.ai/hou-zg-meituan/fully-async-policy-ablation_stale?nw=nwuserhouzg

### 128卡 7B require_batches 消融实验

在多次测试下，我们发现流式每次下发样本的数量会影响训练的response长度，进而影响训练时长，我们通过修改
`async_training.require_batches` 验证对与结果的影响。

| require_batches 	 | step  	  | gen  	  | old_log_prob 	 | update_actor 	 | total time<br>100 step 	 | total time<br>200 step 	 | total time<br>300 step 	 |     acc/mean@1         	      |
|:-----------------:|:--------:|:-------:|:--------------:|:--------------:|:------------------------:|:------------------------:|:------------------------:|:-----------------------------:|
| 1               	 | 203.47 	 | 30.88 	 | \            	 | 181.08       	 | 3h 31m                 	 | 8h 29m                 	 | 17h 36m                	 | max: 0.349<br>last: 0.326   	 |
| 2               	 | 158.72 	 | 26.32 	 | \            	 | 128.08       	 | 3h 35m                 	 | 7h 38m                 	 | 13h 57m                	 | max: 0.351<br>last: 0.3406  	 |
| 4               	 | 124.64 	 | 25.62 	 | \            	 | 95.06        	 | 3h 13m                 	 | 6h 46m                 	 | 10h 53m                	 | max: 0.3521<br>last: 0.3521 	 |

> source data: https://wandb.ai/hou-zg-meituan/fully-async-policy-ablation_require_batches?nw=nwuserhouzg

### 30B模型模式实验

我们在 Qwen3-30B-A3B-Base 模型上通过`async stream pipeline with staleness samples` 策略，相比于 colocate 方案取得了 1.7
倍的性能提升。值得说明的是，这距离异步方式所能带来的性能提升上限还有很大空间。首先，对比实验中使用的最大响应长度仅为
8k，这远低于此前实验的 20k 序列长度，因此 rollout 的长尾效应并不明显。其次，我们采用了极为倾斜的资源分配方案，rollout 使用了
96 张 GPU，而 trainer 仅使用了 32 张 GPU，这并不是最优的配置。在实验过程中，我们观察到当前的 verl 实现存在一些限制，比如要求数据必须能被
GPU 数量整除，这使得资源调整的灵活性受到影响。此外，随着异步训练和部署的加速，性能差距也在逐渐缩小。因此，未来我们将重点关注如何实现更灵活的资源分配和动态调整资源。

* 机器：H20
* 模型：Qwen3-30B-A3B-Base
* rollout长度：max_response_length : 8K tokens;
* 算法： GRPO
* 数据集： TRAIN_FILE: dapo-math-17k.parquet TEST_FILE: aime-2024.parquet
* Engine: vllm+Megatron
* rollout.n: 16
* ppo_mini_batch_size: 128
* test_freq: 20

* colocate sync:
    * step:400
    * train_batch_size: 512

* fully_async_policy
    * total_rollout_steps: 512*400
    * trigger_parameter_sync_step: 512/128 = 4
    * staleness_threshold: 0.5
    * partial_rollout: True

| Training Mode        | Resource Allocation | Step    | Gen    | Old Log Prob | Ref    | Update Actor | Total Time 100 Step | Total Time 200 Step | Total Time 300 Step | Total Time 400 Step | Acc/Mean@1                 |
|----------------------|--------------------|---------|--------|--------------|--------|--------------|---------------------|---------------------|---------------------|---------------------|-----------------------------|
| Colocate Sync        | 128                | 497.89  | 348.05 | 28.73        | 20.86  | 86.27        | 13h 36m             | 1d 3h 48m           | 1d 19h 4m           | 2d 11h 39m          | max: 0.3500<br>last: 0.3208 |
| Fully Async Policy   | 96:32              | 282.75  | 22.06  | \            | 50.05  | 206.63       | 6h 45m (2.01x)      | 14h 48m (1.88x)     | 1d 0h 9m (1.78x)    | 1d 10h 41m (1.72x)  | max: 0.3813<br>last: 0.3448 |

> source data: https://wandb.ai/hou-zg-meituan/fully-async-policy-30B?nw=nwuserhouzg

### checkpoint-engine参数同步消融实验
我们在Qwen2.5-Math-7B，Qwen3-30B-A3B和Qwen3-235B-A22B三个模型上测试了checkpoint-engine参数同步的单步参数同步耗时，使用的参数均为默认参数配置。实验均在H20机器上完成，并使用megatron训练引擎。
| model |  trainer rank 	  | rollout rank	  | checkpoint-engine 	 | total sync time 	 |
|:-----------------:|:--------:|:-------:|:--------------:|:--------------:|
| Qwen2.5-Math-7B   | 4        | 4       | False      | 0.12s      |
| Qwen2.5-Math-7B   | 4        | 4       | True      | 0.02s      |
|  Qwen3-30B-A3B     | 16        | 16       | False      | 15.76s   |
|  Qwen3-30B-A3B     | 16        | 16       | True      | 4.38s   |
|  Qwen3-235B-A22B    | 64        | 64       | False      | 58.57s   |
|  Qwen3-235B-A22B    | 64        | 64       | True      | 23.70s   |

### use_trainer_do_validate 实验测试
我们在Qwen2.5-Math-7B模型上测试了`use_trainer_do_validate`参数的影响。这个结果展示使用`use_trainer_do_validate=True`可以减少验证时间开销，并且训练器节点的空闲时间也减少了。

* Machine: H20
* Model: Qwen2.5-Math-7B
* Rollout length: max_response_length FSDP2: 10K tokens;
* Algorithm: DAPO
* Dataset: TRAIN_FILE: dapo-math-17k.parquet TEST_FILE: aime-2024.parquet
* Engine: vllm+FSDP2
* rollout.n: 16
* ppo_mini_batch_size: 32
* test_freq: 10

* fully_async_policy
    * total_rollout_steps: 512*400
    * require_batches: 4
    * trigger_parameter_sync_step: 4
    * staleness_threshold: 0.5
    * partial_rollout: True

|  training mode  | resource allocation | step  |  gen  | old_log_prob | update_actor | validate time | total time<br>50 step | acc/mean@2 |
|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|
| colocate sync      | 16  |  484.623  |  52.939	 |   0	 |   430.263   |  205.080  	 |     7h9m  	 |     22.6     |
| fully_async_policy | 8:8 |  489.953  |  52.622	 |   0	 |   435.874   |  95.699  	 |     7h2m  	 |     21.0    |


## 多轮工具调用

参考 **recipe/retool** 和 **ToolAgentLoop**，我们为 **fully_async_policy** 实现了支持partial rollout的多轮工具调用循环 *
*AsyncPartialToolAgentLoop**。

### 核心设计

`AsyncPartialToolAgentLoop` 继承自 `ToolAgentLoop`，其核心是适配了 `fully_async_policy` 的异步训练模式。当
`partial_rollout=True` 时，Rollouter 在与 Trainer 同步参数前会中断正在进行的生成任务。`AsyncPartialToolAgentLoop` 能够：

1. **中断任务**: 响应中断信号，保存当前的生成状态。目前，中断会发生在GENERATING过程中，或其他状态结束后；
2. **恢复任务**: 在参数同步完成后，从保存的状态恢复，继续执行，而不是从头开始。

### 使用方法

`fully_async_policy`多轮与工具调用的RL训练与 `recipe/retool` 类似，通过在配置文件中指定 `multi_turn` 相关配置来启用。

1. **SFT 阶段**: 首先，需要对模型进行 SFT训练，使其具备遵循工具调用格式指令的能力。
2. **配置启用**: 在 `fully_async_policy` 的训练配置中，设置以下参数:
   ```yaml
   actor_rollout_ref:
     rollout:
       multi_turn:
         enable: True # 在fully_async_policy模式下将默认使用AsyncPartialToolAgentLoop
         # 其他 multi_turn 相关配置
   ```
3. **配置async参数**: 为提高效率，在启用多轮工具调用时，同时开启 `partial_rollout`和`staleness_threshold`：
   ```yaml
   async_training:
     partial_rollout: True
     staleness_threshold: 0.5
     # 其他async参数
   ```
4. **example**: 参考`recipe/fully_async_policy/shell/dapo_7b_async_retool.sh`

### 实验结果

为验证 `fully_async_policy` 在多轮工具调用任务中的性能，我们将其与标准 `colocate` 同步模式进行了对比。实验具体设置如下。

* **SFT模型**: 实验基于 `Qwen2.5-7B-Instruct` 模型，使用`ReTool-SFT`数据集训练6个epoch；
* **RL算法**: DAPO
* **数据集**:
    * 训练集: `DAPO-Math-17k`
    * 测试集: `aime_2025`
* **资源与模式对比**:
    * `colocate sync`: 32卡 H20
    * `fully_async_policy`: 16卡 Trainer + 16卡 Rollouter
* **关键配置**:
    1. **工具调用配置**:
        * `multi_turn.enable: True`
        * `multi_turn.max_user_turns: 16`
        * `multi_turn.max_assistant_turns: 16`
        * `multi_turn.tool_config_path: recipe/retool/sandbox_fusion_tool_config.yaml`
    2. **`colocate sync`配置**:
        * `ppo_mini_batch_size: 16`
        * `train_batch_size: 64`
    3. **`fully_async_policy`配置**:
        * `ppo_mini_batch_size: 16`
        * `trigger_parameter_sync_step: 4`
        * `require_batches: 1`
        * `staleness_threshold: 1`
        * `partial_rollout: True`

|    training mode   	| Resource allocation 	|   step  	|   gen   	| old_log_prob 	| update_actor 	| total time<br>100 step 	| total time<br>200 step 	|   aime_2025<br>acc/mean@30  	|
|:------------------:	|:-------------------:	|:-------:	|:-------:	|:------------:	|:------------:	|:----------------------:	|:----------------------:	|:---------------------------:	|
| colocate           	| 32                  	| 375.47  	|  228.03 	| 35.19        	| 111.84       	| 9h 46m                 	| 22h 28m                	| start:0.1078<br>last:0.2056   	|
| fully_async_policy 	| 16: 16              	|  221.36 	| 40.59   	| \            	| 179.58       	| 6h 19m<br>(1.55x)      	| 14h 4m<br>(1.60x)      	| start:0.11<br>last:0.2044 	|

> source data: https://wandb.ai/hou-zg-meituan/fully-async-policy-multiturn-tool?nw=nwuserhouzg
