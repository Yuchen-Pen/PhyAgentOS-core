# RoboDojo＋纯Pi0.5：接口映射与最小实现清单

> 当前实现进度（2026-11-01）：阶段3/4实现与正式安装完成，真实成功单回合和正常Core自动结算已验证；停止／连续case实机验证记录见[验证报告](robodojo-pi05-validation.md)，使用入口见[README](../example/robodojo-benchmark/README.md)。下文保留阶段1设计依据和当时缺口，不能把历史“尚未实现／尚未运行”当作当前状态；未修改公共Core核心契约。

阶段1交付，2026-10-31。本文是依据固定源码形成的实施依据；后续阶段3已实现原生Skill生成/安装和离线测试，当前入口见[实现说明](../example/robodojo-benchmark/README.md)，不代表225真实评测通过。

目标：环境和模型配置就绪后，使用者可以通过PAOS调用已支持的平台与策略，安排评测、查看进度并取得结果。首期使用RoboDojo＋纯Pi0.5，不增加外层VLM动作监督或在线Lesson更新。

## 1. 基线、来源与责任

| 对象 | 固定版本／来源 | 本期用途 |
|---|---|---|
| 公共Core | `feature/benchmark@a606ddfc193fc87968297a7901220c251f6de492`，个人分支`robodojo-pi05` | 复用任务、case、执行身份、调度、对账及报告 |
| 刘意评测链路 | [PhyAgentOS-Repo](https://github.com/liuy2839/PhyAgentOS-Repo/tree/d827bc35b81f6625b10464660bb71ead3937f953) | 复用原生Session工具面、RoboDojo仿真适配和策略节点收发逻辑 |
| 师兄交付包 | ZIP SHA256 `878fcc48683603ea01785658f2befb225cd16a5ffb000d3ff41d1e4f951ac81a`，`robodojo/paos_skill_robodojo_0.2.6/profiles/pi05/` | 复用Pi0.5权重锁、环境准备与`Pi_05`启动约定；旧Action端点不作为新Session协议 |
| Pi0.5权重 | HF dataset `RoboDojo-Benchmark/RoboDojo`，revision `43dacb13d3051a9ccd421f4e7e08e4e3eb29dfab`，checkpoint `59999` | 原大小／SHA256锁校验，权重与源码分开存放 |

用户最新分工：首期接入中尚未明确归属的工作，默认由宇琛这条线承担，包括必要的构建、配置和适配修正；先自行查代码和资源。不能查明的事实仍明确记录，不以默认责任代替事实核实。改动在个人副本内完成；共享服务与刘意个人环境不因这项分工而获得写入授权。

最初34份关键文件仅用于审阅；当前已补齐上述commit的完整checkout供实际构建。审阅快照保持原件，不作为部署包。

## 2. 调用链与复用边界

```text
个人部署配置＋固定case清单
  → 公共Core：冻结计划，分配单个worker和case身份
  → 我方私有benchmark控制适配器：开始／查询／停止／对账
  → 刘意原生工具：benchmark.describe(Query)、benchmark.run(Session)
  → bench_tool → bench_data(RoboDojo) ↔ policy(Pi_05)
  → 原生ToolResult＋逐episode JSON＋日志／视频
  → 我方核验、归档、生成带Core身份的快照和轨迹
  → 公共Core结算case，汇总报告
```

这里有两个不同的`benchmark.run`：外层管理整份Core计划，内层运行一次原生评测。它们使用不同的绑定与client；适配器调用固定的原生Gateway，不能递归调用自己的控制工具。

公共Core要求一个带`benchmark.yaml`的独立控制Skill，声明三个不同工具：Session、Query、Action，以及`root/episode/receipt/history`四种查询视图。刘意原生Skill只有describe和run，不能直接作为Core控制Skill使用。新增私有控制Skill连接两边，保留原生Skill及工具协议。

复用Core的`BenchmarkService`、`BenchmarkClient`、`PrivateBenchmarkTransport`、runner、supervisor及报告核验。已有真实适配示例`example/robotwin-rsi-benchmark/`可参考控制Skill安装和绑定方式；其中VLM、候选动作闸门、RoboTwin专有结果字段和逐case杀进程的停止方式不适用于本期。

## 3. 配置与身份字段对应

下表是我方配置的实施约束；标为“新增”的名称尚未成为公开API。

| 输入／身份 | 原生对应 | 处理要求 |
|---|---|---|
| 平台、策略 | describe中的`benchmark/suite/policy_id` | 配置固定RoboDojo／`Pi_05`／joint；启动后核对实际返回值，不能只检查配置文件 |
| case_key | 原生没有同名字段 | Core清单内唯一；我方保存与原生invocation的关联 |
| environment.native_task_id（新增） | `task_ids=[id]` | 避免与Core字符串`task_id`混淆；从实际describe选择并核对任务名 |
| environment.init_state_id（新增） | `init_state_ids=[id]` | 按该任务的能力描述校验范围；实际布局还须读取episode的`metrics.layout_id` |
| 一条Core case | `num_runs=1`，单任务、单初态 | 一次原生Session最多对应一个评测回合；重复实验展开成独立case，不把多集批次混作一集 |
| environment.layout_seed（新增） | 原生参数`seed` | RoboDojo用它选择布局目录；不是策略RNG seed |
| `world_epoch` | 无原生同名字段 | Core逻辑执行代次；不能冒充物理环境或策略的随机种子 |
| Core七项scope | `task_id/run_id/subtask_id/attempt_id/worker_id/world_epoch/binding_digest` | 适配器原样保存；原生返回的`outputs.run_id`另存为native invocation身份，绝不覆盖Core run_id |
| 外层request_id、payload_digest | 原生启动接口未暴露等价幂等键 | 在适配器持久化请求、意图、结果和receipt；原生启动回包丢失时按第6节处理 |
| max_steps | 原生参数及结果`max_steps` | 当前adapter reset会改成实际环境`step_lim`；不能承诺传入值就是执行上限，见第7节 |
| episode_timeout_s | Core整回合墙钟预算 | 超时发起stop并继续对账；不等于原生步数上限，也不表示立即停止 |
| worker／并行 | 原生run `max_concurrency=1` | 首期一个明确授权worker、`parallelism=1`、`rounds=1`；不可对同一仿真器并发派发 |
| GPU、Python、端口、源码／资产／权重、结果／视频路径 | 节点环境与bench配置 | 我方配置显式指定，输出只写个人目录；不沿用`/data0/liuyi`或固定GPU编号 |

Core当前`max_decisions`必须≥1，不能设0。纯Pi0.5路径不发送`ready`快照，而是`running → terminal`；runner只在`ready`调用`decide`，因此无需为零外层模型调用先修改Core计划契约。实施中设置会立即报错的`decide`保护函数，并测试调用次数为0；Pi0.5自身的推理照常进行。

## 4. Pi0.5观察、动作与episode边界

| 内容 | 已有契约 | 我方接入要求 |
|---|---|---|
| 模型加载 | `XPolicyLab.policy.<POLICY_NAME>.model.Model(deploy)` | `POLICY_NAME=Pi_05`，使用其openpi环境；显式传入checkpoint／归一化资产，不改G05/ReKep默认行为 |
| 观察 | `proprio_state`＋head／left_wrist／right_wrist图像＋instruction | 关节状态、末端姿态按源规格解包，三相机对应正确，检查图像类型、尺寸与通道；ARX X5当前观察规格28维 |
| 动作 | joint，按`get_robot_action_dim_info`生成分量名及顺序 | ARX X5当前14维；名字、顺序、数值有限性和维数须全部匹配，不截断／补零掩盖错误 |
| rollout | 新完整观察触发，chunk内每步update_obs | tick只作心跳，不能按tick额外推理或跳过动作；chunk末动作携带现有标记，让原生结束检查一致 |
| reset／instruction | 同一`policy_command`通道，跨通道到达可能无序 | 保留足够队列深度；每集清图像、新鲜帧集合、旧chunk与模型状态，不混用上一集指令／动作 |
| 模型随机性 | 现有部署seed与布局seed不是同一概念 | 单独核实Pi_05／openpi的RNG与重置行为，在复现说明记录；未核实前不声称逐位可复现 |

优先复用刘意通用策略节点的观测驱动循环及旧Pi0.5 profile的模型启动约定，增加Pi0.5专用薄启动器/profile。暂不为接入而重写策略SDK、仿真控制或Pi0.5算法。

## 5. 状态、结果与归档

原生Session的HTTP 202仅表示接纳。已知invocation用status/result查询；result返回202表示仍待完成。Session通过`stop`结束，不走Action的`cancel`。

工具结果有三层，必须分别保留：

1. `ToolResult.status`：工具执行结论。
2. `ToolResult.outputs`七字段：`run_id/status/completed_episodes/total_episodes/success_rate/result_file/failure_summary`；outputs.status还可能是partial。
3. `result_file`指向的`benchmark_execution_result_v1`：逐episode判定、错误、步数、布局、指标及视频索引。工具执行succeeded不等于机器人任务成功。

| 已核事实 | Core outcome／处理 |
|---|---|
| 唯一目标episode，success=true，termination=success，无error_code，其他身份／计数／预算校验通过 | success，二值native_score=1；仍保留原生primary_metric及全部metrics |
| 唯一目标episode，无基础设施错误，原生failed／partial，或在已记录有效horizon结束为timed_out | failure，二值native_score=0；保留原生termination，partial不改写为成功 |
| stop已在边界落地，目标episode尚未完成 | cancelled；保留原生stopped及已执行证据，不伪造失败回合 |
| 请求stop后当前episode正常完成 | 按真实episode判定success／failure；额外记录停止请求及实际落地时刻 |
| 模型／通信／仿真报错、原生error_code，或已确认停下但结果缺失／不可信 | infra_error；不能混算为机器人任务失败 |
| 启动接纳与否不明、身份不符、状态无法确认、执行是否停止仍未知 | state=unknown，保留worker并对账；不能以terminal或stopped掩盖未知执行 |

核验至少包括：schema、benchmark/suite/policy与运行配置、原生invocation关联、任务/初态/run_index=0、恰好一个计划episode、计数与成功率一致、错误字段、实际布局和有效步数上限、结果路径归属及原始文件哈希。完整结果顶层没有Core scope，也没有直接序列化invocation_id；invocation由接纳回包、outputs.run_id及该Session的结果文件名关联，再写入我方带scope的索引。不能虚构原生结果本来含这些字段。

归档以Core case/attempt为单位保存配置、源码与模型身份、原生invocation、原始ToolResult、原生JSON只读副本及SHA256、快照/receipt/history、停止证据和日志/视频索引。结果路径须位于配置允许的个人输出根内，不能信任任意返回路径或把stub结果当真实评测。原始JSON和指标不覆盖。

## 6. 开始、查询、停止与异常恢复

1. **部署／冻结**：安装原生Pi0.5 Skill并核实际能力；用真实SkillInstaller安装独立私有控制Skill，生成安装器回执。按Core现有验证契约冻结case与binding。不能手写回执绕过校验。
2. **开始case**：检查唯一worker空闲、scope和case参数；先持久化启动意图，再提交一次原生Session，收到202后保存native invocation。相同request_id＋相同payload返回原receipt；同ID不同payload拒绝。
3. **查询**：已知invocation才查询原生status/result；未终结时返回running，结果通过核验后才返回terminal。快照有单调版本、完整scope和history_digest，内容不变不重复扩张轨迹。
4. **停止**：`episode.stop`接纳receipt与“已经停止”分开。对未结束的原生Session请求stop，轮询其实际终态／结果。停止在当前episode边界落地，可能持续较久；未证实落地时`stopped=false`，不得释放worker或启动下一case。
5. **结算与下一case**：确认该原生Session已无后续动作、最终快照terminal/stopped、历史完整和receipt已对账后，让Core结算并复用空闲服务运行下一case。常驻仿真和策略进程可保留；这里的stopped指该次执行结束，不要求逐case杀掉整个服务。
6. **结束整份计划**：所有子case结算后停止外层控制Session，满足Core root查询、start_digest、history和invocation终态检查。关闭个人部署的生命周期与停止单case分别实现，只操作有归属证据的本人进程。

当前`ForgeToolClient.start_session`只有arguments/caller_id，没有调用方指定invocation_id或幂等键的接口。原生endpoint对同一个执行key可去重，不代表重新调用Gateway会获得同一个key。若202回包丢失，适配器必须记录unknown并阻止再次派发；现有源码不足以证明可按caller_id查回执行，不能盲重试。自动找回能力作为我方后续增强；首版可在个人服务有明确归属、完成停机核验后人工恢复，未知记录继续保留。已知invocation的断连可继续查询对账。

适配器状态与receipt需要持久化；进程重启时拒绝自动重发未结算启动。Core已有对账与watcher，先适配其接口，不另写一套任务调度或放宽settled校验。

## 7. 已查出的具体缺口与归属

| 缺口／风险 | 本期处理 | 责任与验收位置 |
|---|---|---|
| 云端只有g05／rekep，没有Pi0.5 profile | 个人实现Pi0.5薄启动器、profile和准备入口 | 我方；阶段3 |
| 原生两工具与Core三工具契约不同 | 新增独立控制Skill与生命周期／结果适配 | 我方；阶段4 |
| `reset_episode`无条件把cfg.max_steps改成env.step_lim，结果也随之更新 | 显式区分原生task horizon与工程预算；默认保留原生评测horizon，传入工程上限不得被静默忽略。实测前在我方副本落实并测试请求／实际值和超时分类 | 我方；阶段3/4。未解决前不宣称步数预算可靠 |
| 原生结果可能批次succeeded但个别episode有error_code；落盘失败也可能未转failed | 强制读逐episode与原始文件，错误／缺文件归基础设施问题 | 我方；阶段4 |
| 原生启动回包丢失后缺少已证实的自动找回接口 | unknown＋保留worker＋禁止自动重发；归属明确后恢复，记录限制 | 我方；阶段4 |
| README主build入口遗漏ReKep产物，而另一个入口已有；deploy相对路径／安装步骤不一致 | 首期构建入口只准备实际Pi0.5依赖并完整安装，真实校验内容哈希；不把ReKep整套修复设为本期前置条件 | 我方；阶段3 |
| `/data0`示例路径、固定GPU、环境／资产权限和Pi0.5环境未就绪 | 个人配置与预检；不可读资源另找授权来源或个人准备 | 我方；阶段2，未核实项明确保留 |
| 未找到可读范围内225真实PAOS完成证据 | 以本人阶段5固定验证清单生成新证据；同伴后续结果可读来复用 | 我方；阶段5，不重复等待其承诺的代码交付 |

## 8. 最小代码落点及实施顺序

以下为阶段1确定的代码落点；阶段3实际使用prepare.py生成Pi0.5 Skill及其profile，三个Python节点直接通过独立解释器启动，仅Gateway使用节点制品，避免额外打包薄壳。实施不扩大到公共框架重写。

| 计划落点 | 作用／复用依据 | 完成阶段 |
|---|---|---|
| `example/robodojo-benchmark/config.example.json`、`preflight.py` | 个人路径／环境／GPU／端口／模型回执和单case配置；读取describe并核对 | 2/3 |
| `example/robodojo-benchmark/skill-src/robodojo-bench/` | Pi0.5 launcher＋profile的dataflow/gateway/bench_config及manifest；从固定刘意基线派生并记录来源 | 3 |
| `example/robodojo-benchmark/prepare.py` | 补齐固定源码、构建所需节点、调用现有安装器与模型／策略准备入口；大文件在源码目录外 | 3 |
| `example/robodojo-benchmark/native_client.py`、`runtime.py` | 复用ForgeToolClient调用原生Session；管理本人部署生命周期和归属；按需承载horizon补丁及来源记录 | 3/4 |
| `example/robodojo-benchmark/controller.py` | 独立控制Skill三个工具、四种视图、scope、幂等receipt、持久化和stop对账 | 4 |
| `example/robodojo-benchmark/results.py` | 原生单episode核验、outcome映射、原件与身份归档 | 4 |
| `example/robodojo-benchmark/control_bundle.py`、`run.py` | 复用SkillInstaller／PrivateBenchmarkDeployment／AgentTaskCoordinator；零外层模型调用、单worker运行清单 | 4 |
| `tests/test_robodojo_benchmark.py`、示例README | 必要行为测试、配置到取结果的使用方法、真实验证证据索引和限制 | 3/4/6 |

暂不计划修改`PhyAgentOS/agent/benchmark/`和`PhyAgentOS/forge/benchmark/`核心契约。若有行为测试证明现有能力不足，先给出失败证据和最小兼容修正，再调整清单。

阶段3先让Pi0.5通过原生工具面可用，再做阶段4的Core整计划适配；离线工程工作不必等待全部模型下载。阶段5实际任务、初态、重复、时间／步数预算和GPU安排仍需在实验清单中确定，本文的“每Session一回合”是映射设计，不代表已授权任意数量实验。

## 9. 验证清单

| 检查 | 必须观察到的行为 | 证明范围 |
|---|---|---|
| Pi0.5配置／动作契约 | 模型路径显式、相机正确、分量名/顺序/维数匹配；reset不残留旧chunk | 离线夹具＋阶段5真实输入输出 |
| 自主运行 | running→terminal，decide调用0次，不需要VLM/API | 真实Core＋模拟原生Gateway的行为测试 |
| 单次派发／重复请求 | 一case一次启动；同request重放不新增执行，冲突拒绝 | 行为测试 |
| 结果 | success、partial、timed_out、error_code、缺失、身份不符分别处理 | 固定结果夹具；标清合成材料 |
| 停止 | stop接纳后仍running时不释放worker；边界确认后才结算 | 行为测试＋非破坏性实机证据 |
| 断连／重启 | 已知invocation可查回；未知接纳禁止重发；跨进程未结算记录保留 | 行为测试 |
| 连续case | 第一回合停止／结算后第二回合新建runner，服务可复用，scope/指令/chunk无串集 | 行为测试＋阶段5真实连续执行 |
| Core整计划 | root/episode/receipt/history闭合，固定清单全覆盖，成功率与异常分类一致 | 现有Core核验与报告，不以程序退出0替代 |

## 10. 主要源码定位

Core相对路径均指本仓库固定基线：

- [binding.py](../PhyAgentOS/forge/benchmark/binding.py)：`read_contract/install_receipt`，三工具、四视图、安装后不可变校验。
- [service.py](../PhyAgentOS/forge/benchmark/service.py)：`freeze/enable_private_adapter`，验证契约与绑定。
- [contracts.py](../PhyAgentOS/forge/benchmark/contracts.py)、[private.py](../PhyAgentOS/forge/benchmark/private.py)：case/快照、计划约束与私有适配权限。
- [runner.py](../PhyAgentOS/agent/benchmark/runner.py)：`run/stop_and_check`，只在ready推理及结算前停止／历史核验。
- [supervisor.py](../PhyAgentOS/agent/benchmark/supervisor.py)、[client.py](../PhyAgentOS/forge/benchmark/client.py)、[evidence.py](../PhyAgentOS/forge/benchmark/evidence.py)：派发、scope/receipt、未知执行对账与报告。
- [tool_client.py](../PhyAgentOS/forge/tool_client.py)：原生工具查询及Session的start/status/result/stop。
- [RoboTwin示例](../example/robotwin-rsi-benchmark/experiment.py)：控制Skill安装与Core绑定的参考，不直接复用其VLM执行方式。

刘意源码均指上文固定commit，主要证据：

- `forge_robodojo_eval/robodojo/bench_tool/descriptor.py`：`BENCH_DESCRIPTOR`，Query/Session、max_concurrency=1、stop边界。
- 同目录`endpoints.py`：`normalize_plan/BenchSessionEndpoint`；`result_shape.py`：七字段及stopped/partial区别。
- `forge_robodojo_eval/robodojo/_core/session.py`：`capabilities_payload/BenchSession`，能力描述、每次新runner、保持adapter。
- 同目录`result.py/runner.py`：原生schema、逐episode错误、超时、落盘及invocation结果文件名。
- `forge_robodojo_eval/robodojo/adapters/robodojo_adapter.py`：`num_init_states/reset_episode/step`，布局seed、原生奖励注册、horizon覆盖、布局指标。
- `forge_robodojo_eval/robodojo/policy_node.py`：模型加载、观察／动作规格、chunk节奏与重置；`skill-src/robodojo-bench/launchers/policy_node.py`及g05 profile为部署派生依据。

阶段1验证方式是逐项静态核对上述接口、真实字段与调用条件。后续实现、回归与实机验证已完成到本页顶部指向的报告状态；单一任务的少量接入样本不等于总体成功率。
