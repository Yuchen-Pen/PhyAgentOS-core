# RoboDojo＋纯 Pi0.5 评测接入

本示例让 PAOS 公共 Core 管理 RoboDojo＋Pi0.5 的案例清单、运行状态、停止与结果。
Pi0.5 自主产生机器人动作，不请求外层 VLM、不调用付费模型 API、不在线更新 Lesson。
复用刘意固定版本的工具面和策略收发逻辑；所有补丁只作用于个人生成副本，未修改公共 Core 契约。

[接口设计与来源](../../docs/robodojo-pi05-integration.md) · [实机验证与限制](../../docs/robodojo-pi05-validation.md)

## 范围与依赖

当前支持单 worker、`rounds=parallelism=max_decisions=1`，一条 Core case 对应一次原生单回合 Session。
案例可以有多条，按清单顺序执行；重复环境要使用不同的 `case_key`。
这是 Python/终端入口，尚未提供自然语言或网页一键配置界面。

| 依赖 | 已验证身份／要求 |
|---|---|
| Core 基线 | `feature/benchmark@a606ddfc193fc87968297a7901220c251f6de492` |
| 刘意工具／数据面源码 | `liuy2839/PhyAgentOS-Repo@d827bc35b81f6625b10464660bb71ead3937f953`，完整、干净 checkout |
| RoboDojo | `f57e1633ffed59761bc8e568cfe7530c4b33381c` |
| XPolicyLab | `5d0be1499634d3176258c8f2e31f5fe816497b0a` |
| cuRobo | RoboDojo 固定子模块 `d17b54ce32cba095c0b000c4c58777075d11de0e`，安装 `.[cu12]` |
| Isaac Sim／IsaacLab | 实机为 5.1／0.54.2；本部署复用只读 IsaacLab 安装，其源码没有 Git 身份，不能声称完全独立复建 |
| 模型／资产 | HF dataset `RoboDojo-Benchmark/RoboDojo@43dacb13d3051a9ccd421f4e7e08e4e3eb29dfab`，checkpoint `59999` |
| Dora／Gateway | Dora CLI 与 dora-rs 0.4.1；Gateway 1.1.1，须包含静态资源，不能只有入口程序 |
| 主机 | Linux、可用 NVIDIA CUDA 与 Vulkan／Kit 渲染；实机为 RTX PRO 5000、驱动 580.126.20 |

将控制、仿真、工具面和策略解释器分开配置。控制环境 Python≥3.11，安装本仓库的运行依赖。
仿真与 Pi0.5 各用兼容环境，不能把 Core 的 websockets 等版本直接混入仿真环境。
仿真环境还需 Isaac Sim/IsaacLab、Torch、msgpack/msgpack-numpy、transforms3d、pyglet、open3d、
websockets 14.2、shapely、omegaconf、hidapi、prettytable、Dora 和已固定 cuRobo。
具体实测版本见验证报告，个人 overlay 不会使同伴基础环境成为可写目标。

## 1. 配置与资源准备

在本目录执行，以下路径均替换为本人绝对路径；含空格时保留引号。
`python` 指控制环境解释器，三个节点分别使用配置里的解释器。

```bash
cp config.example.json "/personal/path/deployment.json"
```

填写 `work_root/source_repo/robodojo_root/models_receipt/gateway_archive/python`，按资源安排填
`simulation_gpu/policy_gpu`（单卡序号或 UUID）和个人端口。两进程可共用一张有足够显存的卡；
这里的 JAX 显存比例默认 0.3。`null` 表示尚未选择 GPU，不能用于真实启动。
`work_root` 只写本人目录；改变配置或计划时建议使用新运行根，保留旧日志。

固定源码后，初始化 RoboDojo 的 cuRobo 子模块，在个人仿真环境安装它；运行 RoboDojo 官方
`utils/update_embodiment_config_path.py`，从模板生成本机 `Assets/Robots/x5/curobo.yml` 等配置。
纯 joint 策略的场景初始化也会构造规划器。不要照搬旧机器绝对路径或运行修改共享环境的整套安装脚本。

模型锁原字节 SHA256 为 `a7040c52335750a50131a50cff9169ce22dd18b9fe9d545d8bc470512e452cc1`，
本目录附有 `pi05-checkpoint.lock.json`，仅列推理必需的 17 个文件（约 11.59 GiB）。
`prepare_models.py` 从官方 HF 地址按固定版本续传，逐文件验大小及 SHA256，全部通过才写 ready 回执。
已有同结构缓存可用 `--check-only` 验证，失败不会留下 ready 状态。

```bash
python prepare_models.py "/personal/path/model-cache"
python prepare_assets.py --write-catalog "/personal/path/assets-catalog.json"
python prepare_assets.py "/personal/path/assets-catalog.json" "/personal/path/asset-cache" \
  --reuse "/read-only/existing/RoboDojo" --workers 8
```

资产约 38.44 GiB／15365 文件。`--reuse` 可省略；提供时只读源，复制到本人目录后校验 LFS SHA256，
不修改源权限或内容。把本人 `RoboDojo/Assets` 链接到已验证的 `asset-cache/Assets`，
不覆盖未知目录。将配置的 `models_receipt` 指向 `model-cache/pi05-models.json`。
这些命令会下载大文件；代码提交不含模型、资产、缓存或视频。

## 2. 生成、安装与策略准备

```bash
python prepare.py "/personal/path/deployment.json" \
  --output "/personal/path/bundles/v1/robodojo-bench-pi05"
python install.py "/personal/path/bundles/v1/robodojo-bench-pi05"
python prepare_policy.py "/personal/path/deployment.json" \
  --uv "/personal/path/bin/uv" --python-install-dir "/personal/path/python"
python preflight.py "/personal/path/deployment.json"
```

输出目录最后一级必须是 `robodojo-bench-pi05`；已有输出不会覆盖。
安装使用 Core 正式打包器、NodeInstaller、SkillInstaller 与环境构建器，不能手写安装回执。
Dora 直接用不同解释器执行三个 Python 源码节点，只有 Gateway 需要二进制制品。
`prepare_policy.py` 按 openpi 原 `uv.lock` 安装模型环境，并验证 CPU 配置、14 维归一化／动作、
28 维观察、跨环境 Arrow 消息和 tokenizer。ready 回执不代表真实 GPU 推理或场景验收。

原 openpi 锁中的数据辅助库存在 NumPy 约束冲突，个人兼容层使用 lerobot 0.3.2、datasets 3.6.0、
gymnasium 0.29.1、rerun-sdk 0.22.1；保留模型 JAX/Flax/Torch 版本。仿真和策略的 NumPy/PyArrow
可以不同，依靠实际消息往返验证，不能仅看导入成功。

Gateway 归档根目录必须只有常规可执行文件 `gateway`。
225 旧制品缺静态资源，已从官方 `adapter-forge-gateway@c3fb206a16ab970486cd1e523e70a5c7a4f73cbb`
及原 `uv.lock` 重建。PyInstaller 要包含 `src/forge_gateway/resources`（使用绝对 `--add-data` 源路径），
再把二进制装进上述归档。已有合格制品可复用，文件名不能代替内容哈希。

## 3. 启动、状态与执行

使用官方 Dora 0.4.1 CLI，先核发布 SHA256，然后配置三个空闲私人端口；Gateway 使用部署配置端口。
不要调用会复用默认共享服务的 `dora up/destroy`。

```bash
python service.py "/personal/path/deployment.json" configure \
  --dora-binary "/personal/path/dora" --ports 19032 19033 19034
python service.py "/personal/path/deployment.json" start
python service.py "/personal/path/deployment.json" status
python run.py "/personal/path/deployment.json" "/personal/path/plan.json"
python run.py "/personal/path/deployment.json" "/personal/path/plan.json" --execute
```

从 `plan.example.json` 复制案例清单并填写预算。无 `--execute` 只预览，不派发。
执行前必须是本配置管理的 ready 服务；程序读取真实 describe，核对任务 ID/名称、初态范围、策略和布局，
不接管无本人运行记录的 Gateway。启动冷检查上限 900 秒，单回合墙钟预算由计划指定。

`step_limit_mode=native` 使用任务原生 horizon；`cap` 用配置上限与原生 horizon 的较小值。
原始结果同时保留 requested/native/effective 值，不静默覆盖。达到目标可提前结束，700 是上限而非必须跑满。

单卡可见时 CUDA 使用逻辑 `cuda:0`，Kit 使用物理 NVIDIA 序号。
启动器先按 UUID 核 CUDA/Vulkan，再显式设置 Kit `activeGpu`；reset 后复核。
CPU 仿真且 Fabric 关闭时，显式启用 `physics/updateToUsd`，否则关节能动而相机可能冻结。
`render_state_sync` 记录实际通道和显卡；录像末帧复用原生已采集观察，不增加物理或渲染步。

225 缺少的 EGL/Xt 等用户态图形库仅解包在本人 `native-libs/root/usr/lib/x86_64-linux-gnu`，
`receipt.json` 保存来源和哈希；迁机须重新核依赖。Kit 使用个人 portable root，缓存不写基础安装。

## 4. 正常停止与重新执行

在执行 `run.py --execute` 的终端按 **Ctrl+C**，或向该本人进程发送 SIGTERM。
信号转为公共 Core 的 `cancel_task`：关闭后续派发、请求原生 stop，并等待权威边界对账。
重复信号不强杀服务。程序保留 `stop-request.json`、`stop-acknowledgement.json` 和最终报告。

**原生 stop 在当前 episode 边界生效，不是立即急停。** 接纳时仍可能执行动作，worker 继续占用；
当前回合可以真实成功，而外层整批因用户停止显示 cancelled，二者分别记录。
超过对账预算仍不确定时报告 blocked，不冒充已停，也不重发新任务。

只有所有 root／case 均已证明结束，才允许同配置、同计划开始新批次：

```bash
python run.py "/personal/path/deployment.json" "/personal/path/plan.json" --new-batch --execute
```

这会保留历史，不重新加载常驻模型。完整计划结束后关闭个人服务：

```bash
python service.py "/personal/path/deployment.json" stop
python service.py "/personal/path/deployment.json" renew
```

`stop` 核 UID/PID 启动身份、参数、监听 socket 和原生流，只操作本人记录进程；active/unknown 时拒绝关服务。
`renew` 只归档已经死亡的旧进程回执，不启动。下一次再 `start`。
异常断连后，默认重启拒绝自动派发；已知 invocation 可用以下入口对账：

```bash
python reconcile.py "/personal/path/deployment.json" "/personal/path/plan.json" --close-root
```

它只处理原执行，不重发 start，不改写原始 Core 报告。尚有控制进程占用 journal 时先恢复该进程，
不要同时运行第二个 owner。原生启动回包丢失且无法确认是否接纳时，保留 unknown 和 worker，不能盲目重试。

## 5. 查看结果

| 输出位置（相对 work_root） | 内容 |
|---|---|
| `core-runs/run_*/configuration.json` | 固定配置、案例、源码哈希、安装包、环境／模型身份 |
| `core-runs/run_*/core-report.json` | Core 整批和子案例状态、是否 settled、外层模型调用数 |
| `control/state.json` | 当前 active、root、原生 invocation、停止请求和回执；由唯一 owner 写入，勿手改 |
| `control/episodes/*/result-index.json` | Core 七项 scope、原生结果、JSON／视频 SHA256、case 身份 |
| `results/<suite>_<invocation>.json` | 原生 JSON 原件，包含成功、步数、错误和实际预算 |
| `videos/<invocation>/` | 该次 Session 专属录像，不与下一案例混放 |
| `runtime-logs/`、`forge-runtime/.../out/<flow-id>/` | 服务、节点和图形检查日志 |

Gateway 正常终态是 `phase=completed`，ToolResult 的执行结论是 `status=succeeded`；
机器人是否成功以逐 episode 的 `success/termination/error_code` 为准。
Core `task.status=succeeded` 表示评测已完整结算，不能独自证明机器人成功。
缺依赖、相机失败、缺结果或身份不符归 infra_error，不算机器人失败率。

单回合录像通常为 `num_steps+1` 帧，三相机按 head／left_wrist／right_wrist 横向排列。
既要检查 JSON，也要核动态画面和终帧；单张正常截图无法证明相机随动作更新。

## 验证与限制

```bash
python -m pytest -q tests/test_robodojo_profile.py tests/test_robodojo_assets.py \
  tests/test_robodojo_models.py tests/test_robodojo_benchmark.py \
  tests/test_robodojo_service.py tests/test_robodojo_render_state.py \
  tests/test_robodojo_refactor.py
```

以上命令在仓库根目录执行；服务归属测试需要本地监听 socket，但不启动模型、GPU 或付费 API。
模拟 Gateway 结果只证明工程行为。真实回合和停止／连续案例证据另见[验证报告](../../docs/robodojo-pi05-validation.md)。
目前验证范围是单一瓶子任务／初态／布局，不建立多任务成功率或策略泛化结论；没有在线 Lesson、外层 VLM 或性能收益对照。

状态、补丁及公共校验的集中实现见 [Forge 实现对齐说明](../../docs/robodojo-forge-unification.md)。
本轮 95 项离线测试通过，真实源码的 45 个原生文件与已验证版本逐字节相同；未新增 GPU 评测。
