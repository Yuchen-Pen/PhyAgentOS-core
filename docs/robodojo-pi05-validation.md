# RoboDojo＋Pi0.5 接入验证

2026-11-01，在225完成了本期约定的单回合、运行中停止和同服务连续案例检查。
这证明固定配置下的工程接入可用；验证只覆盖瓶子任务的一个初态与布局，不建立多任务成功率。
使用说明见[示例 README](../example/robodojo-benchmark/README.md)，
可校验的精简记录见[225-lifecycle.json](../example/robodojo-benchmark/validation/225-lifecycle.json)。
原始模型、资产、Core报告、JSON、录像与节点日志留在本人数据盘，未加入源码提交。

## 实测结果

任务为 `put_bottles_into_dustbin`，task=0/init=0/layout=0，joint动作，native horizon=700，
Core每案例等待上限1800秒；仿真与本地Pi0.5共用GPU0，外层模型和付费API调用均为0。

| 检查 | 原生结果 | 录像／Core结果 |
|---|---|---|
| 首个有效完整回合 v4 | 487步成功，未用满700步 | 488帧，三相机动态画面和终帧复核；正常completed自动结算 |
| v5运行中停止 | 第一回合开始后约15秒发SIGINT；先stopping，再于486步回合边界stopped | 487帧；接纳后仍保留worker、stopped=false；当前回合实际成功，Core整批cancelled且settled；排队的第二case没有启动 |
| v5连续case A | 476步成功 | 477帧，Core child success |
| v5连续case B | 480步成功 | 481帧，Core child success；开始时间不早于A完成时间 |

三次v5 Session共用一个受管理Dora流和同一常驻策略：模型初始化／ready日志各一次，start和idle各三次，
未调用服务重启。连续案例分别具有不同的Core scope、native invocation、JSON和视频路径。
逐帧解码检查帧数为动作步数+1；首／中／末抽看可见机械臂与腕部视角变化，
第二案例首帧重新出现初始瓶子布局，结束时桌面清空。机器人成功由原生任务判定，像素变化统计只作录像核查。

停止语义是**回合边界停止**。当前回合成功与整批取消可以同时成立，不能改成机器人失败，
也不能把收到stop回执当作物理执行立即停止。客户端停止入口调用公共Core cancel_task，并等待其权威对账；
超时未证实时保留blocked和worker，不强杀共享服务。

实机结束后本人四节点和私人Dora正常停止，GPU0回到18MiB/0%，19061—19064均关闭，旧进程回执已归档。
结束前直接复测PID的尝试受限于执行环境的进程视图，因此连续性依据是完整Dora流和策略生命周期日志，
不是一份虚构的结束PID快照。节点日志中的关机警告保留，资源回收以服务记录、端口和GPU检查核实。

## 自动检查与历史失败

78项测试及16个子检查通过，包含真实Core连接合成Gateway的生命周期测试、
停止接纳后保留worker、阻止排队案例派发、未知接纳禁止重发、身份／结果核验、
物理与渲染同步、录像末帧、私人进程／端口归属，以及模型锁校验。
端口测试需要本地监听socket。测试环境有一个未安装pytest-asyncio导致的配置警告；
异步行为由unittest异步测试执行，所有检查通过。

前三次正式Session均原样保留：

1. 初次场景初始化缺msgpack，0动作，基础设施错误。
2. 完成700步但RGB冻结，不接纳为有效结果；原Core遗漏completed导致等待，其原始blocked报告没有重写。
3. 换到GPU3后Kit仍选物理GPU0，相机初始化失败，0动作，Core自动归档infra_error。

后续诊断确认CPU仿真同时关闭Fabric和physics/updateToUsd会导致物理运动而相机姿态不更新。
已显式启用USD写回并在reset检查；为Kit显式绑定物理GPU序号，不能用CUDA重编号后的0替代。
录像终帧复用原生结束检查已采集的观察，不增加物理步。
真实Gateway正常终态是phase=completed，已与ToolResult.status=succeeded分别核验。

v5连续性验证没有重新挑选布局或只保留成功片段；停止测试单独记录，不能与普通评测混成成功率。

## 版本和环境

| 对象 | 实际身份 |
|---|---|
| Core基线 | a606ddfc193fc87968297a7901220c251f6de492 |
| 刘意原生链路 | d827bc35b81f6625b10464660bb71ead3937f953 |
| RoboDojo／XPolicyLab | f57e1633ffed59761bc8e568cfe7530c4b33381c／5d0be1499634d3176258c8f2e31f5fe816497b0a |
| cuRobo | d17b54ce32cba095c0b000c4c58777075d11de0e |
| HF数据版本／checkpoint | 43dacb13d3051a9ccd421f4e7e08e4e3eb29dfab／59999 |
| 模型锁SHA256 | a7040c52335750a50131a50cff9169ce22dd18b9fe9d545d8bc470512e452cc1 |
| Gateway官方源码 | adapter-forge-gateway v1.1.1@c3fb206a16ab970486cd1e523e70a5c7a4f73cbb |
| Gateway归档／二进制SHA256 | b672d3b0c9061c83be1df856be23af1da5e1dfbdfbaebaec69bc271c7ea747c1；ab4596acb59af31bfa0b506690cb07ef8442522a3ad2d7440a8c6355d55e9aad |
| Dora CLI 0.4.1 SHA256 | 024e0a9e...，完整值见本地发布校验回执，非完整哈希不用于校验 |
| 仿真环境 | Python3.11.18、Isaac Sim5.1、IsaacLab0.54.2、Torch2.7.0+cu128、NumPy1.26.0、PyArrow23.0.1 |
| 策略环境 | Python3.11.18、JAX0.5.3、Flax0.10.2、Torch2.10.0、NumPy1.26.4、PyArrow24.0.0、dora-rs0.4.1 |
| 硬件 | RTX PRO5000 72GB，驱动580.126.20；本次仿真与策略共用GPU0 |

源码提交保留生成器、控制适配、配置模板、依赖准备、检查入口和测试；
生成的原生Skill带来源／文件哈希和原始许可证，使用正式Core安装器。
原始模型锁按交付字节保存；新prepare_models.py使用既有受测下载原语，新增准备行为通过本地小夹具验证，
未为展示重新下载11.59GiB权重。真实评测使用此前已逐文件校验的完整资源。

## 使用边界

- 验证了固定任务的接入，未验证其他任务／布局、策略泛化、成功率分布或性能优势。
- IsaacLab只读复用既有安装，缺Git身份；没有宣称完全独立、逐位可复现的环境重建。
- 原生启动202回包丢失后，无法证明是否派发时保留unknown，不能自动重发；已知invocation可查询对账。
- CLI提供正常Ctrl+C/SIGTERM与服务管理，不新增自然语言／网页前端，也不提供物理机器人急停。
- 单worker串行运行，rounds=1；更多平台、策略、并发和Lesson实验需要另行设计。
