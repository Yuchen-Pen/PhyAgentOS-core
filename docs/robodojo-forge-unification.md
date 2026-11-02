# RoboDojo Pi0.5：Forge 实现模式对齐

本次对应审阅的三项建议：集中管理 controller 生命周期、版本化源码补丁、共用校验与持久化。
评测目标仍是由 PAOS 管理 RoboDojo＋纯 Pi0.5 的案例、进度和结果。

## 1. 状态与归属

[`lifecycle.py`](../example/robodojo-benchmark/lifecycle.py) 定义 `RootState/RootPhase`、
`EpisodeState/EpisodePhase`，保留 journal schema_version=1 的 JSON 格式。
`controller.py` 负责请求、原生调用和持久化时机，状态规则归入以下入口：

| 入口 | 负责的规则 |
|---|---|
| `require_scope`、`RootState.owns` | 核对 Core 身份；历史 root 的停止请求不能控制新 root 的 worker |
| `claim_episode` | 仅开放的匹配 root 可以占用唯一 worker |
| `transition_episode` | 转换状态、维护单调观察版本／历史摘要，拒绝改写终态 |
| `request_episode_stop`、`request_root_stop` | 记录停止意图；对已结束回合的确认集中释放 worker |
| `settle_roots` | 未完成的停止保留占用；所属回合都权威停止后关闭 root |
| `validate_journal` | 恢复及落盘前检查身份、终态与 worker 归属，不自动修补后重发 |

`running → unknown → running/terminal` 允许断线后的对账恢复；`terminal` 不可重开。
回合权威结束后仍等待 Core stop 确认再复用 worker。停止接纳和真正停止始终分开。
启动回包丢失仍保存 unknown、禁止自动重发。根状态内部区分 running/stopping/stopped，
对外接口继续使用原有 running/stopped，未改变 Core 协议。

参考 Gateway `domain/action_invocations.py` 的类型化记录、集中转换和独立并发占用字段。
该参考实现把 unknown 列为终态；本适配器的 unknown 表示物理执行未确认，继续占用 worker，
没有照搬参考中的终态／释放策略。

## 2. 可追溯补丁

[`patches/native-v1.json`](../example/robodojo-benchmark/patches/native-v1.json) 是
`robodojo-pi05-native/v1` 补丁目录，固定针对刘意交付的
`d827bc35b81f6625b10464660bb71ead3937f953`。
5 组、15 项命名修改覆盖相机、步数／观察、录像目录、回合结束处理和常驻策略。

[`source_patches.py`](../example/robodojo-benchmark/source_patches.py) 使用不可变的
`SourcePatch/TextEdit`：核对版本与目标提交、拒绝重复／越界路径，每个锚点必须恰好出现一次，
检查生成 Python 的语法。所有文件先在内存检查，之后写入 prepare 的临时构建目录；
prepare 只有完整构建成功才发布新目录，失败不留下半个可用安装包。

`PROVENANCE.json.patches` 保存补丁版本、目录 SHA256、每个文件与每项修改前后的 SHA256。
安装包另含完整 `NATIVE_PATCHES.json`，其字节也纳入 `bundled_files`。
`upstream_files`、干净 checkout／固定 commit 检查和禁止覆盖旧构建均保留。
以后改变补丁语义需显式增加版本并更新执行器支持，不能仅放宽锚点计数以适配新源码。

## 3. 共用校验与持久化

[`support.py`](../example/robodojo-benchmark/support.py) 仅使用标准库，能同时被控制环境与系统 Python 启动器调用：

- `deployment_identity` 与公共 Core digest 的规范 JSON 算法一致，控制日志、安装包回执共用。
- `verify_bundle` 由 `NativeRuntime.verify_bundle` 和 `launch_plan` 共用，核部署身份及包内文件哈希。
  旧安装包未记显式 deployment_identity 时仍核 deployment.json 身份和哈希；新包另存显式身份。
- `atomic_write` 先在同目录完成临时文件写入及 fsync，再替换日志或用硬链接独占发布证据，最后同步目录。
  不可变证据只接受完全相同的重放；不同内容和软链接均拒绝。原 controller 独占锁及日志外部修改检查保留。
- `ErrorCode/BenchmarkError` 集中管理控制拒绝、包身份不符、补丁失败、结果无效、启动／查询／停止状态不明。
  `unknown` 与 `infra_error` 的业务区别不变，也不改写原生 episode.error_code。

[`results.py`](../example/robodojo-benchmark/results.py) 将 ToolResult 检查与纯结果结构检查
`validate_native_result` 提为共用入口。路径读取和视频归属／哈希仍在 `validate_result`，
原字节归档仍在 `archive_result`。额外拒绝布尔 seed/count 以及错误类型的 metrics，避免 Python 数值相等造成误接纳。

## 4. 证据与边界

- **95 项离线测试、30 项子检查通过**。包括公共 Core 双案例结算、取消与丢包，
  旧日志恢复、跨 root 停止隔离、终态不可重开、补丁拒绝且不部分写入、并发证据发布及写入中断。
- 真实固定源码的新构建与此前成功实机安装包的 **45 个原生文件逐字节相同**。
  部署路径、启动帮助文件及新增回执另行生成，因此整个包的哈希会变化。
- 此前 v4/v5 的 **2 份控制日志、4 份实机结果结构**通过新入口只读校验，结果分类一致。
  这不是重跑，也不增加统计样本。
- 系统 Python 在禁用 site-packages 的情况下通过新安装包的 tool 启动计划检查；
  正式打包／安装流程由合成源测试覆盖。
- 本轮新增 Session=0、GPU 推理=0、付费 API=0；新真实源码构建未安装／启动。
  历史实机记录仍属于原运行版本，不能宣称新版已完成新的 GPU 验收。

可公开摘要见 [`validation/forge-unification.json`](../example/robodojo-benchmark/validation/forge-unification.json)。
完整本机证据位于个人项目 `runtime/robodojo-pi05-v6-forge-unification/verification/`，不把运行环境或视频提交仓库。
测试命令见示例 README，既有 pytest 的 `asyncio_mode` 未识别警告保留；异步用例使用 unittest 执行，未跳过。

## 5. 师兄提供的参考源码

附件作为设计参考，没有运行其安装脚本、替换公共 Core 或升级当前 Gateway／消息依赖。
`master` ZIP 不包含可验证的 Git commit，以下仅以压缩包 SHA256 标识快照：

| 附件 | SHA256 | 本轮查阅点 |
|---|---|---|
| forge_gateway-master.zip | a26cbeef6a02e6a8790fe3c838c7633d2f8be6dbad3350b01ed9e533359e8136 | domain/action_invocations.py：状态转换与并发占用 |
| forge-master.zip | 0399d9c6395f0d30d9d37caddbfaf89a6356b32b669cf7f668e5bfa6706645ff | packages/tool endpoint/models.py、wire/errors.py：类型、结果配对与分类错误 |
| forge-forge-msgs-v1.0.1.zip | c23c6e85c6178a826035f1d575a37acbe9aff3eabac572a4b52a872a528b89e1 | packages/msgs control.py：命令字段及结构校验；保留原生消息协议 |
