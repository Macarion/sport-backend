# 用户进程架构改造计划

> grilling 设计会话产出的共享理解终稿。决策依据见 [adr/](adr/)，旧架构现状见 [multiprocess_architecture.md](multiprocess_architecture.md)，术语见仓库根 `GLOSSARY.md`。
> 状态：**已实施（2026-11-03），冒烟通过**（启动/HTTP/WS 控制面/清理；按要求未启前端，原步骤 1 的"浏览器实测"以接口级验证替代）。

## 1. 终态目录结构

```
sport-backend/
├── start.bat                    # python src/main.py
├── vendor/  docs/
└── src/                         # Python 项目根（sys.path 根，不是被 import 的包）
    ├── main.py                  # 唯一入口：轻顶层 + guard 内托管 daphne、清理、预热（ADR-0002）
    ├── manage.py                # Django 管理（平移）
    ├── backend/                 # Django project（平移，settings/routing/asgi）
    ├── api/                     # Django app（平移，views/urls/websocket/user_manager…）
    ├── core/                    # 契约：task.py、shm.py、registry.py（字符串懒注册）、log.py（零重依赖）
    ├── sports/                  # base.py(BaseSport)、situp.py(SportSitup)、pullup.py(SportPullup)
    ├── worker/                  # run_user_worker() 用户进程入口
    └── legacy/                  # 死代码簇（new_*、sport_manager、process_manager、base_sport）
```

旧文件去向：`launcher/queue_manager/inference_worker/evaluation_worker`、根 `main.py`、`src/__init__` 的 Launcher 导出 **删除**；`task.py` 双份合入 `core/task.py`；`shared_memory_pool`（全局池）→ `core/shm.py`（per-user 槽位）；`config.py` 拆解（IPC 段删、项目规格进 sports 类属性、日志进 `core/log.py`）；`sport_situp/pullup` → `sports/`。

## 2. 已定决策

| 决策点 | 结论 |
|---|---|
| 进程归属 | Web 入口直接 spawn 用户进程，无 Manager（ADR-0001） |
| 入口形态 | `src/main.py` 受保护入口、程序化托管 daphne（ADR-0002） |
| 注册表 | `core` 字符串懒注册 `{"situp": "sports.situp.SportSitup", …}` + importlib 解析，配解析测试；web 只读 keys |
| 封装 | `SportSitup/SportPullup(BaseSport)`，接口 `start()/process()/stop()`；Inferencer/Evaluator 不合并，由包装类持有链式调用 |
| 模型加载 | `start()` 内同步加载（子进程）；loading 期间帧照常入有界队列（丢最旧） |
| 预热 | Web 启动后台 spawn 一次性加载进程打热文件缓存，**日志带"预热"**；首次最坏 ~13s、之后 ~4–5s |
| 销毁触发 | 显式 stop + WS 断线 + 60s 空闲 + 父进程退出（daemon）+ 进程崩溃 → error 后销毁 |
| IPC | 双 `mp.Queue`（帧入/结果出，随 spawn 继承）+ 4 槽 SHM 同槽就地绘制回传；背压计数在 web 会话内维护（不依赖 Windows 上不可靠的 `qsize`） |
| 槽位尺寸 | 按全局最大帧（沿用 `MAX_FRAME_BYTES` 口径），保持现状对 ≤720p 输入的容忍 |
| 协议 | 现有 WS/HTTP 消息冻结，仅新增 `phase`(loading/ready) 与 `error` 两条 |
| 范围 | situp+pullup 一次全迁不留双轨；sitreach/jump 不动；死代码簇挪 `legacy/` |
| 容量 | CPU、≤8 并发（每进程实测 ~324MB，单帧 48ms） |

## 3. 一帧的数据流（新）

浏览器 WebRTC 帧 → web 解码 → 写空闲 SHM 槽 → `in_q.put(frame_id, slot, config)`（积压丢最旧）→ 用户进程 `get` → attach 就地读 → `SportSitup.process`（YOLO 就地绘制 → Evaluator 状态机计次/画提示）→ `out_q.put(frame_id, slot, eval_data)` → web 轮询线程读回槽 → 推 WebRTC 输出轨 + 下发 `eval_result` → 槽归还。

## 4. 随迁移消失的旧机制（对应旧文档 §6）

- 6.1 乱序/frame0 脆弱 → 单进程 FIFO 保序，会话即进程（`Evaluator` 乱序丢帧检查随旧管线删除）
- 6.2 Manager 每帧 RPC → 无 Manager
- 6.3 评测单点/会话泄漏 → 消失（状态在用户进程内）
- 6.4 finished 双投 → worker 单出口
- 6.5 SHM 超时回收 → 父进程直管槽位
- 6.6 只增不减 → 四路径销毁
- 6.7 双份 task → `core/task.py` 单份
- 6.8 端口不一致 → Manager 配置整体消失

## 5. 落地顺序与验证（每步可运行）

**步骤 1 · 纯位置重组（旧架构照跑）**
`web_server/*` → `src/` 平移；死代码簇 → `src/legacy/`；删 `views.py:20` 死导入与 `:123-125` 死常量；旧 launcher 的 daphne cwd 改一行临时适配。
验证：旧链路全栈照跑——start.bat 起旧系统，浏览器实测 situp 一轮；grep 确认 legacy 簇仍零活引用。

**步骤 2 · 契约包 core/**
`task.py`（双份合一）、`shm.py`（槽位：父创建/attach/release/unlink）、`registry.py`（字符串注册表）、`log.py`；根 `main.py` 加临时 `sys.path` 行让新代码可导入（步骤 6 删）。
验证：一次性脚本——pickle 往返、两键 importlib 可解析、4 槽创建→attach→释放→unlink 循环无泄漏。

**步骤 3 · sports/ 封装**
迁入两个 sport 文件，新增 `base.py` 与包装类；保留过渡 shim `src/sport_situp.py`（旧管线引用点，步骤 6 删）。
验证：smoke 脚本——`SportSitup().start()` 计时、对真实/随机帧 `process()` 返回 points/score 字典、`stop()` 后可重入；旧管线经 shim 仍可跑。

**步骤 4 · worker 进程（不碰 web）**
`worker/run_user_worker(sport_type, in_q, out_q, slots, config)`：get→attach→process→就地绘制→out；`start()` 完成发 `phase:ready`；异常发 `error` 后退出；收到 finished → `stop()` 退出。
验证：假 web 脚本自建队列+槽 spawn 真 worker，灌 100 帧收结果帧（像素已绘制）+ eval_data + loading/ready 时序；杀进程验证 error 行为。

**步骤 5 · web 切换（端到端）**
新 `src/main.py`（轻顶+guard：daphne、清理/信号、预热线程带日志、`set_start_method('spawn')`）；重写 `user_manager`（start=建 Q+槽+spawner 线程 spawn；poll 线程阻塞 get；四路径销毁）；`websocket.py` push_frame 改新通道；views 端点签名不变、未注册 key 返回 error；`start.bat` → `python src/main.py`。
验证：**浏览器端到端**——loading→ready→计分/绘制→stop 干净结束；进程按用户生灭、无孤儿；连续开/关 8 次槽位与进程收敛；jump 端点返回干净 error；日志出现预热记录。

**步骤 6 · 删旧与文档**
删：`launcher/queue_manager/inference_worker/evaluation_worker`、根 `main.py`、`src/task.py`、`src/shared_memory_pool.py`、`config.py`（整体删除：IPC 段随旧文件消失、项目规格已移入 `sports/*.py` 模块顶、日志在 `core/log.py`）、`src/__init__`、`api/task.py`、`api/shared_memory_pool.py` 桩、MANAGER env、过渡 shim 与临时 sys.path 行、`src/start.bat`（daphne CLI 启动方式废弃）；`Evaluator` 乱序丢帧检查——**situp 与 pullup 均已删除**（单进程 FIFO 下乱序不可能：web 单点写入 + 单 worker 顺序消费。situp 的 `frame_id` 初始化为 -1、守卫本就永不触发，属死代码；pullup 于前端改造轮按用户确认删除，`self.frame_id = frame_id` 赋值保留——它仍是算法帧计数器。语义差异：pullup 首帧 frame0 不再被丢弃、会进入算法一次，引体向上计数建议实测回归）。
验证：grep 零残留（`MANAGER_|AUTH_KEY|cmd_queue|inference_worker|evaluation_worker|queue_manager|web_server`）；全栈冒烟；README/start.bat/文档更新。
遗留拍板：`legacy/` 簇最终删除由你确认后单独执行。

## 6. 明确不做（本轮）

- 子进程直连 WebRTC 媒体面（隐患清单见 ADR-0001；worker 帧源无关接口已预留）
- 模型常驻服务进程（CPU 实测无优势）
- sitreach/jump 适配（jump 端点行为变化见 ADR-0001）
- 前端协议改动（仅允许新增 `phase`/`error`）
- `legacy/` 簇的删除
