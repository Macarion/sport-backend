# 人脸应答专用管道直读与客户端门面化

调用方接入面过宽：转发线程（FaceRelay）→ `in_q` → `handle_control` 的应答通路把传输簿记泄漏进业务，`submit`/`finish`/`poll_timeout` 三个入口与帧号对齐散布在 face_client、worker、sport 三个模块。决定（经两轮评审定案）：**保留无状态服务端**，把客户端收敛为四方法门面 `configure(request_q)` / `send(student_id, crop)` / `poll()` / `reset()`；应答不再混入 worker 的 `in_q`——请求仍经共享 `request_q`，每请求捎带**每客户端一条持久专用结果管道**的写端，读端由 `face_client` 在调用方线程 `poll(0)` 非阻塞直读，**无新线程**。服务端仍 `get → recognize → reply_to.send`，仅在应答回带请求的毫秒 `timestamp`。

- **单任务限流不变**：`send` 见在途（未超时）即拒（`False`，本帧丢弃，不排队）；超时只在 `poll` 判定与上报——在途且管道读空超过 `FACE_GATE_TIMEOUT` → 返回 `{'ok': False, 'timeout': True}` 一次并放行重投；管道已有对应结果则先返回结果（结果已到即不算超时）。调用方每帧**先 poll 后 send**。
- **陈旧应答过滤 = 学号 + 毫秒时间戳双键**：应答回带 `student_id` 与请求时间戳，`poll` 只保留与最近一次 send 完全对应的结果，其余（迟到/陈旧）读到即丢；`reset` 清在途/超时标记并**排空管道**（驻留复用进程跨会话不残留）。`frame_id` 仅留作日志，不参与过滤。
- ADR-0004 三不变量保留：不经 web、直回请求方、单任务无排队。

## Considered Options

- **服务端按学号深度 1 槽位（最新覆盖，超时重发折叠成一次检测）**：暂缓。用户决定先保留无状态服务端；调用方单任务限流因"一会话一学号"已等价于按学号主限流，服务端态只换取忙时折叠，不值当让渡"零状态、零清理、失败隔离"。
- **每请求新建 Pipe + 转发线程（原方案）**：被拒。每 worker 一条常驻线程、每帧一条 Pipe、结果绕行 `in_q`，且 sport 侧需帧号对齐过滤——复杂度的根源。
- **无过滤**：被拒。换会话后迟到应答可能以旧对象放行，不安全。
- **按帧号过滤**：被拒。`frame_id` 跨会话不唯一（新会话从头计数），存在撞号窗口；且对人脸用途"结果是否对应刚发的那张图"不重要，"是否对应当前学生+本次请求"才重要。
- **让服务端登记/监听多条客户端管道**：被拒。服务端须记住客户端集合 = 有状态，违背保留无状态的前提；`reply_to` 随请求捎带保持服务端无感。

## Consequences

- `poll` 是唯一消费点，超时/结果/空三态内聚在客户端状态机；worker `in_q` 消息表收敛为 `start`/`stop`/`task`，`face_result` 分支与 `BaseSport.handle_control` 基方法删除（唯一调用点即被删分支）。
- 结果语义 = "最近一次 send 的对应应答"：超时重发后的旧应答到达即被双键丢弃，不再提前放行——与原帧号对齐行为一致。
- 服务端无状态的代价照单全收：不取消、不对账，`reset` 后迟到应答由客户端过滤兜底；识别失败/回发失败仍是记日志丢弃。
- 验证口径（用户选定）：仅 `py_compile`；重复 pickle 同一写端、`poll(0)`、排空语义依赖 Python `Connection` 既有行为（原 `reply_to` 已依赖同一机制），不做执行级冒烟。

## 修订（2026-11-05，类化）

`FaceClient` 由模块级全局改为**类**（用户指正）：`FaceClient(request_q, student_id, timeout=FACE_GATE_TIMEOUT)` 构造即携带学号并建实例专用管道，`send(crop)` / `poll()` 为实例方法，原模块级状态（`_in_flight`/`_timestamp`/管道句柄/`configure`/`reset` 函数）全部收敛或删除。装配改为：worker 启动时 `sport.set_face_request_q(face_request_q)`（镜像 `set_emit` 的注入模式），门禁在**会话首帧**用 `task.get_uid()` 实例化（`start` 消息不带 uid，uid 只随帧任务到达）；**换会话换实例**——`sport.reset()`/`close()` 丢弃旧实例，原 `reset()` 的"清状态+排空管道"语义由实例替换承担（管道随实例回收关闭，服务端若仍回发仅记 warning）。

## 修订（2026-11-05，门禁入 FaceClient 与 phase 收敛）

- **闸门归属**：`_face_gate` 改名 `face_recognition(student_id, crop)` 移入 FaceClient（`_face_ok` 同迁），`_crop` 移入 Evaluator（新增 `_face_admitted`：门禁未接直接放行，否则裁剪选中者后询门禁）；FaceClient 改由 **worker 启动时实例化**（`FaceClient(face_request_q) if face_request_q else None`，`start` 不带 uid 故学号仍随每帧 `task.get_uid()` 传入）、每会话 `reset()`；`BaseSport` 的 `set_emit`/`set_face_request_q` 删除，仅余 `set_face_client`，sport 不再持有任何队列/管道逻辑。
- **phase 信封取消**：模型加载通知独立为 `{'type':'loaded','value':True|False}`（reset 后 True、load 失败 False），原 `phase` 信封与 `ready`/`face`/`face_result` 全部退出——`ready` 更名消除与评测返回值 `data.ready` 的歧义；人脸结论改经 `FaceClient.status()` 随 Evaluator 返回值 **`data.face`**（`state`: checking/passed/failed/timeout）逐帧承载，前端按状态变化更新提示（`user_manager` 置 `session.ready`/晚到补发、`websocket` 转发器同步改为 `loaded`）。
- ADR-0004 三不变量与无状态服务端不变。
