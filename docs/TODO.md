# 短期改造 TODO（按序执行，步间确认）

> 工作方式：每步完成后停下，确认后再进下一步；不要一次连做多步。
> 通用注意：
> - 文中 file:line 为编写时快照，仓库经历过重组，落地前以符号名（类/函数）在当前文件树重新定位。
> - 每步收尾固定动作：`python -m py_compile` 触及文件 + 官方入口（`start.bat` → `python src/main.py`）启动冒烟。
> - 已批准的边界：`legacy/` 允许断（本就已断）；jump 的 `sam2.1_hiera_small.pt` 缺失不修；人脸注册流程不进本批；`start_me.bat` 弃用清理不进本批。
> - **前端 5 秒断流回退：暂缓**（第 6 步只做双条件切换）。

---

## 第 1 步：config 统一（地基） ✅ 已完成（2026-11-04）

> 偏差记录：`JUMP_DEVICE` **未迁移**——`model_runtime._configured_jump_device` 是"Django settings 优先、env 兜底"的解析函数而非静态常量，迁入 config 会改变语义，留在原处。另：`main.py` 的 `SRC_DIR/BACKEND_ROOT` 与 config 重复，已顺手去重（`main.py` 改从 config 导入）；`SITUP_YOLO_*` 按计划改名 `POSE_YOLO_*`；`POSE_YOLO_MODEL_PATH` 本步仍为相对路径（第 2 步锚定）。
> 验证：py_compile 全过；`api.config` 引用仅剩已批准放断的 `legacy/new_situp.py:78`；预热双项目加载成功；冒烟 `STEP1_SMOKE_OK`（latest_data / jump400 / situp start→ready→end）。

**目标**：单一 `core/config.py`，后续每步只往里加常量，零返工。

改动：
- 新建 `src/core/config.py`（无副作用，只读 env，不调 `logging.basicConfig`）：
  - 迁移：`CAMERA_INDEX`（原 `api/config.py:2`，活消费 `views.py:61→3883`）；`IDLE_TIMEOUT = 60`（现 `run_user_worker.py:20` 与 `user_manager.py:22` 两处同值重复，合一后两处改 import）；`LOG_LEVEL`（env 读取语义与 `core/log.py:5` 对齐，`api/config.py` 删除后其 import 期 `basicConfig` 副作用随之消失，`core/log.py` 成唯一日志配置点）。
  - 迁移 `sports/situp.py:11-17` 的 `SITUP_MARKER_*`（仍 situp 专属）与 `SITUP_YOLO_*`——**改名为中性 `POSE_YOLO_MODEL_PATH/IMG_SZ/BOX_CONF/KPT_CONF`**（第 8 步 pullup 复用同一权重，避免再改名）。
  - 迁移 `sports/pullup.py:11` `PULLUP_STATIC_IMAGE_MODE`（第 8 步随 MediaPipe 一并删除，本步先照迁）。
  - 迁移 env 项：`DAPHNE_PORT`（`main.py:17`）、`JUMP_DEVICE`（`api/jump/model_runtime.py:29`，保持 `os.environ.get` 覆盖语义）、`CAMERA_INDEX`。
  - 新增常量（本步建骨架，第 2/7 步开始消费）：`VENDOR_DIR`（= `Path(__file__).resolve().parents[2] / 'vendor'`，绝对锚定 sport-backend/vendor）、`FONT_DIR`、`REPORT_DIR`（= `src/report_pdf/`，与 `views.py:118-119` 读取端同处）、`FACE_MODEL_DIR`、`FACE_DB_PATH`（= `data/faces.db`）。
- `api/websocket.py:10` 的 `get_logger` 改从 `core.log` 导入（消掉全仓第二个 `basicConfig` 的先 import 者生效问题）。
- 删除 `src/api/config.py`（其唯一活常量已迁；`legacy/new_situp.py:78` 的引用随之更断——已批准）。

注意：
- 不动 `backend/settings.py`（Django 框架配置）；`SPORTS` 注册表留 `core/registry.py`（它是代码不是配置）。
- `core/config.py` 必须零副作用——PDF 字体注册发生在 Django import 期，会先于很多模块执行。

验证：编译 + 启动冒烟；全仓 `api.config` 引用零命中（legacy 除外，属预期）。

## 第 2 步：vendor 资产归位（修活 bug） ✅ 已完成（2026-11-04）

> 偏差与事实修正：① `faces.db` 实测 12KB **非空**（侦察两份结论相左，以现场为准）——已随迁至 `data/faces.db`，未丢弃；② 环境与 `requirements.txt` 缺 `onnxruntime`（face_sdk 硬依赖）——已装 1.31.1 并补进 requirements；③ `model_runtime` 只把 `VENDOR_ROOT` 改指 `config.VENDOR_DIR`，其本地 `BACKEND_ROOT/PROJECT_ROOT` 保持 src 语义不动（`RuntimePaths` 消费方不受影响）；④ `views.py` 本地 `REPORT_DIR` 改为从 config 导入（`BASE_DIR` 保留供 img1.jpeg）；⑤ 报告返回值改绝对路径——views 侧 `Path(pdf_path).resolve()` 对绝对路径为恒等，读写同处反而修复了原分叉；⑥ `main.py` chdir 注释更新（vendor 已绝对路径，chdir 只服务 latest.jpg/auth.log 等未锚定项）。
> 验证：py_compile 全过；残留引用 grep 仅剩 legacy 的 simsun（已批准）与新 FONT_DIR 行；`FONT_YOLO_FACE_OK`（reportlab 加载两字体 + YOLO 权重存在 + faces.db 存在 + `import face_sdk` 经 bootstrap 成功 + PIL 加载）；官方入口启动、预热双项目（YOLO 绝对路径加载）；`LATEST_OK` + situp start/stop 成功。

**目标**：`sport-backend/vendor/` 成为唯一资产区，消灭"官方入口下字体/报告输出断裂"。

改动（移动用 git mv 保历史）：
- `src/vendor/wcx/` → `sport-backend/vendor/wcx/`（整树含代码，第三方库）；移空后删 `src/vendor/`。
- `src/msyh.ttf`、`src/msyhbd.ttf` → `sport-backend/vendor/fonts/`（统一小写扩展名，代码里混用的 `.TTF` 一并改）。
- `src/face_sdk/` → `sport-backend/vendor/face_sdk/`（整包含 `models/`；`faces_db/` 不跟——DB 归 `sport-backend/data/faces.db`，新建 `data/` 目录，`init_sdk(db_path=…)` 参数现成，第 7 步接线）。
- 消费方改 `VENDOR_DIR`：
  - `sports/situp.py:14`：`'vendor/yolo_pose/yolov8n-pose.pt'` → `str(VENDOR_DIR / 'yolo_pose/yolov8n-pose.pt')`（绝对路径后不再依赖 `main.py:53` 的 chdir；chdir 本身保留，还有别的 cwd 项依赖，超范围）。
  - `api/jump/model_runtime.py:81-91`：`WCX_DIR` 改由 `VENDOR_DIR/'wcx'` 推导；`validate_vendor_assets()`（:106-113）清单同步——`sam2.1_hiera_small.pt` 本就缺失，保持现状不修、清单里标注 known-missing。
  - 三份 PDF 生成器字体（`pdf_report.py:21-23,37-38`、`situp_pdf.py:24-26,43-44`、`pullup_pdf.py:24-26,43-44`）→ `FONT_DIR`；kivy `LabelBase.register` 在 import 期执行，导入顺序 = `urls→views`，确保那时 `core.config` 可用（同根绝对导入 ✓）。
  - 报告**写出**（`pdf_report.py:576`、`situp_pdf.py:458,460`、`pullup_pdf.py:459,461`）→ `REPORT_DIR`（与读取端同处，修读写分叉）；临时图（`temp_gauge_*.png` :161、`str(id)+'.png'` `pdf_report.py:308`）同样锚 `REPORT_DIR`，不再 cwd 相对。
  - 视频叠字 `simsun.ttc`（`sports/situp.py:401-402`、`sports/pullup.py:146-147,921-922`）→ `str(FONT_DIR/'msyh.ttf')`，保留 try/except 回退。
- `core/config.py` 加 sys.path 引导：`sys.path.insert(0, str(VENDOR_DIR))`（模块级一行），使 `from face_sdk import …` 写法不变；本步只验证可 import，第 7 步才写调用方。
- 删除死脚本 `start_with_env.bat`（根已无 `main.py`，启动即失败）。

注意：
- `face_sdk.py:60-63` 读 caffemodel 的临时 `os.chdir` 保留不动（有 finally 还原）。
- 移动后顺手清理 `src/__pycache__`、根 `__pycache__` 里旧路径的残留 pyc（可选）。

验证：官方入口启动（YOLO 权重绝对路径加载成功）；触发一次 PDF 报告生成（字体注册 + 写出到 `src/report_pdf/`）；`python -c "import face_sdk"` 从 sport-backend 根通过。

## 第 3 步：四段生命周期 rename（只改名不改行为） ✅ 已完成（2026-11-04）

> 范围补充：除计划内的 `run_user_worker` 调用点外，`main._prewarm` 也调 `start/stop`（侦察时发现的第三处调用点），一并改为 `load/close`；两处 `RuntimeError('…未调用 start()')` 文案同步改 `load()`；worker 收口日志 `stop 失败`→`close 失败`。`reset()` 最小实现 = 重建 `Evaluator`（两实现一致）。jump `session.py` 与 legacy 的同名 `start/stop` 与 BaseSport 无关，不动。
> 验证：py_compile 全过；`sport.start()/sport.stop()` 全仓零残留；预热双项目（`load→close` 路径）；连续两会话冒烟 `STEP3_SMOKE_OK`（两条 WS 序列均 `[phase:ready, end]`）。

改动：
- `sports/base.py`：`start→load`、`stop→close`、新增 `reset`（均 abstract）。
- `SportSitup`（`situp.py:822-847`）/`SportPullup`（`pullup.py:974+`）：
  - `load()`：行为同现 `start()`——建 `Inferencer` + `Evaluator`（**本步保持两样都建**，单会话语义零变化）。
  - `reset()`：最小实现 = `self._evaluator = Evaluator()`（重建实例而非逐字段清，后续加状态字段不会漏；`Inferencer` 每帧无状态、不动——已核实 `process()` 对 `self` 零赋值）。
  - `close()`：同现 `stop()`。
- `worker/run_user_worker.py`：`sport.start()→sport.load()`、finally `sport.stop()→sport.close()`。

注意：`handle_control` 是第 7 步的事，本步不加，避免半成品。`docs/`/`README` 里 start/stop 的文字描述不动（超范围）。

验证：编译 + 单会话全流程冒烟（行为应与改前逐字节一致）。

## 第 4 步：进程复用池（最大行为变更，专项验证） ✅ 已完成（2026-11-04）

> 设计要点（实现与 TODO 的偏差/补充）：① **ready 兜底 = 进程死亡即时检测**（pump 空轮询发现 `not proc.is_alive()` → 立即报错收口），未做"超时自动重 spawn"——自动重试会掩盖确定性加载失败，错误直接透传前端更诚实；② **out_q 单读者原则**：同 uid 重启时 HTTP 侧等旧泵收口（≤5s，超时放弃本次启动），不做双读者抢 `finished`；③ 测试中发现真缺口：**进程死亡后才接入的 WS 会悬挂**——`register_consumer` 对会话已收口的晚接入补发 `test_end`（前端立刻收到 end）；④ worker 会话空闲超时 = 驻留窗口 +10s 余量，保证 web 的停止信号先到、收口走 `finished` 而非超时；⑤ `shutdown_all` 增加 pump cancel + 驻留池销毁；⑥ 环境观察（非本步回归，steps1–3 同样）：supervisor/proc 停止**不触发** Python 信号/atexit，`shutdown_all` 用进程内驱动验证。
> 验证：py_compile 全过；专项套件 `STEP4_ALL_OK`——A 基本会话、B 复用（同 PID 不新增）、C 同 uid 快速重测、D 并发（池空即 spawn 第二进程）、F 驻留 60s 自灭（`leftover=[]`，服务日志两条"驻留超过 60 秒无新会话，退出"）、E 进程死亡后晚接入 WS **0.0s 补 end**；服务日志另证连续 6 会话复用同一 PID；`shutdown_all` 进程内驱动：有会话 3.0s（无 pump 路径）、空状态 0.00s、无 worker 残留。
> 协议修订（完成后追加，用户实测通过）：`in_q` 统一为 dict 信封——`{'type':'start'}` / `{'type':'stop'}` / `{'type':'task','task':{…}}`；`SportTask.to_bytes/from_bytes` 删除，改白名单 `to_dict/from_dict`（**`_frame/_shm` 永不入列**，类 docstring 铁律 + `__reduce__` 对整对象 pickle 直接抛错——防止 `put(task)` 把整块视频帧拖过管道、共享内存失去意义）；stop 不再伪装成 `finished=True` 的帧任务（`finished` 字段一并删除）；与 `out_q` 的 `{'type':…}` 协议完全对称。验证：`ENVELOPE_ALL_OK`（`__reduce__` 抛错、白名单 7 键、真实帧 SHM 往返 4 条 result、stop→finished→归池、同 PID 复用、shutdown 清池零残留）。

**目标**：同项目连续测试不再等 8 秒模型加载；并发各进程独立驻留 60 秒；无 Manager、无类型封顶、无常驻。

改动：
- `api/user_manager.py`：
  - 新增池：`self._idle = {sport_type: [(proc, in_q, out_q), …]}`（web 进程内字典，LIFO 取用）。
  - `start_sport_test`：先取池（`proc.is_alive()` 检查，死条目丢弃），有→复用并投 `{'type':'start'}` 控制消息；无→现有 spawn 路径。
  - `_finalize` 拆两条路：**归池**（进程活着且无 error → 只收 shm 槽、关会话条目，保留 proc+queues 入池）vs **销毁**（error/进程已死/web `shutdown_all` → 现有 join/terminate 路径；`shutdown_all` 必须连池一起清）。
  - `_pump` finally（:294）：按**会话对象身份** pop（`if self._sessions.get(uid) is session`），修同 uid 快速重测把新会话条目弹出的既存竞态。
  - 会话空闲 60s（:266-269）：停止流程照旧投 `finished`，但收口改归池。
  - 归池前 drain `out_q` 至空（协议上 worker 发 `finished` 即停发 result、FIFO 保证其前的 result 已入队，drain 是双保险）。
  - 冷启动/复用的 `ready` 等待加超时兜底：超时 → 销毁该进程、重新 spawn。
- `worker/run_user_worker.py`：
  - `finished` 后不退出：改 `in_q.get(timeout=IDLE_TIMEOUT)`（60s 从 config 导入）等下一个 `start`；超时 → `close()` 退出。
  - `in_q` 混载协议：`bytes` = SportTask；`dict` = 控制消息（本步只认 `start`：收到 → `reset()` → `put phase:ready` → 继续收帧）。`face_result` 分发留给第 7 步。
  - error 路径不变（发 `error` 后退出）→ web 收到即知不入池。
- 前端零改动（`ready` 语义不变，复用时它只是来得快）。

注意：
- 池无上限：并发结束各进程独立驻留、60 秒后各自自灭（用户明确的设计，勿加"类型封顶"）。
- 池随 web 重启清空（worker 是 daemon，随之退出），预期行为。
- 复用会话的 `_Session` 要能区分"进程归我 spawn 还是归池"——finalize 语义不同，用字段标记。

验证（专项，必做）：
1. 同项目连续两会话：第二个 `ready` 毫秒级、评测正常；
2. 并发两会话：两进程并存，各自结束归池、60s 后同灭；
3. 同 uid 立即重测：会话条目不丢；
4. 注入 worker 异常/杀进程：下次自动 spawn 新的；
5. `shutdown_all` 后无残留进程。

## 第 5 步：助力检测修正（小、独立） ✅ 已完成（2026-11-04）

改动（`sports/situp.py:350-383 _has_assist_behind_head`）：
- 方向：`dir_x = sign(head_x - heal_x)`（脚→头横轴；`heal_x` 即脚踝参考点，`situp.py:507` 注释"肩、脚踝"已核实）。`heal_x` 未缓存（=0）时直接返回 False（无法判定，不误报）。
- 深度条件：`0 < (bcx - head_x) * dir_x ≤ w * assist_behind_depth`（纯横坐标，去掉 2D 方向投影——起身时方向向量转竖直导致上方行人误判的 bug 即此）。
- 纵坐标带：`abs(bcy − 被测者框中心y) ≤ w * assist_lateral_width`（被测者框 = `person_boxes[selected_idx]`，不加新参数）。
- 新增 hint：检测到后方有人 → `hints.append('后方有人')`，挂进现有 `assist_count/assist_hold_frames` 的计数节奏（count>0 且未满时显示）。

注意：**75 帧判定与 `is_assist_cheat` 置位逻辑一行不动**（用户确认沿用原判定，本步只换几何 + 加显示）。

验证：合成 `person_boxes` 脚本跑该函数（起身姿态的上方人不触发、头后同高度人触发）；hint 出现→75 帧后 `is_assist_cheat` 置位的实拍冒烟。

**完成记录**：
- 几何重写按方案落位：`dir_x = sign(head_x − heal_x)`；`heal_x==0`（未缓存）、`head_x==heal_x`（方向退化）、`selected_idx` 越界 → 一律 False（无法判定不误报，后两项为方案外补充守卫）。横轴 `0 < (bcx−head_x)·dir_x ≤ w·assist_behind_dx`；纵轴 `|bcy − 被测者框中心y| ≤ w·assist_band_dy`（被测者框 = `person_boxes[selected_idx]`）。
- 参数改名（用户追加要求，横纵轴语义）：`assist_behind_depth → assist_behind_dx`、`assist_lateral_width → assist_band_dy`；全仓旧名零残留（`GLOSSARY.md` 助力词条本就描述"头部后方、纵坐标相近"，无需改）。
- 新增 hint + 绘制「后方有人」：`0 < assist_count < assist_hold_frames` 时显示（绘制黄色 `(0,255,255)`，与确认态品红区分）；**75 帧判定与 `is_assist_cheat` 置位一行未动**。
- 验证：按用户要求仅编译（`py_compile` 过），原计划的几何脚本与实拍冒烟本步免做。

## 第 6 步：前端预览双条件切换（5 秒断流回退：暂缓） ✅ 已完成（2026-11-04）

改动（`sport-frontend/src/views/student/StudentTestingSitup.vue`）：
- `:306-309 ontrack` 改为只存不挂：`remoteStream = ref(null)`，ontrack 里存 `evt.streams[0]`，**不再立即写 `srcObject`**（现状是 ICE 一通就把本地预览顶掉，模型加载期黑屏——本步修的就是它）。
- 切换条件 = `phase:'ready'` 已收（:349-355 现有分支加标志）**且** `remoteStream` 非空，两处互查：任一后到，条件齐了才 `resultVideo.srcObject = remoteStream`。
- 切换前画面 = 本地采集流（`openCamera` :251-252 挂的那路），一直保留。
- **5 秒断流回退本地：暂缓，不做**（记录在此，勿当遗漏）。

注意：只改仰卧起坐页；`StudentTestingPullup.vue`/`StudentTestingStandjump.vue` 是同样的 ontrack 覆盖模式，列为后续（本批 pullup 还没换 YOLO，不同时动两个页面）。

验证：浏览器实测——开始→本地画面持续到 ready→切后端处理画面，ready 前无黑屏、切换不闪断（`browser.open` 实测，不靠测试代替）。

**完成记录**：
- `remoteStream`/`remoteReady` + `attachRemoteIfNeeded()`（幂等：`srcObject` 身份比对，同一流不重复挂）。偏差：方案写 `ref(null)`，实际两标志无任何模板依赖，用普通局部变量（行为等价，少一层响应式）。
- `ontrack` 只存不挂：`remoteStream = evt.streams[0] || new MediaStream([evt.track])`（空 `streams` 数组兜底），存后立即互查。
- `phase:'ready'` 分支置 `remoteReady = true` 并互查——两处谁后到谁触发切换；切换前画面 = `openCamera` 挂的本地采集流，一直保留。
- `ws.onopen` 重置两标志：同组件内二次测试不复用上次的 ready（否则新连接 ready 前就切走本地画面）。
- **5 秒断流回退仍暂缓（未动，勿当遗漏）**；只改 `StudentTestingSitup.vue`，pullup/standjump 页同模式列为后续。
- 验证：按用户要求仅编译（`npm run build` 过，✓ built in 9.78s）；原计划的 `browser.open` 浏览器实测本步免做。

## 第 7 步：人脸门禁（依赖 1/2/3/4） ✅ 已完成（2026-11-04）

改动：
- 新建 `src/core/face_service.py`（spawn 目标）：
  - 循环收 `{'type':'recognize','student_id','crop','frame_id'…}` → 调 `face_sdk.recognize` → `reply_to.put({'type':'face_result', …})`（`reply_to` = 所属 worker 的 `in_q`，mp.Queue 多生产者，直回不经 web）。
  - 启动即 `init_sdk(model_dir=FACE_MODEL_DIR, db_path=FACE_DB_PATH)` 预载（web 启动时异步 spawn，`main.py` 挂），166MB 加载移出任何会话路径。
  - daemon；web 重启随死，属预期（ADR-0004）。
- 新建 `src/core/face_client.py`（worker 进程内模块级）：
  - `configure(reply_q)`：worker 启动时注入自己的 `in_q`。
  - `submit(student_id, crop, frame_id)`：**单任务无排队**——in-flight 时直接返回 False（本帧丢弃识别资格，不缓存不排队）；空闲才发。完成后由下一帧重新触发。
- `worker/run_user_worker.py`：`in_q` 收到 `face_result` → `sport.handle_control(msg)`。
- `sports/base.py`：新增 `handle_control(msg)`——**给默认空实现，不进 abstract**（pullup 本批不接人脸，勿强制实现）。
- `SportSitup` 接入：
  - 环境检测通过、即将发 `ready` 的节点 → 插入人脸阶段：挂起 `ready`，发 `phase:'face'`；每帧（无 in-flight）用 `person_boxes[selected_idx]` 裁剪当前帧 → `submit`。
  - `handle_control(face_result)`：`ok` → 恢复原流程发 `ready`（走既有读秒）；`fail` → 发 `phase:'face_result' {ok, conf}`，前端提示重试。
  - **未注册放行（18.b 过渡）**：本功能在config.py添加配置控制开关，先 `is_registered(student_id)` 判断——未注册直接放行并提示（不进识别），已注册才卡门。识别与"未注册"的返回区分以 `face_sdk.py:236` 实际行为为准，实现时先读。
  - 裁剪坐标 = 原帧目标人员的 YOLO 框（与 `person_boxes` 同一坐标系，直接切片）；crop 以 ndarray 进消息（pickle ✓）。
- 前端 `StudentTestingSitup.vue`：`phase:'face'` → 显示"正在进行人脸识别，请注视摄像头"；`phase:'face_result'` → 显示通过/未通过（conf 可选展示）。`ready`→读秒链路不动。
- `data/` 目录不存在则创建（`FACE_DB_PATH` 落盘处）。

注意：
- 识别阶段画面评测照常进行，仅计数被门禁挂起；结果可能跨多帧，状态机按 `frame_id` 对齐触发帧。
- 复用池下换会话 = 重建 `Evaluator`，人脸阶段状态天然清零（第 3 步 reset 已保证）。

验证：未注册学生→放行+提示；`register()` 插一条后→通过才 `ready`、失败重试不积压（在途丢帧生效）；前端提示时序；复用两会话下人脸阶段干净。

**完成记录**：
- 新增 `core/face_service.py`（spawn 目标，启动即 `init_sdk` 预载；`is_registered` 服务内先查库——未注册不进识别直接放行）+ `core/face_client.py`（单任务无排队：在途拒投、应答/超时清在途、下一帧重投）。
- **应答通路偏差**：原案 `reply_to` 直携 worker `in_q` 的 mp.Queue——**Queue 不能穿过另一条队列**（spawn `assert_spawning` 实测拒绝），改为 `submit` 现场建单向 Pipe、写端随请求捎带（Connection 可 pickle 实测过）、读端转发线程落回 `in_q`；web 仍不在回路（ADR-0004 已补修订段）。
- `worker/run_user_worker.py`：`set_emit`/`face_client.configure` 启动注入、`face_result` 路由（`finish`+`handle_control`）、`start` 时 `face_client.reset()`（驻留复用跨会话清在途）。
- `sports/base.py`：`set_emit` + `handle_control` 默认实现（**非 abstract**，pullup 不强制）；`situp.py`：`Evaluator(ready_gate)` 在 `ready_count>=10` 转 READY 前挂起逐帧重询，`SportSitup._face_gate`（发 `phase:face`、裁 YOLO 框 `copy()` 脱 SHM 投递、超时按未通过重试）+ `handle_control`（帧号对齐丢陈旧应答；未注册/通过→放行，失败→持续提示）。
- **人脸阶段状态放 `SportSitup.reset()` 显式清零**（原案"重建 Evaluator 天然清零"不覆盖封装层状态，等价实现）；配置 `FACE_GATE_ENABLED`（**默认关**，同预热口径）+ `FACE_GATE_TIMEOUT=10`。
- `user_manager` spawn 带 `get_request_q()`（门禁关时 None）；phase 透传 `ok/conf/registered` 至前端。前端 `faceHint` 优先于 hints 展示：通过 2s 自清、失败持续、`start()`/error 清残留。
- 验证：py_compile 9 文件过；`npm run build` 过（9.93s）；**冒烟 ALL PASS（16 项 / 16.7s）**：IPC 往返、单任务在途拒投、空库未注册放行、已注册→recognize 未匹配不放行（临时插库用后即清）、闸门挂起→`phase:face`→应答回收→放行→`face_result`、陈旧应答丢弃、reset 清零、worker 冷启动 ready + `face_result` 路由不崩 + stop 收口。未覆盖：环境合格→READY 状态机全链路（需真人入画，人工验证）、已注册→**匹配通过**（需录入流程，不在本批）。

## 第 8 步：pullup 换 YOLO（代码量最大，放最后，专项回归） ✅ 已完成（2026-11-05）

改动（`sports/pullup.py`，硬切换）：
- `Inferencer`（:23-66）：MediaPipe → YOLO Pose（`POSE_YOLO_*` 第 1 步已中性命名，权重同 situp 的 `vendor/yolo_pose/yolov8n-pose.pt`）；**选人复制 situp 的 `_select_person`**（中心+面积加权，用户后续自己调"正中间"）；绘制换 COCO 骨架（复制 `COCO_SKELETON`，Q9 定：代码复制不抽公共库）；输出 dict 对齐 situp 结构（含 `person_boxes/selected_idx`）。`PosePoint` 两文件各留一份（z 置 0，visibility=Yolo 关键点置信度）。
- `Evaluator` 全表索引 33→17 重映射（对照表，实现时逐项过）：
  - 杠线基准 `points[19]/[20]`（食指指尖，:312-317）→ 双腕 COCO 9/10——**口径变化：指尖→腕**，上线前人工确认一次杠线高度语义；
  - 地面/脚 `31/32`（:295-303,:502-505）→ 踝 15/16；
  - 肘角 `16-14-12`（:345-353）→ 右腕 10-右肘 8-右肩 6；
  - 肩 `11/12`→`5/6`；耳 `7/8`→`3/4`；髋 `23/24`→`11/12`；`record_keypoints` 索引表（:816-822）0/11/12/23/24 → 0/5/6/11/12；
  - 鼻 `0` 不变；`points[19]/[20]` 跨中心判断（:410）→ 双腕；
  - visibility：MediaPipe `≥90` 口径（:834-839）→ YOLO 置信度（`POSE_YOLO_KPT_CONF` 同口径）；
  - 手部特有检查（拇指/小指侧 `F_H` :480-499 本就整段注释）→ 删，不做映射。
- 删死代码：`State` 枚举（:70-76，唯一引用在注释行）、`k1`/`k2`（:343,:603-611，恒 0）、失败文案 r=4/5 不可达分支（:659,:695-707）、`PULLUP_STATIC_IMAGE_MODE`（连同第 1 步迁走的该常量一并删除）。
- 静态图模式、`smooth_landmarks` 等 MediaPipe 参数全删。

注意：
- 计数语义（120° 直臂、鼻过杠、侧柱借力阻断）保持不变——只换感知来源；`num_all`/提示文案逻辑不动。
- pullup 不接人脸（第 7 步只做仰卧起坐，`handle_control` 默认空实现即可）。

验证（专项）：录屏/实拍逐场景对照——弯臂计数、鼻过杠、脚沾地、侧柱借力、离杠终止；与旧 MediaPipe 版同素材跑一遍，差异只应来自"指尖→腕"口径并经人工确认。

**完成记录（2026-11-05；范围按用户指示收窄：只做关键点索引替换 + 同款选人，评测逻辑与死代码一律不动，故上列"删死代码/专项回归"项未执行）**：
- `Inferencer`：MediaPipe 33 点 → YOLO Pose COCO 17（与 situp 共用 `POSE_YOLO_*` 权重与参数）；**逐字复制** situp 的 `_select_person`（中心 0.9 + 面积 0.1 加权）、`_draw_persons`、`_draw_keypoints`、`COCO_SKELETON`；输出增 `person_boxes/selected_idx` 对齐 situp 结构（pullup Evaluator 不读，无影响）。
- **visibility = YOLO 关键点置信度**（0~1，与原 MediaPipe 同量纲）→ `record_keypoints` 的 ≥90% 判定一行未改。注意：YOLO 置信度分布低于 MediaPipe visibility，头肩比缓冲收样可能变稀——属评测侧调参，按指示未动。
- active 索引替换：杠线 19/20→双腕 9/10；脚 31/32→踝 15/16；肘角 16-14-12→10-8-6；肩 11/12→5/6；耳 7/8→3/4；髋 23/24→11/12；眼 1/4→1/2；`record_keypoints` 字典→0/5/6/11/12；手部点（15/17/19/21 与 16/18/20/22）全部退化为腕 9/10（F_H 消费侧本就注释，计数无读者，行为无影响）；**着色循环 `range(33)`→`range(len(points))`** + 可到达颜色分支重映射（此为冒烟揪出的漏网点）。
- 死代码原样保留（用户指示）：注释块中的旧 33 索引、不可达 elif（手/脚/嘴着色分支）、State/k1/k2、`PULLUP_STATIC_IMAGE_MODE` 常量（config 节注释同步改口径）。
- 验证：`py_compile` 过；**冒烟 ALL PASS（6.2s）**：无人帧全链路不崩 + 17 点伪造悬挂姿态 60 帧评测不崩、60/60 产出 dict（`angle/num_all/score/timestamp` 键与前端契约一致）。未覆盖（按"仅编译+简单冒烟"口径）：真实人像双人选人实测（`_select_person` 为 situp 逐字复制）、真实视频计数回归（上列专项回归人工验证另排）。

---

**序列之外已识别、不进本批**：jump `sam2.1_hiera_small.pt` 缺失；人脸注册/录入流程；`legacy/` 整体删除；`start_me.bat` 弃用清理；pullup/standjump 页的预览切换（第 6 步同模式后续）；pullup 选人改"正中间"（用户自调）。

---

## 追加改动（2026-11-04）：开机预热开关

- `core/config.py` 新增 `PREWARM_ENABLED = False`（**默认关**，置 `True` 恢复原"启动即预热"）；`main.py` 按其决定是否 spawn 预热进程，关闭时打日志说明由首个会话懒加载。
- 同步 `README.md`、`docs/architecture.md` 的"启动即预热"表述为开关语义；ADR-0002 等历史决策记录不改（开关不改变入口结构）。
- 验证：py_compile 过（沿用本批仅编译口径）。

---

## 追加改动（2026-11-05）：人脸通路专用管道直读（ADR-0005）

- 动机：调用方接入面过宽——relay 线程 + `in_q` 路由 + `submit/finish/poll_timeout` 三入口 + 帧号对齐散布三个模块。经两轮评审定案：**保留无状态服务端**，客户端封装为四方法门面。
- `core/face_client.py` 重写：`configure(request_q)`（删 `in_q` 实参，建**每客户端一条持久结果管道**，不建线程）/ `send(student_id, crop)`（在途未超时拒投）/ `poll()`（每帧先调：在途先非阻塞读——命中"学号+毫秒时间戳"对应结果即返回；读空且超 `FACE_GATE_TIMEOUT` → 返回 `{'ok':False,'timeout':True}` 一次并放行重投）/ `reset()`（清状态+排空管道）。删除 `FaceRelay`/`_pending`/`finish`/`submit`/`poll_timeout`。
- `core/face_service.py`：仅回带请求 `timestamp` 一行 + docstring 更新；服务端逻辑不变（保持无状态）。
- `worker/run_user_worker.py`：删 `face_result` 路由分支；`configure` 实参收敛；in_q 消息表收敛为 `start`/`stop`/`task`。
- `sports/situp.py`：`_face_gate` 改**先 poll 后 send**，应答三分支（未注册放行/成功/失败）内联；删 `_face_req_fid` 字段与 `handle_control` 覆写；`sports/base.py` 删 `handle_control` 基方法（唯一调用点即被删分支）。
- 陈旧应答语义：学号+时间戳双键过滤，只认最近一次 send 的对应结果——超时重发后的旧应答到达即丢，行为与原帧号对齐一致（`frame_id` 仅留日志）。`poll` 顺序为**先读后判超时**：管道已有对应结果则返回结果（结果已到即不算超时），避免擦边结果被超时误丢。
- 验证（按用户选定"仅编译"口径）：`py_compile` 5 文件过；全仓 grep 无 `finish/submit/poll_timeout/handle_control/FaceRelay/_face_req_fid` 残留。未做执行级冒烟——重复 pickle 同一写端、`poll(0)`、排空语义依赖 Python `Connection` 既有行为（原 `reply_to` 已依赖同机制），已向用户说明并按其决定省略。
- 同日修订（用户指正）：`FaceClient` **类化**——`FaceClient(request_q, student_id, timeout)` 构造携带学号并建实例专用管道，`send(crop)`/`poll()` 为实例方法，模块级全局状态全部收敛为实例字段；`configure`/`reset` 模块函数删除，BaseSport 新增 `set_face_request_q`（镜像 `set_emit`），worker 的两处模块调用改为 `sport.set_face_request_q(...)`；门禁在会话首帧以 `task.get_uid()` 实例化（`start` 不带 uid），`sport.reset()`/`close()` 丢弃旧实例——原 `reset()` 的"清状态+排空管道"语义由换实例承担。验证：`py_compile` 4 文件过，grep 无模块级调用残留。
- 同日修订（第二轮，用户指正）：**门禁归属与 phase 收敛**——`_face_gate` 改名 `face_recognition(student_id, crop)` 连同 `_face_ok` 移入 FaceClient，`_crop` 移入 Evaluator（新增 `_face_admitted` 裁剪后询门禁）；sport 删 `_send_phase`/`_face_gate`/`_crop`/`_face_on`/`_face_ok`，`BaseSport` 删 `set_emit`/`set_face_request_q`、仅余 `set_face_client`——FaceClient 改由 **worker 启动时实例化**（`FaceClient(face_request_q) if face_request_q else None`）、每会话 `reset()`。**phase 消息收敛**：只留模型加载通知 `{'type':'phase','phase':'加载','value':True|False}`（reset 后 True；load 失败 False），`ready`/`face`/`face_result` 全部退出 phase 通道（`ready`→`加载` 消除与 `data.ready` 的歧义）；人脸结论改经 `FaceClient.status()` 随 Evaluator 返回值 **`data.face`**（`state`: checking/passed/failed/timeout）逐帧承载。user_manager（置 `session.ready`/晚到补发）与 websocket 转发同步改名；前端 `StudentTestingSitup.vue` phase 分支收敛为"加载"、人脸提示改由 `data.face.state` 变化驱动（新增 `lastFaceState` 边沿跟踪，会话重置清零）。验证：`py_compile` 6 文件过、全仓无 `set_emit/_send_phase/_face_gate/ready_gate/phase:ready` 残留、`npm run build` 过（10.38s）。
- 同日修订（第三轮，用户指正）：加载通知信封简化——`{'type':'phase','phase':'加载','value':…}` → **`{'type':'loaded','value':…}`**（type 即语义，`phase` 字段与 `phase` 信封整体取消）；`websocket.phase` 转发器改为 `websocket.loaded`，`user_manager` 泵分支改 `mtype=='loaded'`（整条消息直转 channel），晚到补发同改，前端分支改 `msg.type === "loaded"`。验证：`py_compile` 3 文件过、前后端 grep 无 `'phase'` 消息残留（jump debug 的 `phase` 字段无关）、`npm run build` 过（9.17s）。

---

## 追加改动（2026-11-05）：situp 进入流程改两阶段（坐姿+人脸 → 躺下）

- 动机（用户）：原流程"躺下合格 → 人脸"自相矛盾——人脸要坐着拍。改为 **阶段一** 坐姿合格即启动人脸：`_sit_ok`（臀部贴地：不高于脚踝 40px；双腿屈膝：双膝高于脚踝 20px，排除伸直）→ `_face_admitted` → `face_recognition`，过人脸置 `face_done`；**阶段二** 提示"请平躺" → 原躺下判定（肩/头/脚贴地线连续 10 帧 + 躯干-脚底夹角差≤10 再 10 帧）→ READY。
- 连带修正：地面线拟合缓存与红色拟合线在坐姿阶段**冻结**（坐姿下的肩-踝拟合无意义，会污染阶段二判定与画面）；环境判定块按 `face_done` 分段；hints 分支改为阶段一"请坐下，双腿屈膝固定，准备测试"、阶段二(0/-4)"请平躺，双腿保持不动"，`测试环境检测中/请准备` 与单字"请平躺"退役，-1/-2/-3 异常提示保留。
- 前端：`人脸识别通过`→`人脸识别完成`，`is_waiting_for_flat` 语音改"请平躺，双腿保持不动"；`data.face` 四态提示与 `data.ready` 读秒逻辑不变。**Evaluator 返回值本轮无需新增键**（hints + data.face 已覆盖四条通知）。
- 验证：`py_compile` 过；**两阶段流转冒烟 ALL PASS（8 项）**：站立出"请坐下"且不判坐姿、坐姿置 `face_done` 仍 WAITING、躺下 10+10 帧转 READY、返回 `ready=True`、人脸过后出"请平躺"；`npm run build` 过（9.18s）。未覆盖：真人姿态阈值（40px/20px）实拍校准。

---

## 追加改动（2026-11-05）：State 枚举重构 + 人脸失败第 3 次才提示

- **状态机**（用户）：`WAITING/READY/…` → **`WAIT`（等待坐下）/`FACE`（人脸核验中）/`READY`（识别通过等待躺下）/`FLAT`/`RISE`/`WAIT_FLAT`/`FINISHED`**（数值整体后移一位）；`face_done` 字段删除——由 `READY` 状态取代（`state == READY` 即人脸已过）。连带键位调整：测试阶段门 `>= READY` → `>= FLAT`；`ready` 返回值 `>= READY` → `>= FLAT`（触发时刻 = 躺下合格，与原行为同帧，前端倒计时不受影响）；躺下合格直转 `FLAT`（取消原"进 READY 等 is_flat 确认"的中间帧——躺下门的 10 帧贴地线已含等效确认）；平躺判定列表移除不可达的 `READY`；地面线拟合更新键到 `READY` 阶段（坐姿与测试期冻结）；作弊提示门 `>= FLAT`。
- **失败计数展示**（用户）：`FaceClient` 新增 `_fail_count`（会话内 `failed`/`timeout` 判定累计，`reset` 清零），`status()` 注入 `fail_count` 随 `data.face` 下发；前端仅当 `fail_count >= 3` 才显示"人脸识别未通过，请重试"——前两次静默重试（维持"正在进行人脸识别"提示），通过时照常切"人脸识别完成"。
- 验证：`py_compile` 过；**状态机+计数冒烟 ALL PASS（13 项）**：WAIT+请坐下 → 坐姿过人脸转 READY（face_done 取代验证）+请平躺+`ready=False` → 躺下转 FLAT+`ready=True`；FaceClient 自管道注入三连失败 count 1→2→3、通过后计数保留、reset 清零；`npm run build` 过（9.09s）。遗留观察：躺姿坐标头朝左时躯干角≈177°→修正角 60°会直转 RISE——旧流程同几何会卡死在 READY，均依赖生产"头朝右"机位（既有约束，非本批引入）。
