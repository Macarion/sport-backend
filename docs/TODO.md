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

## 第 3 步：四段生命周期 rename（只改名不改行为）

改动：
- `sports/base.py`：`start→load`、`stop→close`、新增 `reset`（均 abstract）。
- `SportSitup`（`situp.py:822-847`）/`SportPullup`（`pullup.py:974+`）：
  - `load()`：行为同现 `start()`——建 `Inferencer` + `Evaluator`（**本步保持两样都建**，单会话语义零变化）。
  - `reset()`：最小实现 = `self._evaluator = Evaluator()`（重建实例而非逐字段清，后续加状态字段不会漏；`Inferencer` 每帧无状态、不动——已核实 `process()` 对 `self` 零赋值）。
  - `close()`：同现 `stop()`。
- `worker/run_user_worker.py`：`sport.start()→sport.load()`、finally `sport.stop()→sport.close()`。

注意：`handle_control` 是第 7 步的事，本步不加，避免半成品。`docs/`/`README` 里 start/stop 的文字描述不动（超范围）。

验证：编译 + 单会话全流程冒烟（行为应与改前逐字节一致）。

## 第 4 步：进程复用池（最大行为变更，专项验证）

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

## 第 5 步：助力检测修正（小、独立）

改动（`sports/situp.py:350-383 _has_assist_behind_head`）：
- 方向：`dir_x = sign(head_x - heal_x)`（脚→头横轴；`heal_x` 即脚踝参考点，`situp.py:507` 注释"肩、脚踝"已核实）。`heal_x` 未缓存（=0）时直接返回 False（无法判定，不误报）。
- 深度条件：`0 < (bcx - head_x) * dir_x ≤ w * assist_behind_depth`（纯横坐标，去掉 2D 方向投影——起身时方向向量转竖直导致上方行人误判的 bug 即此）。
- 纵坐标带：`abs(bcy − 被测者框中心y) ≤ w * assist_lateral_width`（被测者框 = `person_boxes[selected_idx]`，不加新参数）。
- 新增 hint：检测到后方有人 → `hints.append('后方有人')`，挂进现有 `assist_count/assist_hold_frames` 的计数节奏（count>0 且未满时显示）。

注意：**75 帧判定与 `is_assist_cheat` 置位逻辑一行不动**（用户确认沿用原判定，本步只换几何 + 加显示）。

验证：合成 `person_boxes` 脚本跑该函数（起身姿态的上方人不触发、头后同高度人触发）；hint 出现→75 帧后 `is_assist_cheat` 置位的实拍冒烟。

## 第 6 步：前端预览双条件切换（5 秒断流回退：暂缓）

改动（`sport-frontend/src/views/student/StudentTestingSitup.vue`）：
- `:306-309 ontrack` 改为只存不挂：`remoteStream = ref(null)`，ontrack 里存 `evt.streams[0]`，**不再立即写 `srcObject`**（现状是 ICE 一通就把本地预览顶掉，模型加载期黑屏——本步修的就是它）。
- 切换条件 = `phase:'ready'` 已收（:349-355 现有分支加标志）**且** `remoteStream` 非空，两处互查：任一后到，条件齐了才 `resultVideo.srcObject = remoteStream`。
- 切换前画面 = 本地采集流（`openCamera` :251-252 挂的那路），一直保留。
- **5 秒断流回退本地：暂缓，不做**（记录在此，勿当遗漏）。

注意：只改仰卧起坐页；`StudentTestingPullup.vue`/`StudentTestingStandjump.vue` 是同样的 ontrack 覆盖模式，列为后续（本批 pullup 还没换 YOLO，不同时动两个页面）。

验证：浏览器实测——开始→本地画面持续到 ready→切后端处理画面，ready 前无黑屏、切换不闪断（`browser.open` 实测，不靠测试代替）。

## 第 7 步：人脸门禁（依赖 1/2/3/4）

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
  - **未注册放行（18.b 过渡）**：先 `is_registered(student_id)` 判断——未注册直接放行并提示（不进识别），已注册才卡门。识别与"未注册"的返回区分以 `face_sdk.py:236` 实际行为为准，实现时先读。
  - 裁剪坐标 = 原帧 YOLO 框（与 `person_boxes` 同一坐标系，直接切片）；crop 以 ndarray 进消息（pickle ✓）。
- 前端 `StudentTestingSitup.vue`：`phase:'face'` → 显示"正在进行人脸识别，请注视摄像头"；`phase:'face_result'` → 显示通过/未通过（conf 可选展示）。`ready`→读秒链路不动。
- `data/` 目录不存在则创建（`FACE_DB_PATH` 落盘处）。

注意：
- 识别阶段画面评测照常进行，仅计数被门禁挂起；结果可能跨多帧，状态机按 `frame_id` 对齐触发帧。
- 复用池下换会话 = 重建 `Evaluator`，人脸阶段状态天然清零（第 3 步 reset 已保证）。

验证：未注册学生→放行+提示；`register()` 插一条后→通过才 `ready`、失败重试不积压（在途丢帧生效）；前端提示时序；复用两会话下人脸阶段干净。

## 第 8 步：pullup 换 YOLO（代码量最大，放最后，专项回归）

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

---

**序列之外已识别、不进本批**：jump `sam2.1_hiera_small.pt` 缺失；人脸注册/录入流程；`legacy/` 整体删除；`start_me.bat` 弃用清理；pullup/standjump 页的预览切换（第 6 步同模式后续）；pullup 选人改"正中间"（用户自调）。
