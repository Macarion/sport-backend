# 后端架构（现状速览）

> 供快速恢复上下文用；细节以代码与 `docs/TODO.md`（进度）、`docs/adr/*`（决策）为准。
> `docs/multiprocess_architecture.md` 与 `docs/refactor_plan.md` 是**历史文档**（旧架构，已被重构替换）。

## 1. 进程拓扑

- **唯一入口 `src/main.py`**（`start.bat` → `python src/main.py`，ADR-0002）：`chdir` 到 `sport-backend/` 根 → 按 `PREWARM_ENABLED`（`core/config.py`，**默认关**）决定是否 spawn 一次性预热进程（注册表全解析 + 逐项加载模型打热文件缓存）→ 按 `FACE_GATE_ENABLED`（**默认关**）spawn 人脸服务（ADR-0004 预载）→ 程序化托管 daphne（`DAPHNE_PORT`，默认 8090）→ atexit/SIGTERM 收口 `user_manager.shutdown_all()`。
- **web 进程**（daphne，单进程，Django + channels）：Django 应用平铺在 `src/`（`manage.py`、`backend/`、`api/`）。`sys.path` 根 = `src/`，全仓顶层绝对导入（`core.*` / `api.*` / `sports.*`），无 `from src.` 包结构。
- **用户进程**（每会话 spawn 或从驻留池复用，daemon）：`worker/run_user_worker.py`，一个进程承载一个体测项目的推理 + 评测全链路。
- **预热进程**：一次性，跑完即退；**默认不启动**（`PREWARM_ENABLED=False`，置 True 才 spawn，关闭时模型由首个会话懒加载）。
- **人脸服务进程**（`FACE_GATE_ENABLED` 开启时由 `main.py` spawn，daemon 随 web 死，属预期 ADR-0004）：启动即 `init_sdk` 预载 face_sdk 166MB，单任务识别、**服务端无状态**；应答经**随请求捎带的专用结果管道写端**直落所属 worker，由 `FaceClient` 实例在调用方线程 `poll(0)` 直读（无转发线程、不经 `in_q`、不经 web；ADR-0005，Queue 不能穿队列见 ADR-0004 修订）。

## 2. 会话与驻留池（ADR-0003，已完成）

- **四段生命周期**（`sports/base.py` ABC，两实现 `SportSitup`/`SportPullup`）：
  `load` 进程级加载模型（每进程一次，冷启动 ~4–8s）→ `reset` 会话级复位（= 重建 `Evaluator`，模型保留；每会话一次含首次）→ `process` 逐帧 → `close` 进程级释放。
- **会话流程**（`api/user_manager.py`，核心文件）：
  `start_sport_test` → 先取驻留池（LIFO、`is_alive` 过滤、双 drain），池空则 spawn → **显式 `{'type':'start'}` 定界**（冷启动/复用同路径）→ worker `reset()` 后回 `phase:ready` → 帧循环 → `stop` → `finished` → **归池驻留 60s**（`IDLE_TIMEOUT`，worker 自计时）→ 超时自灭。
- **池**：web 进程内字典 `_idle {sport: [(proc,in_q,out_q)]}`，**无 Manager 服务器**；并发多进程各自驻留、无类型封顶、无常驻。
- **归池条件**：`finished` 回执已确认 且 进程存活 且 无 `error_seen`；否则销毁（join→terminate→关队列）。`error`/进程死亡不入池。
- **out_q 单读者原则**：唯一读者是该会话的 pump（`_pump`，daphne 事件循环上）；同 uid 快速重测时 HTTP 侧等旧泵收口（≤5s，超时放弃本次启动），绝不双读者抢 `finished`。
- **收口路径**：HTTP stop / WS 断线（unregister 等价停止）/ 会话空闲 60s → 停止信号 → pump 见 `finished` → 归池；pump 空轮询发现**进程死亡即时报错收口**（不等 60s）。
- **竞态修复**：`_pump` finally 按**会话对象身份** pop（同 uid 重测不弹掉新条目/新消费者）；**进程死亡后才接入的 WS** 在 `register_consumer` 时补发 `test_end`（前端不悬挂）。
- `shutdown_all`：清全部会话（cancel pump）+ **清驻留池**（terminate + 关队列）。

## 3. 消息协议

- **in_q（web→worker）：全部 dict 信封**，mp.Queue 底层 pickle：
  - `{'type':'start'}` 开始会话；`{'type':'stop'}` 停止（会话中→回 finished 转驻留；驻留中→退出进程）；
  - `{'type':'task','task':{…}}` 帧任务体。
- **out_q（worker→web）**：`loaded`（`{value: True|False}`——reset 后 True、load 失败 False；人脸结论由 `result.data.face` 承载）/ `result`（每帧必回，带 slot 回执）/ `finished` / `error`。
- **`SportTask` 序列化铁律**（`core/task.py`）：只经 `to_dict()` **白名单 7 键**（uid/frame_id/sport_type/shm_name/frame_shape/config/data）；**`_frame`/`_shm` 永不入列**——它们是进程内 attach 句柄，整对象入队会把整块视频帧拖过管道、共享内存失去意义。类 docstring 写明铁律，`__reduce__` 对整对象 pickle **直接抛错**（防 `put(task)` 手滑）。`to_bytes/from_bytes`、`finished` 字段已删除。
- **web→前端 WS**（`api/websocket.py`，下行）：`loaded`（`value` 透传）/ `data`（评测结果，含 `face` 人脸状态）/ `end` / `error`；上行 `offer`/`ice`/`config`。

## 4. 视频链路（4 槽共享内存背压）

浏览器 WebRTC 上行 → `handle_video` → `push_frame`（锁内取空闲槽 → 写 SHM → 投 task 信封；**4 槽满则丢帧**，有 `帧统计` 5s 日志：入队/丢弃/结果/空结果）→ worker `link_frame()` 按名 attach → `sport.process()` **就地绘制** → `result`（slot 回执）→ `_on_result`（槽先归还、读 SHM 拷贝 → `output_track.push_frame` → 浏览器）。会话收口只回收槽，不杀进程（复用关键）。

## 5. 体测项目层（`sports/`）

- `core/registry.py`：字符串懒注册表（`{'situp': 'sports.situp.SportSitup', …}`），**web 零体育代码**——类解析只发生在用户子进程内（importlib）。
- `sports/situp.py`（仰卧起坐，YOLO Pose / COCO17）：`Inferencer` **每帧无状态**（`process` 对 `self` 零赋值，故 `reset` 无需清它）；选人 = 画面中心距离 + 框面积加权（`_select_person`）；`Evaluator` **七态状态机**（2026-11-05：`WAIT` 等坐下 → `FACE` 人脸中 → `READY` 过人脸等躺下 → `FLAT` 躺下合格进测试 → `RISE`/`WAIT_FLAT`/`FINISHED`，+ 计次 + 6 类作弊标记 + 助力检测；原 `face_done` 标记由 `READY` 状态取代）：`WAIT` 经 `_sit_ok`（臀贴地：不高于踝 40px；双膝高于踝 20px 即屈膝）→ `FACE` 经 `_face_admitted`→`face_recognition` → `READY`；躺下合格（肩/头/脚贴地线连续 10 帧 + 躯干-脚底夹角差≤10 再 10 帧）→ `FLAT`（`ready` 返回值随 `FLAT` 置真，前端倒计时触发时刻不变）；地面线拟合只在 `READY` 阶段更新，坐姿与测试期冻结。`SportSitup` 链式封装（inferencer→evaluator），自身不持任何队列/管道；人脸结论经 `status()` 随返回值 `data.face`（`state`: checking/passed/failed/timeout、`fail_count` 会话内失败/超时累计——**前端第 3 次起才显示"未通过"**，前两次静默重试）逐帧给前端——不走消息推送（`phase` 信封已整体取消，见 §3）；hints 按状态出"请坐下…"→"请平躺…"。`FaceClient` 由 worker 启动时实例化（`BaseSport.set_face_client` 注入）、每会话 `reset()`；应答按**学号+毫秒时间戳**双键过滤；ADR-0005。
- `sports/pullup.py`（引体向上）：**YOLO Pose COCO 17**（第 8 步完成，与 situp 共用 `POSE_YOLO_*` 权重、`_select_person` 中心+面积加权选人、骨架/关键点绘制；输出含 `person_boxes/selected_idx`）。评测器 active 索引已重映射（杠线→双腕 9/10、脚→踝 15/16、肘角→10-8-6、肩 5/6、耳 3/4、髋 11/12、`record_keypoints`→0/5/6/11/12），visibility=YOLO 置信度与原 MediaPipe 同量纲。**评测逻辑与死代码按用户指示原样保留**（注释中的旧 33 索引、不可达 elif 分支、State/k1/k2、`PULLUP_STATIC_IMAGE_MODE` 常量均未动）。
- `core/config.py`：**统一配置，零副作用**（不配 logging——唯一日志配置点 `core/log.py`；不读 Django settings）。路径全部绝对锚定（`VENDOR_DIR` 等），模块级 `sys.path.insert(VENDOR_DIR)` 使 `import face_sdk` 可用。`IDLE_TIMEOUT=60` 双端同口径；`POSE_YOLO_*` 两项目共用；`FACE_GATE_ENABLED`（默认关）/ `FACE_GATE_TIMEOUT=10` 人脸门禁开关。

## 6. 资产与依赖

- `sport-backend/vendor/`（唯一资产区）：`yolo_pose/`（YOLO 权重）、`wcx/`（第三方整树，jump 用）、`face_sdk/`（代码+`models/` 166MB onnx 等）、`fonts/`（msyh×2）。
- `sport-backend/data/faces.db`：人脸库（运行时可写，不进 vendor；`init_sdk(db_path=…)` 由 config 传入）。
- 报告读写同处 `src/report_pdf/`（`REPORT_DIR` 锚定，修过"官方入口读写分叉"）；PDF 生成器字体注册在 **Django import 期**走 `FONT_DIR`。
- `onnxruntime==1.31.1` 已入 `requirements.txt`（face_sdk 硬依赖）。
- **启动必须走 `main.py` 入口**（chdir + 托管，预热可选）；`start_me.bat` daphne 直启已废弃；`start_with_env.bat` 已删（死脚本）。

## 7. 关键注意事项

- **测试环境观察**：supervisor/proc 的停止**不触发** Python 信号/atexit——验证 `shutdown_all` 要进程内驱动，不能靠停服务看日志。
- **spawn 子进程会重跑主模块顶层**：任何含 `mp.Process` 的脚本必须 `if __name__ == '__main__'` 守卫（`main.py` 有；临时测试脚本栽过两次）。
- `legacy/` = 死代码簇（import 已断，如 `api.base_sport` 不存在），**允许断**，整体删除未排期。
- 本批已识别、不修：jump `sam2.1_hiera_small.pt` 缺失（checkpoints 空）、人脸注册/录入流程、pullup/standjump 页的预览切换。
- 前端 `sport-frontend/src/views/student/StudentTestingSitup.vue`：单 `<video>`，`ontrack` 只存不挂，`loaded(value:true)`+轨道双条件齐了才切（第 6 步定型；**5 秒断流回退已暂缓**）；`loaded` 只改文案，读秒由 `data.ready` 上升沿触发；人脸提示由 `data.face.state` 变化驱动（checking/passed/failed/timeout）。

## 8. 进度与文档索引

- **`docs/TODO.md`**：8 步改造序列——**第 1–8 步全部 ✅**（config 统一 / vendor 归位 / 四段 rename / 进程复用池 / 助力检测 / 前端预览双条件切换 / 人脸门禁 / pullup 换 YOLO，每步有偏差记录与验证证据）；序列外遗留见 TODO 末尾"不进本批"清单。
- `docs/adr/`：0001 web 直拉用户进程、0002 托管 daphne 入口、0003 驻留池、0004 人脸单任务无排队、0005 人脸应答专用管道直读。
- 仓库根 `GLOSSARY.md`：领域词汇（驻留窗口/空闲进程/四段生命周期/人脸门禁/单任务识别/助力/选人/ready 等）。
