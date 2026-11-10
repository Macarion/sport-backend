# Jump 多进程接入说明

实施日期：2026-11-09。两个开发 worktree 均从成功 fetch 的 `origin/multiprocess` 完整检出，分支名均为 `codex/jump-multiprocess`。增量改动保留在工作区，未提交、未推送、未合并。

| 仓库 | fetch 后的基线提交 | 开发目录 |
|---|---|---|
| https://github.com/Macarion/sport-backend | `463f4c7fef31e35f0896e2ec3db8ada914e1cb94` | `C:\Users\Rational\.codex\worktrees\jump-multiprocess-backend\backend` |
| https://github.com/Macarion/sport-vue | `9f07e42f4ec19c12df71178406469c689f380c48` | `C:\Users\Rational\.codex\worktrees\jump-multiprocess-frontend\frontend` |

原 `D:\Python_Project\ysj-project` 下前后端的 `wcx` 分支、未提交修改，以及 `D:\Python_Project\wcx_jump` 参考项目均保留。切换到上述新目录开发，不要在旧目录启动本次实现。

## 实现范围

- 新增 `sports.jump.sport.SportJump`，通过字符串注册表进入远端现有 spawn worker。每进程加载、预热两份 YOLO 模型；每次 reset 后首帧建立独立 uid/session_id/输出目录。
- 保留远端 Web 状态机和原 ArUco/选帧/测距规则。远端缺失的算法依赖取自旧后端 `vendor/wcx`，保留所需函数原实现，删去未使用的 CLI 和旧模型加载器。没有迁入本地新选帧、16 角点标定、去畸变或参数 UI。
- 迁入本地 YOLO 分割适配器，默认整图、640、conf 0.25、NMS IoU 0.7、目标框匹配 IoU 0.1、FP32、max_det 300、retina_masks=True。输出原尺寸 uint8 0/1 mask；`iou_predictions=[]`，实例检测置信度不作为 mask 精度。
- 缺少权重、模型任务/类别错误、空 mask、匹配不到当前测试人均明确失败。运行不导入 `wcx_jump` editable 包，不需要 SAM2。首次推理预热在 `loaded(True)` 之前，避免短视频前段被初始化阻塞。
- 输入仍走 HTTP start/stop、`/ws/webrtc/<uid>/`、队列和共享内存。上限 1280×720，期间改变尺寸会报错。收帧时间戳进入状态机；异步缓存复制像素；返回图像复制后才归还共享内存槽。
- 计算期间继续收发视频；完成后持续绘制原分辨率落地帧、目标框、脚跟及厘米成绩。JSON 保留会话、状态、统计和测量/实例元数据，不带 base64 图像。
- worker 在 finished 前调用 end_session。Jump 取消计算、退出编码/日志线程，释放锁后等待，清理总预算 4 秒；超时发送 error 并退出，不归池。阻止停止后发布结果，防止 task 排在 stop 之后；旧 WebSocket 断线不会停止新消费者的会话。
- ArUco 和准备条件满足后、ARMED 前调用原 FaceClient。提交未绘制人物裁剪的副本；已注册必须识别通过，未注册提示后放行；失败和超时重试。目标丢失/明显更换清理识别状态。请求戳跨 reset 单调递增，隔离同一毫秒换人时的迟到应答。
- 页面区分接口启动、模型加载、连接及检测状态；本地预览等模型就绪、远端轨道和处理数据齐全再切换。连接超时 30 秒，模型加载超时 120 秒。统一关闭 RTC/WS/摄像头/录制器，忽略旧回调；成绩按 session 去重，空值显示 `-`，录像扩展名与实际 MIME 一致。

## 起跳规则同步（2026-11-10）

Web 已同步本地 `wcx_jump/realtime_fast_ready.py` 的默认起跳规则：

- `FastReadyConfig.dynamic_armed_baseline=True`：人脸放行、进入 ARMED 时用准备帧初始化最近 5 帧窗口。仅接纳 `active_y2 >= baseline_y2 - threshold * 0.5` 的观测，取窗口最大值修正基线，只增不减。
- `single_frame_takeoff_ratio=2.5`：保留原连续两帧上升判定，或单帧上升达到普通阈值的 2.5 倍时触发。普通阈值仍为 `max(30px, 图像高度 * 0.03)`，720p 默认单帧阈值为 75px。
- 新增诊断字段记录修正前基线、是否修正及 `takeoff_reason=two_frame|single_frame_strong|none`。人脸核验、超时优先级、丢人清空连续帧、起跳后 2 秒窗口、旧落地选帧和测距函数均保留。起跳可能按新规则提前，这是预期行为。

验证：全套 46 项后端测试（含新增 12 项起跳/门禁/时序测试）通过；10,000 次观测与本地动态基线实现一致。真实 spawn worker + 共享内存的 720p/15fps 回放中，test5 起跳帧 59→58、落地帧保持 63、成绩保持 192.74745178222656cm；test8 起跳帧 17→16、落地帧保持 21、成绩保持 178.1114501953125cm。两片新旧版本均走完整状态链、提交丢帧为 0、进程正常退出。视频回放关闭人脸，门禁时序由单测覆盖；本轮未进行现场摄像头实测。完整对照在 `outputs/validation/takeoff-upgrade/comparison.json`，规则来源哈希记录在 `jump-source-manifest.json`。

后端无自动热重载。更新后重启后端再开启测试，以确保驻留 worker 使用新规则；前端无需重新构建。

## 资源和环境

源文件、权重及字体 SHA256 见 [jump-source-manifest.json](jump-source-manifest.json)。`sha256` 是迁入前原文件哈希，`target_sha256` 是适配后的目标文件哈希。参考项目尚无 Git 提交，以文件哈希追溯。原有绘制函数取自远端遗留 new_jump 页面服务代码。

| 资源 | 开发目录中的位置 | 当前状态 |
|---|---|---|
| YOLO11 检测 | `vendor/jump/yolo11n.pt` | 已从本地复制，校验 detect/person 0 |
| YOLO11 分割 | `vendor/jump/yolo11n-seg.pt` | 已从本地复制，校验 segment/person 0 |
| ArUco/Web 辅助算法 | `vendor/wcx/` | 已迁入，随代码交付 |
| YOLO Pose（公共项目） | `vendor/yolo_pose/yolov8n-pose.pt` | 远端已有 |
| 字体 | `vendor/fonts/msyh.ttf`、`msyhbd.ttf` | 已从旧后端复制 |
| 人脸 SDK、模型、注册库 | `vendor/face_sdk/`、其 `models/`、`data/faces.db` | 已从用户提供的 sdk.zip 部署；原库 1 条记录保留 |

权重、字体、运行输出和人脸资源由 `.gitignore` 排除，当前 worktree 内已放好前两类资源。迁往其他机器需按清单另行提供，不会自动下载或回退模型。

已验证环境为 `jumppy12` / Python 3.12.7 / RTX 4060 8GB，关键版本见 [jump-tested-environment.json](jump-tested-environment.json)。复用现有环境；本次补装远端 requirements 已要求但环境缺少的 pyttsx3 2.99（及其 comtypes 依赖）和 onnxruntime 1.31.1，未重新安装整份历史 requirements，也未安装参考项目 editable 包。

## 启动

后端 PowerShell（在新 backend 根目录）：

```powershell
conda activate jumppy12
# 人脸资源已部署，开启真实人脸服务。
$env:FACE_GATE_ENABLED = '1'
$env:DAPHNE_PORT = '8090'
$env:PYTHONUTF8 = '1'
python src/main.py
```

人脸资源已经配置，`FACE_GATE_ENABLED=1` 启动真实人脸服务；只有专门的无门禁回放才显式设为 `0`。现有注册库来自 SDK 包，注册/识别统一使用 `core.config.FACE_DB_PATH`，不要误用 SDK 默认的 faces_db 路径。`JUMP_DEVICE=auto` 沿用现有设备选择行为，也可显式使用 `cpu`。

前端 PowerShell（在新 frontend 根目录）：

```powershell
npm ci
$env:VITE_API_BASE_URL = 'http://127.0.0.1:8090/'
npm run dev
```

业务 HTTP 和 Jump WebSocket 共用该环境地址；默认 HTTP 启停接口和信令路径保持不变。日常使用登录后的立定跳远页面。`tests/jump-browser.html` 仅是本地验收入口，不进入生产构建。

## 验证结果

| 验证 | 结果 |
|---|---|
| Django `python src/manage.py check` | 通过 |
| 后端 unittest | 34 项通过（迁移 mask/选人/校验、取消、身份隔离、共享内存、10 次 worker 会话及同 uid 立即重启、驻留退出、故障路径） |
| 同一落地帧和目标框模型对照 | 新后端与本地参考独立进程运行：实例选择、mask SHA256、厘米测量完全相同，192.9953 cm；无 SAM2 导入 |
| `test5.mp4`，720p / 15 fps 实时回放 | CALIBRATING → WAITING_READY → ARMED → IN_JUMP → PROCESSING → RESULT；192.7475 cm；99 接收 / 99 提交 / 0 丢帧；落地帧 63；干净退出 |
| `test8.mp4`，720p / 15 fps 实时回放 | 同完整状态序列；178.1115 cm；58 接收 / 58 提交 / 0 丢帧；落地帧 21；干净退出 |
| 真实 Chrome + 录像摄像头 + WebRTC + 后端 | 连续两轮 RESULT：191.5639 / 191.8568 cm，session_id 不同，复用同一个 worker，停止后 tracks ended、视频引用清空，无页面异常 |
| 真实模型生命周期/公共回归 | Jump、situp、pullup 各 10 次 load 后 reset/process/end；模型保持同一实例；图像绘制正常；Jump 会话线程全部退出，无新增残留线程 |
| 前端连接单测 | Node 5 项通过，覆盖对象消息、连接/模型独立超时、error/end、取消与迟到回复 |
| 页面模拟验收 | 本地/远端预览切换、重复启动禁止、人脸提示和语音门禁、未注册放行、成绩去重、空成绩、停止与迟到回复均通过 |
| 前端 ESLint / production build | 修改文件 lint 通过，构建通过；保留原有大 bundle 提示 |

浏览器回放由 canvas 摄像头在 ARMED 后播放录制视频，验证真实 WebRTC、编码、共享内存和页面链路；没有使用实体摄像头，也没有执行真人身份识别。WebRTC 重编码和实时采样会改变选中的帧，所以两次浏览器结果可不同；固定帧的模型迁移对照要求完全一致且已通过。

最初 test8 因首次模型推理阻塞导致前段丢帧、起跳超时；加载阶段预热后原起跳阈值不变，两段最终回放均为 0 丢帧。

详细运行报告/截图在本 worktree `outputs/validation/`：`test5-final/`、`test8-final/`、`browser-final/`、`parity/`、`lifecycle/`、`page/`。原始落地帧、mask、成绩文本在报告所指的 `outputs/<uid>/<session_id>/`；逐帧诊断在 `temp/jump_diag_*.jsonl`。短片结束后回放脚本重复末帧以等后台结果，`tail_submitted` 与视频提交数分开计数。

可重复运行（backend 根目录）：

```powershell
python -m unittest discover -s tests -v
python scripts/replay_jump.py D:/Python_Project/wcx_jump/inputs/test5.mp4 --output outputs/validation/test5
python scripts/replay_jump.py D:/Python_Project/wcx_jump/inputs/test8.mp4 --output outputs/validation/test8
python scripts/smoke_sports.py --video D:/Python_Project/wcx_jump/inputs/test5.mp4 --output outputs/validation/lifecycle
python scripts/verify_jump_models.py --report outputs/validation/test5/report.json --reference D:/Python_Project/wcx_jump --output outputs/validation/parity
```

`--reference` 仅在对照测试的独立子进程中读取本地参考源码；生产入口和真实模型 smoke 均断言未导入 `wcx_jump`。

## 未完成的现场验收

人脸 SDK、模型和注册库已部署，包内样例及真实人脸服务验证已通过；仍需注册学生真人识别及现场换人重测。实体摄像头跳远、运动计数准确性和多路 GPU 容量不在本次自动验证结果内。本轮容量只验证单测位复用，不据此承诺并发 GPU 负载。

为让远端基线可运行，同步修复了 `views.py` 的 sitreach 包路径、前端 ESLint globals 位置，以及 StudentUser.vue 未使用却会使 Vite 开发依赖扫描失败的 uuid 导入。


## 人脸资源部署追加记录（2026-11-09）

用户提供的 `D:/Python_Project/ysj-project/require/sdk.zip` 和 `face_integration.zip` 已分别解压到同目录的 `face_sdk/` 和 `face_integration/`。运行使用 sdk.zip 的接口版本；模型为 deploy.prototxt、res10_300x300_ssd_iter_140000.caffemodel、w600k_r50.onnx。三份模型与另一压缩包的模型逐字节哈希一致。部署路径及来源哈希见 [face-resource-manifest.json](face-resource-manifest.json)。未复制 Python 3.13 的缓存字节码。

`jumppy12` 已安装 onnxruntime 1.31.1，现有 NumPy、OpenCV、PyTorch 未升级。人脸特征模型使用 CPUExecutionProvider，Jump YOLO 保持原 CUDA 运行方式。

验证命令：

```powershell
python scripts/check_face_runtime.py --image D:/Python_Project/ysj-project/require/face_integration/test/test_dengchao.jpg
```

实际加载模型成功；包内演示图与现有学号 23281166 的档案匹配（相似度 1.0）；临时数据库注册/查询/同图核对通过；空白图被拒绝；真实 spawn 人脸服务完成未注册放行、已注册失败、已注册匹配的 IPC 收发。生产 `data/faces.db` 哈希未变，仍为原来的 1 条记录。该演示数据测试不代表现场真人验收。

本次运行地址：前端 `http://127.0.0.1:5173/`，后端 `http://127.0.0.1:8090/`，门禁开启。后端输出在 `temp/backend-runtime.out.log` 和 `temp/backend-runtime.err.log`。资源、模型、数据库和压缩包均未提交到 Git。


开启人脸门禁后的浏览器验收：先将 test5.mp4 转为 1280×720 / 15fps，运行前端脚本加 `--expect-face`，两轮分别为 197.8472 / 198.7189 cm，session_id 独立、同进程复用、视频流停止释放正常，均收到实际人脸服务的 `passed / registered=false`（测试账号未注册）。报告位于 `outputs/validation/browser-face-720p/`。首轮直接播放原始 4K 文件时错过了连续起跳帧并超时；720p 素材重测通过，未调整业务判断阈值。
