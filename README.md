# 体测自动化系统后端

单机多进程：浏览器经 WebRTC 上传视频，**每个受测用户一个独立进程**完成姿态推理与计次评测，绘制后的结果帧与成绩回传前端。

## 启动

```bat
cd sport-backend
start.bat        :: 等价于 python src/main.py
```

- daphne/Django 监听 `8090`（`DAPHNE_PORT` 可覆盖）。
- 启动即预热：spawn 一次性进程加载 situp/pullup 模型打热文件缓存（日志带"预热"字样）。
- Ctrl+C / SIGTERM 退出时自动销毁全部用户会话与共享内存槽。

## 目录结构

| 路径 | 职责 |
|---|---|
| `src/main.py` | 唯一入口：轻顶层 + guard 内程序化托管 daphne、清理收口、开机预热（ADR-0002） |
| `src/core/` | 契约层：`task.py` 任务消息（唯一定义）、`shm.py` per-user 共享内存槽、`registry.py` 字符串懒注册表、`log.py` |
| `src/sports/` | 体测项目封装：`base.py`（BaseSport）、`situp.py`（SportSitup）、`pullup.py`（SportPullup） |
| `src/worker/` | `run_user_worker` 用户进程主循环（phase/result/finished/error 单一出口） |
| `src/api/` | Django 应用：`views`/`websocket`/`user_manager`（IPC 客户端）等 |
| `src/backend/` | Django project（settings/routing/asgi） |
| `src/legacy/` | 旧版死代码簇（`new_*`、`sport_manager` 等，待拍板删除） |
| `docs/` | `refactor_plan.md`（改造计划与决策表）、`multiprocess_architecture.md`（旧架构现状记录）、`adr/` |

## 架构要点

- **进程归属**：web 入口进程直接 spawn 用户进程，无 Manager 进程、无全局队列、无 `cmd_queue` 扩缩容（ADR-0001）。
- **IPC**：双 `mp.Queue`（帧入/结果出，随 spawn 继承）+ 每会话 4 槽共享内存同槽就地绘制回传；背压 = 4 槽占满即丢帧（loading 期同理）。
- **生命周期**：显式 stop、WS 断线、60s 空闲、进程崩溃（error 消息）四路径销毁，`_finalize` 幂等收尾。
- **注册表**：`core/registry.py` 字符串懒注册，web 只读 keys；启动预热进程遍历全解析（结构性保证 web 零体育代码）。
- **扩展新项目**：`sports/` 加一个文件 + `registry.py` 加一行字符串，其余全自动（预热自检、keys 校验、error 通道）。

术语见仓库根 `GLOSSARY.md`。
