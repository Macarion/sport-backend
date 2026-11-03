---
status: accepted
---

# 用户进程由 Web 入口进程直接创建

旧架构用轻量 Launcher 托管 Manager/全局队列/推理进程池/评测单点；改造决定改为每个受测用户一个进程（推理与评测合一），由 Web 入口进程（`src/main.py` 托管的 daphne）作为唯一父进程直接 spawn，帧与结果经随 spawn 继承的双 `mp.Queue` 和 per-user 共享内存槽传递，彻底移除 Manager 进程、`AUTH_KEY`/`MANAGER_PORT` 配置与 `cmd_queue` 扩缩容。理由：用户进程天然保序、生命周期由父进程直管（stop/断线/空闲/崩溃四路径销毁）、每帧少两次 socket RPC；实测空 spawn ~80ms、Windows 非主线程 spawn 可用，启动成本由子进程懒加载模型（`start()` 内）+ 开机预热吸收。

## Considered Options

- **保留轻量 Launcher**：situp+pullup 全迁后只剩 spawn 一个职责，Manager 每帧 RPC 与 web/worker 双端配置割裂仍然保留。
- **模型常驻服务进程（推理池）**：实测 CPU batch=4 每图仅省 22%、机器无 GPU（torch cpu 构建），无吞吐优势，且拆散推理/评测合并的前提。
- **子进程直连 WebRTC 媒体面**：省掉的仅 2 次 memcpy（~37MB/s @20fps，对比 48ms/帧的 YOLO 可忽略），代价是信令中继与媒体/控制双平面生命周期同步。worker 接口保持帧源无关，留作未来可插拔优化。

## Consequences

- 未注册的体测项目（如 jump）由 keys 校验直接返回 error、不 spawn：`/api/standjump/start/` 从"崩溃重启循环"变为干净失败（适配延后）。
- 子进程可靠性依赖 ADR-0002 的入口形态。
- 容量按 CPU、≤8 并发设计（每用户进程实测 ~324MB）。
