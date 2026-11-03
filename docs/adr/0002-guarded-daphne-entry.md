---
status: accepted
---

# 以受保护入口程序化托管 daphne

multiprocessing spawn 的子进程会重新执行父进程入口模块的顶层代码（`multiprocessing/spawn.py` 的 `__main__` fixup）；`python -m daphne` 的入口 `daphne/__main__.py` 三行无 `if __name__ == "__main__"` 保护，子进程会先去重启 daphne（端口冲突或直接退出，实测 exitcode 1/3、target 永不执行），父侧管道 `recv()` 永久阻塞——这是历史上"在 web_server 里创建进程会卡住"的实测根因。决定：改用 `src/main.py` 作为唯一入口，顶层只留轻量 import，`if __name__ == "__main__"` 内完成重依赖 import、程序化启动 daphne（等价 `-p 8090 backend.asgi:application`）、信号与清理收口、开机预热（日志带"预热"字样）；start.bat 与 Linux 启动统一为 `python src/main.py`。

## Considered Options

- **给 daphne 打 `__main__` 补丁**：daphne 升级即失效。
- **monkeypatch `multiprocessing.spawn` 的 prep 数据**：绑定 CPython 内部实现，版本脆弱。

## Consequences

- **入口顶层必须保持轻**：入口顶层每 1 秒 import，每个子进程启动就慢 1 秒（实测重型顶层父 1.96s / 子 1.89s，start→target 1964ms）。
- 不再使用 daphne CLI 参数（原仅用 `-p`，程序化等价）。
- 开机预热（spawn 一个加载完即退出的进程打热文件缓存）在本入口内完成。
