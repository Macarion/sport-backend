@REM 使用WebSocket必须要使用daphne启动

daphne -p 8090 backend.asgi:application