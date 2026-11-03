@echo off
conda activate sport && daphne -p 8090 backend.asgi:application