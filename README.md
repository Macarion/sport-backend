# Sport 后端

## 配置

### 更改后端默认使用的摄像头（默认为0）

- 当前，仰卧起坐、引体向上、立定跳远使用前端网页的摄像头，坐位体前屈使用后端的摄像头
```
# api/config.py

CAMERA_INDEX = 0
```

## 启动

### 安装 Python 依赖

```
pip install -r requirements.txt
```

### 安装 ffmpeg

- Windows: 下载 [ffmpeg](https://ffmpeg.org/download.html) 并添加到环境变量中
- Linux: 
```
sudo apt-get install ffmpeg
```

### 添加Vendor文件

- 立定跳远的依赖文件，放在`api/vendor`目录下

### 启动后端

```
daphne -p 8090 backend.asgi:application
```