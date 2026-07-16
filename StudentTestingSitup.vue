<template>
  <div class="container">
    <!-- 摄像头和图表区域 -->
    <div class="camera-chart-container">
      <div class="camera-box">
        <video ref="resultVideo" autoplay muted playsinline style="height:100%;width:100%;object-fit:contain" />
      </div>
      <div class="chart-box">
        <div style="flex: 1;">
          <h3 style="margin: 0;">分数</h3>
          <div style="text-align: center;">
            <div class="count-display">{{ num }}</div>
          </div>
        </div>
        <div style="flex: 1;">
          <h3 style="margin: 0;">状态</h3>
          <div style="text-align: center;">
            <div class="hints-area">
              <div v-for="(hint, idx) in hints" :key="idx" class="hint-text">{{ hint }}</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 控制按钮区域 -->
    <div class="control-buttons">
      <el-button type="primary" @click="start" :disabled="cameraActive">开启摄像头</el-button>
      <el-button type="danger" @click="stop" :disabled="!cameraActive">关闭摄像头</el-button>
      <el-select v-model="selectedDeviceId" placeholder="选择摄像头" style="width: 15rem; margin:0 10px;" :disabled="cameraActive">
        <el-option v-for="cam in cameraList" :key="cam.deviceId" :label="cam.label || `摄像头${camIndex++}`" :value="cam.deviceId"/>
      </el-select>
      <el-button @click="login">返回</el-button>
      <div style="float: right; display: flex; gap: 10px;">
        <el-checkbox v-model="drawingEnabled" @change="onDrawingToggle">绘图开关</el-checkbox>
        <el-button type="success" @click="downloadOriginVideo" :disabled="cameraActive || !hasRecordedVideo">下载原视频</el-button>
      </div>
    </div>

    <!-- 信息展示区域 -->
    <div class="info-container">
      <el-row :gutter="20" class="info-row">
        <el-col :span="8">
          <div class="info-card">
            <h3>个数记录</h3>
            <div class="info-content">
              <div>
                <span class="count-info">有效成绩：</span>
                <span class="count-display">{{ num }}</span>
              </div>
              <hr width="30%" color="green" size="3px" align="center" />
              <div>
                <span class="count-info">总个数：</span>
                <span class="count-display">{{ num_all }}</span>
              </div>
            </div>
          </div>
        </el-col>
        <el-col :span="8">
          <div class="info-card">
            <div class="chart-section">
              <canvas ref="countChartElement"></canvas>
            </div>
          </div>
        </el-col>
        <el-col :span="8">
          <div class="info-card">
            <div class="chart-section">
              <canvas ref="angleChartElement"></canvas>
            </div>
          </div>
        </el-col>
      </el-row>
    </div>
  </div>
</template>

<script setup>
import {
  start_monitor_situp,
  stop_monitor_situp,
} from '@/api/user.js'
import { ref, onMounted, onUnmounted } from 'vue'
import Chart from 'chart.js/auto'
import { useRouter } from 'vue-router'

// 视频参数配置
let mediaParams = {
  video: {
    width: { ideal: 640 },
    height: { ideal: 480 },
    frameRate: { ideal: 15, max: 30 },
  },
  audio: false,
}

let timer1 = null
let timer2 = null

const countChartElement = ref(null)
const angleChartElement = ref(null)
const cameraActive = ref(false)
let num = ref(0)
let num_all = ref(0)
let count = ref(0) // 用于显示个数
let countChartInstance = null
let angleChartInstance = null
const router = useRouter()

let timestamps = []
let angles = []
let nums = []
const isRegister = ref(false)
const detectStarted = ref(false)

// 摄像头列表与选中设备
const cameraList = ref([])
const selectedDeviceId = ref('')
let camIndex = 1
const resultVideo = ref(null)

let webrtc
let mediaStream = null;
let mediaRecorder = null
let recordedChunks = []
let hasRecordedVideo = ref(false)

const username = localStorage.getItem('username')
console.log('获取到的username:', username)

// 绘图开关
const drawingEnabled = ref(true)
const hints = ref(["请点击开始按钮"])

const onDrawingToggle = (val) => {
  if (webrtc && webrtc.ws && webrtc.ws.readyState === WebSocket.OPEN) {
    webrtc.ws.send(JSON.stringify({
      type: 'config',
      drawing: val,
    }))
  }
}

const start = async () => {
  if (cameraActive.value) return // 防止重复开启
  isRegister.value = true
  num.value = 0
  num_all.value = 0
  count.value = 0
  nums = []
  angles = []
  timestamps = []
  hints.value = []
  detectStarted.value = false

  // 清除已有定时器
  if (timer1) clearInterval(timer1)
  if (timer2) clearInterval(timer2)
  timer1 = null
  timer2 = null

  // 清除图表数据
  if (countChartInstance) {
    countChartInstance.destroy()
    countChartInstance = null
  }
  if (angleChartInstance) {
    angleChartInstance.destroy()
    angleChartInstance = null
  }

  // 清除图像预览
  cleanup()

  try {
    // 初始化新图表
    initCountChart()
    initAngleChart()

    const res = await start_monitor_situp(username)
    console.log('启动摄像头返回:', res)

    if (res.status === 200) {
      startRecord(username)

      cameraActive.value = true
    }
  } catch (err) {
    console.error('启动摄像头失败:', err)
    cameraActive.value = false
    isRegister.value = false
  }
}

const stop = async () => {
  speakMessage('测试已结束，您的成绩为：' + num.value + '个')
  cameraActive.value = false
  try {
    stopRecord()

    const res = await stop_monitor_situp(username)
    console.log('关闭摄像头返回:', res)
  } catch (err) {
    console.error('关闭摄像头失败:', err)
  }
}

// 加载所有摄像头设备列表
const loadCameraList = async () => {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices()
    cameraList.value = devices.filter(d=>d.kind === 'videoinput')
    if(cameraList.value.length) selectedDeviceId.value = cameraList.value[0].deviceId
  } catch (err) {
    console.error('获取摄像头列表失败', err)
  }
}

// 打开摄像头
const openCamera = async () => {
  // 如果选中了摄像头deviceId则指定设备
  if (selectedDeviceId.value) {
    mediaParams.video.deviceId = { exact: selectedDeviceId.value }
  }
  mediaStream = await navigator.mediaDevices.getUserMedia(mediaParams);
  resultVideo.value.srcObject = mediaStream;
}

const startLocalRecording = () => {
  // 初始化视频录制
  recordedChunks = []
  mediaRecorder = new MediaRecorder(mediaStream, {
                    mimeType: 'video/mp4'
                })
  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > -1) recordedChunks.push(e.data)
  }
  mediaRecorder.start()
}

const stopLocalRecording = () => {
  // 停止视频录制
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop()
  }
  hasRecordedVideo.value = true
}

// 建立 WebRTC 与 WebSocket 连接
const createWebRTC = async (uid) => {

  let pc;

  const ws = new WebSocket(
    `ws://127.0.0.1:8090/ws/webrtc/${uid}/`
  )

  ws.onopen = async () => {

    await openCamera()
    startLocalRecording()

    // 创建 WebRTC RTCPeerConnection 对象
    pc = new RTCPeerConnection({
      iceServers: [
        { urls: 'stun:stun.l.google.com:19301' },
        { urls: 'stun:stun.miwifi.com:3477' },
      ]
    })

    // 添加本地视频流
    mediaStream.getTracks().forEach(track => {
      pc.addTrack(track, mediaStream)
      if (track.kind === 'video') {
        track.contentHint = 'detail'
      }
    })

    // 接收后端回传的视频流
    pc.ontrack = (evt) => {
      console.log("收到后端回传视频流", evt);
      resultVideo.value.srcObject = evt.streams[0];
      speakMessage('测试开始')
    };

    // 接收后端回传的ICE候选
    pc.onicecandidate = (e) => {
      if (e.candidate) {
        ws.send(JSON.stringify({
          type: "ice",
          candidate: e.candidate
        }))
      }
    }

    // 发送 WebRTC offer
    const offer = await pc.createOffer()

    await pc.setLocalDescription(offer)

    ws.send(JSON.stringify({
      type: "offer",
      sdp: offer.sdp
    }))
  }

  // WebSocket 接收消息
  ws.onmessage = async (event) => {

    const msg = JSON.parse(event.data)

    if (msg.type === "answer") {

      // WebRTC 回传的应答消息
      await pc.setRemoteDescription({
        type: "answer",
        sdp: msg.sdp
      })
    }
    else if (msg.type === "eval_result") {

      // 接收评测数据并展示
      const result = msg.data

      num.value = result.score
      num_all.value = result.num_all
      hints.value = result.hints || []
      if (hints.value.length === 0) {
        hints.value = ["正常"]
      }

      timestamps.push(Number(result.timestamp))
      nums.push(Number(result.score))
      angles.push(Number(result.angle))

      updateCharts()
    }
    else if (msg.type === "test_end") {
      console.log("测试结束")
    }
  }

  ws.onclose = () => console.log("RTC WS连接关闭");
  ws.onerror = err => console.error("WebSocket错误", err);

  return { pc, ws }
}

const startRecord = async (uid) => {
  webrtc = await createWebRTC(uid)
}

const stopRecord = () => {
  // 停止录制
  stopLocalRecording();
  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop());
    mediaStream = null;
  }
}

// 下载原视频方法
const downloadOriginVideo = () => {
  if (!recordedChunks.length) return
  // 生成文件名 situp_年月日_时分秒
  const now = new Date()
  const year = now.getFullYear()
  const month = String(now.getMonth() + 1).padStart(2, '0')
  const day = String(now.getDate()).padStart(2, '0')
  const hour = String(now.getHours()).padStart(2, '0')
  const minute = String(now.getMinutes()).padStart(2, '0')
  const second = String(now.getSeconds()).padStart(2, '0')
  const fileName = `situp_${year}${month}${day}_${hour}${minute}${second}`

  const blob = new Blob(recordedChunks, { type: 'video/mp4' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `${fileName}.mp4`
  a.click()
  URL.revokeObjectURL(url)
}

const login = async () => {
  router.push('/student/test')
}

const initCountChart = () => {
  const ctx = countChartElement.value.getContext('2d')
  if (countChartInstance) {
    countChartInstance.destroy()
  }
  countChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: timestamps.map((ts) => formatTimestamp(ts)),
      datasets: [
        {
          label: '仰卧起坐个数',
          data: [],
          borderColor: '#2d7acd',
          backgroundColor: 'rgba(45, 122, 205, 0.2)',
          fill: true,
          tension: 0,
          pointRadius: 0
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: {
            display: true,
            text: '时间',
          },
          ticks: {
            autoSkip: true,
            maxRotation: 0,
          },
        },
        y: {
          title: {
            display: true,
            text: '个数',
          },
        },
      },
      plugins: {
        tooltip: {
          callbacks: {
            label: function (context) {
              return `${formatTimestamp(timestamps[context.dataIndex])}: ${context.raw}`
            },
          },
        },
      },
    },
  })
}

const initAngleChart = () => {
  const ctx = angleChartElement.value.getContext('2d')
  if (angleChartInstance) {
    angleChartInstance.destroy()
  }
  angleChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: timestamps.map((ts) => formatTimestamp(ts)),
      datasets: [
        {
          label: '起伏角度',
          data: [],
          borderColor: '#67c23a',
          backgroundColor: 'rgba(103, 194, 58, 0.2)',
          fill: true,
          tension: 0,
          pointRadius: 0
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: {
            display: true,
            text: '时间',
          },
          ticks: {
            autoSkip: true,
            maxRotation: 0,
          },
        },
        y: {},
      },
      plugins: {
        tooltip: {
          callbacks: {
            label: function (context) {
              return `${formatTimestamp(timestamps[context.dataIndex])}: ${context.raw}°`
            },
          },
        },
      },
    },
  })
}

const updateCharts = () => {
  if (countChartInstance) {
    countChartInstance.data.labels = timestamps.map((ts) => formatTimestamp(ts - timestamps[0]))
    countChartInstance.data.datasets[0].data = nums
    const maxPoints = 100
    if (countChartInstance.data.labels.length > maxPoints) {
      countChartInstance.data.labels = countChartInstance.data.labels.slice(-maxPoints)
      countChartInstance.data.datasets[0].data =
        countChartInstance.data.datasets[0].data.slice(-maxPoints)
    }
    countChartInstance.update()
  }
  if (angleChartInstance) {
    angleChartInstance.data.labels = timestamps.map((ts) => formatTimestamp(ts - timestamps[0]))
    angleChartInstance.data.datasets[0].data = angles
    const maxPoints = 100
    if (angleChartInstance.data.labels.length > maxPoints) {
      angleChartInstance.data.labels = angleChartInstance.data.labels.slice(-maxPoints)
      angleChartInstance.data.datasets[0].data =
        angleChartInstance.data.datasets[0].data.slice(-maxPoints)
    }
    angleChartInstance.update()
  }
}

const formatTimestamp = (timestamp) => {
  if (!timestamp) return ''
  const date = new Date(timestamp)
  return [
    date.getMinutes().toString().padStart(2, '0'),
    date.getSeconds().toString().padStart(2, '0'),
  ].join(':')
}


const cleanup = () => {
  if (resultVideo.value && resultVideo.value.src) {
    URL.revokeObjectURL(resultVideo.value.src)
    resultVideo.value.src = null
  }
}

const speakMessage = (text) => {
  if (!('speechSynthesis' in window)) {
    console.warn('当前浏览器不支持语音播报')
    return
  }
  window.speechSynthesis.cancel()
  const utterance = new SpeechSynthesisUtterance(text)
  utterance.lang = 'zh-CN'
  window.speechSynthesis.speak(utterance)
}

onUnmounted(() => {
  if (timer1) clearInterval(timer1)
  if (timer2) clearInterval(timer2)
  timer1 = null
  timer2 = null
  if (countChartInstance) {
    countChartInstance.destroy()
    countChartInstance = null
  }
  if (angleChartInstance) {
    angleChartInstance.destroy()
    angleChartInstance = null
  }
  cleanup()
  if (cameraActive.value) {
    stop()
  }
})

onMounted(() => {
  initCountChart()
  initAngleChart()
  loadCameraList()
})
</script>

<style scoped>
.container {
  height: 100%;
  display: flex;
  flex-direction: column;
  background: #f5f7fa;
  overflow: hidden;
}

.camera-chart-container {
  flex: 1;
  display: flex;
  min-height: 0;
}

.camera-box {
  flex: 0 0 auto;
  width: 60%;
  aspect-ratio: 16 / 9;
  background: #000000;
  position: relative;
  overflow: hidden;
  display: flex;
  justify-content: center;
  align-items: center;
  color: #ffffff;
}

.chart-box {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: #fff;
  padding: 20px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.chart-section {
  flex: 1;
  height: 100%;
  width: 100%;
  position: relative;
}

.chart-section canvas {
  position: absolute;
  left: 0;
  top: 0;
  width: 100% !important;
  height: 100% !important;
}

.video-preview {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.control-buttons {
  flex: none;
  padding: 12px 20px;
  background: #fff;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
  height: auto;
}

.info-container {
  flex: none;
  padding: 12px 20px;
  height: 35vh;
}

.info-row {
  height: 90%;
}

.info-card {
  height: 90%;
  background: #fff;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.info-card h3 {
  margin: 0 0 15px 0;
  color: #409eff;
}

.info-content {
  height: calc(100% - 60px);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
}

.count-info {
  font-size: 24px;
  font-weight: bold;
  color: #67c23a;
  vertical-align: middle;
}

.count-display {
  font-size: 48px;
  font-weight: bold;
  color: #67c23a;
  vertical-align: middle;
}

.hints-area {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  margin-top: 10px;
}

.hint-text {
  font-size: 32px;
  font-weight: bold;
  color: #e6a23c;
}

.record-details {
  text-align: left;
  width: 100%;
  height: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
}

.record-details img {
  max-width: 300%;
  max-height: 200px;
  object-fit: contain;
}

.record-details p {
  margin: 5px 0;
  color: #606266;
}

@media (max-width: 768px) {
  .camera-chart-container {
    height: 50vh;
  }

  .camera-box {
    width: 50%;
    aspect-ratio: 16 / 9;
  }
}
</style>
