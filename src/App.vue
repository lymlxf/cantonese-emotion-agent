<template>
  <div class="page-wrapper">
    <div class="chat-container">
      <header><div class="status-dot"></div> 粤语智能 Agent (原声对比版)</header>
      
      <div class="chat-box" ref="chatBox">
        <div v-for="(msg, index) in messages" :key="index" :class="['message', msg.isAi ? 'ai' : 'user']">
          <div class="avatar">{{ msg.isAi ? '🤖' : '👤' }}</div>
          <div class="bubble">
            <!-- 情绪标签 -->
            <div v-if="!msg.isAi && msg.emotion" class="emotion-badge">
              🧠 识别情绪：{{ translateEmotion(msg.emotion) }}
            </div>

            <div class="msg-content">{{ msg.text }}</div>
            
            <div class="bubble-tools">
              <!-- 【核心修改】：这里的点击事件变了 -->
              <span @click.stop="handlePlayback(msg)" class="icon-btn">
                {{ msg.isPlaying ? '⏸️ 停止' : '🔊 播放原声' }}
              </span>
            </div>
          </div>
        </div>
        
        <div v-if="isProcessing" class="message ai">
          <div class="avatar">🤖</div>
          <div class="bubble thinking">{{ processingStatus }}</div>
        </div>
      </div>

      <div class="input-area">
        <button :class="['record-btn', isRecording ? 'recording' : '']" @mousedown="startRecording" @mouseup="stopRecording" :disabled="isProcessing">
          {{ isRecording ? '🛑 正在录音...' : '🎤 长按录制粤语' }}
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick } from 'vue'
import axios from 'axios'

// --- 【1. 请在这里填入你的 API Key】 ---
const MY_API_KEY = 

const messages = ref([{ isAi: true, text: '系统已就绪。长按录音，稍后点击“播放原声”可核对录音。' }])
const isProcessing = ref(false)
const processingStatus = ref('')
const isRecording = ref(false)
const chatBox = ref(null)

let mediaRecorder = null
let audioChunks = []

// --- 【核心修复：原生音频播放控制】 ---
// --- 【静音加强版】播放控制函数 ---
const handlePlayback = (msg) => {
  // 1. 【核心新增】：不管谁在说话，先让浏览器所有的 AI 合成音立刻闭嘴
  window.speechSynthesis.cancel(); 

  // 2. 如果是播放“用户原声”
  if (!msg.isAi && msg.audioUrl) {
    // 如果之前已经创建过播放器，先把它关了，防止重叠
    if (msg.audioInstance) {
      if (!msg.audioInstance.paused) {
        msg.audioInstance.pause();
        msg.isPlaying = false;
        return;
      }
      msg.audioInstance.currentTime = 0; // 从头播放
      msg.audioInstance.play();
      msg.isPlaying = true;
    } else {
      // 第一次点击，创建播放器
      const audio = new Audio(msg.audioUrl);
      msg.audioInstance = audio;
      audio.onended = () => { msg.isPlaying = false };
      audio.play();
      msg.isPlaying = true;
    }
  } 
  // 3. 如果是播放“AI 回复”
  else if (msg.isAi) {
    // 逻辑：因为第一步已经 cancel 了，所以这里直接重新开始念
    const utterance = new SpeechSynthesisUtterance(msg.text);
    utterance.lang = 'zh-HK';
    
    // 根据情绪调节语速（这里沿用之前的逻辑）
    if (msg.emotion === 'happy') { utterance.rate = 1.2; utterance.pitch = 1.2 }
    if (msg.emotion === 'sad') { utterance.rate = 0.8; utterance.pitch = 0.8 }

    utterance.onstart = () => { msg.isPlaying = true };
    utterance.onend = () => { msg.isPlaying = false };
    window.speechSynthesis.speak(utterance);
  }
}

const translateEmotion = (emo) => {
  const map = { 'angry': '愤怒 😡', 'happy': '开心 😊', 'neutral': '平静 😐', 'sad': '悲伤 😢' }
  return map[emo] || emo
}

const handleAudioProcessing = async (blob) => {
  const audioUrl = URL.createObjectURL(blob) // 这就是你的原声地址
  isProcessing.value = true
  processingStatus.value = '正在传输原声并分析...'
  
  try {
    const formData = new FormData()
    formData.append('audio', blob, 'test.wav') // 发送给后端的是真 wav
    
    const res = await axios.post('http://127.0.0.1:8000/upload-audio', formData)
    
    // 【关键】：把 audioUrl 存入消息，方便点击播放原声
    messages.value.push({ 
      isAi: false, 
      text: res.data.text, 
      audioUrl: audioUrl, 
      emotion: res.data.emotion,
      isPlaying: false 
    })
    await scrollToBottom()

    processingStatus.value = 'AI 正在组织回复...'
    const aiRes = await axios.post('https://open.bigmodel.cn/api/paas/v4/chat/completions', {
      model: "glm-4",
      messages: [
        { role: "system", content: `你是一个贴心的粤语助手。后端识别出用户情绪是【${res.data.emotion}】。请针对性回复。` },
        { role: "user", content: res.data.text }
      ]
    }, { headers: { 'Authorization': `Bearer ${MY_API_KEY}` } })
    
    messages.value.push({ isAi: true, text: aiRes.data.choices[0].message.content, isPlaying: false })
  } catch (e) {
    messages.value.push({ isAi: true, text: '合体失败，请检查后端。' })
  } finally {
    isProcessing.value = false; await scrollToBottom()
  }
}

const scrollToBottom = async () => {
  await nextTick(); if(chatBox.value) chatBox.value.scrollTop = chatBox.value.scrollHeight
}
const startRecording = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ 
  audio: {
    echoCancellation: true, // 开启回音消除（这就是解决回音的关键！）
    noiseSuppression: true, // 开启噪声抑制
    autoGainControl: true   // 自动增益控制（让音量更平稳）
  } 
})
    mediaRecorder = new MediaRecorder(stream); audioChunks = []
    mediaRecorder.ondataavailable = (event) => audioChunks.push(event.data)
    mediaRecorder.onstop = () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/wav' })
      handleAudioProcessing(audioBlob)
    }
    mediaRecorder.start(); isRecording.value = true
  } catch (err) { alert('麦克风权限被拒绝') }
}
const stopRecording = () => {
  if (mediaRecorder && isRecording.value) { mediaRecorder.stop(); isRecording.value = false }
}
</script>

<style>
/* 样式保持不变 */
html, body { height: 100%; margin: 0; padding: 0; overflow: hidden; background: #e0eafc; }
.page-wrapper { display: flex; justify-content: center; align-items: center; width: 100vw; height: 100vh; }
.chat-container { width: 400px; height: 85vh; background: white; border-radius: 40px; box-shadow: 0 20px 50px rgba(0,0,0,0.1); display: flex; flex-direction: column; overflow: hidden; border: 8px solid #222; }
header { background: #07c160; color: white; padding: 20px; text-align: center; font-weight: bold; }
.chat-box { flex: 1; padding: 20px; overflow-y: auto; background: #f7f7f7; }
.message { display: flex; margin-bottom: 20px; animation: slideIn 0.3s ease; }
.message.user { flex-direction: row-reverse; }
.avatar { font-size: 24px; margin: 0 10px; }
.bubble { padding: 12px 16px; border-radius: 18px; font-size: 14px; max-width: 70%; background: white; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
.message.user .bubble { background: #95ec69; }
.emotion-badge { font-size: 11px; background: rgba(0,0,0,0.05); padding: 2px 8px; border-radius: 10px; margin-bottom: 8px; display: inline-block; color: #666; }
.user .emotion-badge { background: rgba(255,255,255,0.3); color: #000; }
.bubble-tools { margin-top: 8px; display: flex; gap: 12px; border-top: 1px solid rgba(0,0,0,0.05); padding-top: 8px; }
.icon-btn { cursor: pointer; font-size: 12px; opacity: 0.7; color: #007aff; font-weight: bold; }
.input-area { padding: 25px; background: white; text-align: center; }
.record-btn { background: #07c160; color: white; border: none; padding: 15px; border-radius: 50px; width: 100%; font-weight: bold; cursor: pointer; }
.record-btn.recording { background: #ff4d4f; animation: pulse 1s infinite; }
@keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.05); } 100% { transform: scale(1); } }
@keyframes slideIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
</style>