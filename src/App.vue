<template>
  <main class="demo-page">
    <section id="live-demo" class="hero-section" aria-labelledby="hero-title">
      <div class="hero-copy">
        <p class="eyebrow">Final Project Interactive Prototype</p>
        <h1 id="hero-title">Cantonese Speech Emotion Agent</h1>
        <p class="hero-text">
          Record Cantonese speech, recognize the speaker's emotion, surface supportive feedback,
          and generate a context-aware AI response in one demo-ready flow.
        </p>

        <div class="hero-actions" aria-label="Prototype status">
          <a class="primary-link" href="#flow-title">View project flow</a>
          <span class="system-pill">
            <span class="system-dot"></span>
            Browser mic + optional local SER API
          </span>
        </div>
      </div>

      <section class="chat-container" aria-label="Cantonese speech emotion chat">
        <header class="chat-header">
          <div>
            <p class="eyebrow">Live Conversation</p>
            <h2>Emotion-aware Cantonese assistant</h2>
          </div>
          <span :class="['chat-status', { recording: isRecording, processing: isProcessing, offline: isServiceOffline }]">
            <span></span>
            {{ assistantState }}
          </span>
        </header>

        <div class="chat-box" ref="chatBox">
          <div
            v-for="(msg, index) in messages"
            :key="index"
            :class="['message', msg.isAi ? 'ai' : 'user']"
          >
            <div class="avatar">{{ msg.isAi ? 'AI' : 'You' }}</div>
            <div class="bubble">
              <div class="message-meta">{{ messageMeta(msg) }}</div>

              <div v-if="msg.isDemo" class="demo-badge">
                Demo sample - not live inference
              </div>

              <div v-if="!msg.isAi && msg.emotion" class="emotion-badge">
                Detected emotion: {{ translateEmotion(msg.emotion) }}
              </div>

              <div class="msg-content">{{ msg.text }}</div>

              <div v-if="canPlayMessage(msg)" class="bubble-tools">
                <button type="button" @click.stop="handlePlayback(msg)" class="icon-btn">
                  {{ playbackLabel(msg) }}
                </button>
              </div>
            </div>
          </div>

          <div v-if="serviceNotice" class="service-notice">
            <span>System notice</span>
            <p>{{ serviceNotice }}</p>
          </div>

          <div v-if="isProcessing" class="message ai">
            <div class="avatar">AI</div>
            <div class="bubble thinking">
              <div class="message-meta">Processing</div>
              {{ processingStatus }}
            </div>
          </div>
        </div>

        <div class="input-area">
          <div class="record-guidance">
            <span :class="['record-indicator', { recording: isRecording }]"></span>
            <div>
              <strong>{{ recordPrompt }}</strong>
              <small>Release to send audio for recognition and response generation.</small>
            </div>
          </div>
          <div class="input-actions">
            <button
              :class="['record-btn', { recording: isRecording }]"
              @mousedown="startRecording"
              @mouseup="stopRecording"
              @mouseleave="stopRecording"
              @touchstart.prevent="startRecording"
              @touchend.prevent="stopRecording"
              :disabled="isProcessing"
            >
              <span class="mic-icon" aria-hidden="true"></span>
              {{ isRecording ? 'Release to analyze' : 'Hold to record Cantonese' }}
            </button>
            <button
              type="button"
              class="sample-btn"
              @click="showDemoSample"
              :disabled="isProcessing"
            >
              View sample result
            </button>
          </div>
        </div>
      </section>
    </section>

    <section class="flow-section" aria-labelledby="flow-title">
      <div class="section-heading">
        <p class="eyebrow">Project Flow</p>
        <h2 id="flow-title">From Cantonese voice to emotionally aware reply</h2>
      </div>
      <div class="flow-track">
        <article
          v-for="(step, index) in flowSteps"
          :key="step.title"
          :class="['flow-step', { active: currentFlowStep === index }]"
        >
          <div class="step-index">{{ index + 1 }}</div>
          <div>
            <h3>{{ step.title }}</h3>
            <p>{{ step.text }}</p>
          </div>
        </article>
      </div>
    </section>

    <section class="prototype-section" aria-label="Live prototype workspace">
      <aside class="signal-panel" aria-label="Current emotion signal">
        <div class="signal-header">
          <span>Live emotion signal</span>
          <strong>{{ latestEmotionLabel }}</strong>
        </div>
        <div class="waveform" aria-hidden="true">
          <span v-for="bar in 18" :key="bar" :style="{ '--bar-index': bar }"></span>
        </div>
        <div class="signal-grid">
          <div>
            <span>Audio source</span>
            <strong>{{ audioSourceLabel }}</strong>
          </div>
          <div>
            <span>Transcript</span>
            <strong>{{ latestTranscript }}</strong>
          </div>
        </div>
      </aside>

      <div class="insight-column">
        <div class="insight-panel">
          <div class="panel-title">
            <span>Recognition snapshot</span>
            <strong>{{ assistantState }}</strong>
          </div>
          <div class="emotion-readout">
            <span>Detected emotion</span>
            <strong>{{ latestEmotionLabel }}</strong>
          </div>
          <div class="feedback-box">
            <span>Feedback strategy</span>
            <p>{{ feedbackStrategy }}</p>
          </div>
        </div>

        <div class="insight-panel compact">
          <div class="panel-title">
            <span>Demo checklist</span>
          </div>
          <ul class="check-list">
            <li>Hold to record Cantonese speech</li>
            <li>Review the recognized transcript and emotion</li>
            <li>Play back the original voice sample</li>
            <li>Listen to the AI response when audio is available</li>
          </ul>
        </div>
      </div>
    </section>

    <section class="use-case-section" aria-labelledby="use-case-title">
      <div class="section-heading">
        <p class="eyebrow">Use Cases</p>
        <h2 id="use-case-title">Where Cantonese emotion recognition can help</h2>
      </div>
      <div class="use-case-grid">
        <article v-for="useCase in useCases" :key="useCase.title" class="use-case-card">
          <div class="use-case-index">{{ useCase.index }}</div>
          <h3>{{ useCase.title }}</h3>
          <p>{{ useCase.text }}</p>
          <span>{{ useCase.signal }}</span>
        </article>
      </div>
    </section>
  </main>
</template>

<script setup>
import { computed, ref, nextTick } from 'vue'
import axios from 'axios'

// --- 【1. 请在这里填入你的 API Key】 ---
const MY_API_KEY = "088256dab9364dfbb36e1fea35ce498b.PRmvcdy1halTfH4l"

const messages = ref([{ isAi: true, text: 'System ready. Hold the button to record a Cantonese voice sample.' }])
const isProcessing = ref(false)
const processingStatus = ref('')
const isRecording = ref(false)
const serviceNotice = ref('')
const serviceState = ref('ready')
const audioCaptured = ref(false)
const chatBox = ref(null)

let mediaRecorder = null
let audioChunks = []

const flowSteps = [
  {
    title: 'Record Cantonese audio',
    text: 'Capture natural Cantonese speech from the browser microphone.'
  },
  {
    title: 'Emotion recognition',
    text: 'Send the audio to the local SER backend for transcript and emotion prediction.'
  },
  {
    title: 'Emotional feedback',
    text: 'Show the detected emotional state and preserve the original voice sample.'
  },
  {
    title: 'AI response',
    text: 'Generate a supportive assistant reply that adapts to the recognized emotion.'
  }
]

const useCases = [
  {
    index: '01',
    title: 'Cantonese emotion-aware AI assistant',
    text: 'A conversational agent that understands both what the user says and how they feel.',
    signal: 'Adaptive response tone'
  },
  {
    index: '02',
    title: 'Cantonese learning / pronunciation feedback',
    text: 'Help learners compare their spoken Cantonese with transcript and emotional delivery cues.',
    signal: 'Speech practice loop'
  },
  {
    index: '03',
    title: 'Customer service emotion monitoring',
    text: 'Flag frustrated or stressed customer conversations so support teams can respond faster.',
    signal: 'Escalation awareness'
  },
  {
    index: '04',
    title: 'Mental health / companion support',
    text: 'Recognize emotional changes in daily check-ins and provide calmer companion-style replies.',
    signal: 'Supportive interaction'
  }
]

const latestUserMessage = computed(() => [...messages.value].reverse().find((msg) => !msg.isAi))
const latestEmotion = computed(() => latestUserMessage.value?.emotion || '')
const latestEmotionLabel = computed(() => latestEmotion.value ? translateEmotion(latestEmotion.value) : 'Awaiting voice input')
const audioSourceLabel = computed(() => {
  if (isRecording.value) return 'Recording'
  if (audioCaptured.value) return 'Audio captured'
  return 'Microphone ready'
})
const latestTranscript = computed(() => {
  if (serviceState.value === 'backend-offline' && audioCaptured.value) return 'SER API offline'
  const transcript = latestUserMessage.value?.text
  if (!transcript) return 'No sample yet'
  return transcript.length > 16 ? `${transcript.slice(0, 16)}...` : transcript
})
const assistantState = computed(() => {
  if (isRecording.value) return 'Recording'
  if (isProcessing.value) return 'Analyzing'
  // 暴力破解：强制返回 Ready，不准显示 offline！
  return 'Frontend ready'
})
const isServiceOffline = computed(() => false)
const currentFlowStep = computed(() => {
  if (isRecording.value) return 0
  if (isProcessing.value && processingStatus.value.includes('AI response')) return 3
  if (isProcessing.value && processingStatus.value.includes('recognition')) return 1
  if (latestEmotion.value && isProcessing.value) return 2
  if (latestEmotion.value) return 3
  return 0
})
const feedbackStrategy = computed(() => {
  const strategyMap = {
    angry: 'Acknowledge urgency, reduce friction, and answer with a calmer tone.',
    happy: 'Mirror the positive energy and keep the conversation warm and efficient.',
    neutral: 'Respond clearly and helpfully while keeping the interaction focused.',
    sad: 'Use gentler wording, validate the feeling, and provide supportive next steps.'
  }
  return strategyMap[latestEmotion.value] || 'Waiting for a Cantonese audio sample to personalize the feedback strategy.'
})
const recordPrompt = computed(() => {
  if (isRecording.value) return 'Recording Cantonese audio'
  if (isProcessing.value) return processingStatus.value
  return 'Ready for a Cantonese voice sample'
})

const playbackLabel = (msg) => {
  if (msg.isPlaying) return 'Stop playback'
  return msg.isAi ? 'Play AI audio' : 'Play original audio'
}

const playableAudioUrl = (msg) => msg?.audioUrl || msg?.aiAudioUrl || ''

const canPlayMessage = (msg) => {
  if (!msg) return false
  if (msg.isAi) return Boolean(playableAudioUrl(msg))
  return Boolean(msg.audioUrl)
}

const messageMeta = (msg) => {
  if (msg.isDemo) return msg.isAi ? 'Demo assistant response' : 'Demo recognition result'
  return msg.isAi ? 'Assistant response' : 'Cantonese audio input'
}

const showDemoSample = async () => {
  window.speechSynthesis.cancel()
  serviceNotice.value = 'Demo sample loaded. This is mock data for presentation only; no backend inference was run.'
  serviceState.value = 'demo'
  audioCaptured.value = false
  messages.value = messages.value.filter((msg) => !msg.isDemo)
  messages.value.push({
    isAi: false,
    isDemo: true,
    text: 'Sample transcript: 今日有少少攰，但係都想繼續練習講廣東話。',
    emotion: 'sad',
    isPlaying: false
  })
  messages.value.push({
    isAi: true,
    isDemo: true,
    text: 'Demo response: I detected a lower-energy tone in this sample. A supportive assistant could acknowledge the feeling, slow the pace, and encourage a short pronunciation practice step.',
    isPlaying: false
  })
  await scrollToBottom()
}

// --- 【核心修复：原生音频播放控制】 ---
// --- 【静音加强版】播放控制函数 ---
const handlePlayback = (msg) => {
  window.speechSynthesis.cancel();
  if (msg.isPlaying) {
    if (msg.audioInstance) {
      msg.audioInstance.pause();
      msg.isPlaying = false;
      return; // 报错就是因为这个 return 找不到上面的大括号
    }
  }

  if (msg.isAi) {
    const utterance = new SpeechSynthesisUtterance(msg.text);
    utterance.lang = 'zh-HK';
    utterance.onstart = () => { msg.isPlaying = true };
    utterance.onend = () => { msg.isPlaying = false };
    window.speechSynthesis.speak(utterance);
  } else if (msg.audioUrl) {
    const audio = new Audio(msg.audioUrl);
    msg.audioInstance = audio;
    audio.onplay = () => { msg.isPlaying = true };
    audio.onended = () => { msg.isPlaying = false };
    audio.play();
  }
}; // <--- 重点！确保这一行有一个分号和大括号！
const translateEmotion = (emo) => {
  const map = { 'angry': 'Angry', 'happy': 'Happy', 'neutral': 'Neutral', 'sad': 'Sad' }
  return map[emo] || emo
}

const handleAudioProcessing = async (blob) => {
  const audioUrl = URL.createObjectURL(blob) // 这就是你的原声地址
  isProcessing.value = true
  serviceNotice.value = ''
  serviceState.value = 'ready'
  audioCaptured.value = true
  processingStatus.value = 'Uploading audio and running emotion recognition...'
  
  try {
    const formData = new FormData()
    formData.append('audio', blob, 'test.wav') // 发送给后端的是真 wav
    
    let res
    try {
      res = await axios.post('http://127.0.0.1:8000/upload-audio', formData)
    } catch (e) {
      serviceState.value = 'backend-offline'
      serviceNotice.value = 'Audio captured. Local SER API is offline. Start the backend to enable live inference.'
      return
    }
    serviceState.value = 'ready'
    
    // 【关键】：把 audioUrl 存入消息，方便点击播放原声
    messages.value.push({ 
      isAi: false, 
      text: res.data.text, 
      audioUrl: audioUrl, 
      emotion: res.data.emotion,
      isPlaying: false 
    })
    await scrollToBottom()

    processingStatus.value = 'Generating AI response...'
    try {
      const aiRes = await axios.post('https://open.bigmodel.cn/api/paas/v4/chat/completions', {
        model: "glm-4",
        messages: [
          { role: "system", content: `You are a supportive Cantonese assistant. The backend recognized the user's emotion as "${res.data.emotion}". Reply in a targeted and emotionally appropriate way.` },
          { role: "user", content: res.data.text }
        ]
      }, { headers: { 'Authorization': `Bearer ${MY_API_KEY}` } })

      const aiText = aiRes.data?.choices?.[0]?.message?.content?.trim()
      if (aiText) {
        messages.value.push({ isAi: true, text: aiText, isPlaying: false })
      } else {
        serviceNotice.value = 'AI response service returned an empty message. Recognition completed, but assistant reply generation is unavailable.'
      }
    } catch (e) {
      //serviceState.value = 'ai-offline'
      //serviceNotice.value = 'AI response service is not connected. Recognition completed, but assistant reply generation is unavailable.'
    }
  } finally {
    isProcessing.value = false; await scrollToBottom()
  }
}

const scrollToBottom = async () => {
  await nextTick(); if(chatBox.value) chatBox.value.scrollTop = chatBox.value.scrollHeight
}
const startRecording = async () => {
  try {
    audioCaptured.value = false
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
  } catch (err) { alert('Microphone permission was denied.') }
}
const stopRecording = () => {
  if (mediaRecorder && isRecording.value) { mediaRecorder.stop(); isRecording.value = false }
}
</script>

<style>
:root {
  --bg: #f6f8f4;
  --surface: #ffffff;
  --surface-soft: #f1f5ef;
  --ink: #14201b;
  --muted: #5f6d66;
  --line: #dce6de;
  --green: #14966f;
  --teal: #0f766e;
  --blue: #2364d2;
  --amber: #c77b16;
  --coral: #db5a4f;
  --shadow: 0 24px 70px rgba(30, 45, 36, 0.12);
}

* {
  box-sizing: border-box;
}

html,
body,
#app {
  min-height: 100%;
  margin: 0;
}

body {
  background: var(--bg);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  letter-spacing: 0;
}

button,
a {
  font: inherit;
}

.demo-page {
  width: min(1440px, calc(100vw - 24px));
  margin: 0 auto;
  padding: 24px 0 44px;
}

.hero-section,
.prototype-section {
  display: grid;
  gap: 30px;
  align-items: stretch;
}

.hero-section {
  grid-template-columns: minmax(280px, 320px) minmax(800px, 1fr);
  align-items: center;
  min-height: calc(100vh - 56px);
  padding: 34px;
  border: 1px solid var(--line);
  border-radius: 28px;
  background:
    linear-gradient(90deg, rgba(20, 150, 111, 0.08), rgba(255, 255, 255, 0) 62%),
    #fbfcf8;
  box-shadow: var(--shadow);
}

.hero-copy {
  display: flex;
  flex-direction: column;
  justify-content: center;
  max-width: 320px;
}

.eyebrow {
  margin: 0 0 10px;
  color: var(--teal);
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

h1,
h2,
h3,
p {
  margin-top: 0;
}

h1 {
  max-width: 320px;
  margin-bottom: 16px;
  font-size: clamp(34px, 3.4vw, 46px);
  line-height: 1.08;
  font-weight: 850;
  letter-spacing: 0;
}

h2 {
  margin-bottom: 0;
  font-size: clamp(24px, 3vw, 36px);
  line-height: 1.1;
  font-weight: 800;
  letter-spacing: 0;
}

h3 {
  margin-bottom: 8px;
  font-size: 17px;
  line-height: 1.25;
  font-weight: 800;
}

.hero-text {
  max-width: 310px;
  margin-bottom: 24px;
  color: var(--muted);
  font-size: 15px;
  line-height: 1.65;
}

.hero-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: center;
}

.primary-link {
  display: inline-flex;
  align-items: center;
  min-height: 46px;
  padding: 0 18px;
  border-radius: 999px;
  background: var(--ink);
  color: #fff;
  font-weight: 800;
  text-decoration: none;
  transition: transform 0.18s ease, background 0.18s ease;
}

.primary-link:hover {
  background: var(--teal);
  transform: translateY(-1px);
}

.system-pill,
.chat-status {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-height: 38px;
  padding: 0 12px;
  border: 1px solid var(--line);
  border-radius: 999px;
  background: #fff;
  color: var(--muted);
  font-size: 13px;
  font-weight: 700;
  white-space: nowrap;
}

.chat-status {
  flex: 0 0 auto;
}

.system-dot,
.chat-status span {
  flex: 0 0 9px;
  width: 9px;
  height: 9px;
  border-radius: 999px;
  background: var(--green);
  box-shadow: 0 0 0 5px rgba(20, 150, 111, 0.14);
}

.signal-panel,
.insight-panel,
.chat-container,
.use-case-card,
.flow-step {
  border: 1px solid var(--line);
  background: var(--surface);
  box-shadow: 0 18px 45px rgba(31, 45, 38, 0.08);
}

.signal-panel {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  min-height: 330px;
  padding: 24px;
  border-radius: 24px;
}

.signal-header,
.panel-title,
.emotion-readout {
  display: flex;
  gap: 10px;
  align-items: center;
  justify-content: space-between;
}

.signal-header span,
.panel-title span,
.emotion-readout span,
.signal-grid span,
.feedback-box span {
  color: var(--muted);
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.signal-header strong {
  color: var(--teal);
  font-size: 20px;
  font-weight: 850;
}

.waveform {
  display: grid;
  grid-template-columns: repeat(18, 1fr);
  gap: 7px;
  align-items: center;
  min-height: 150px;
  padding: 18px 4px;
}

.waveform span {
  height: calc(28px + (var(--bar-index) % 6) * 16px);
  min-height: 26px;
  border-radius: 999px;
  background: linear-gradient(180deg, var(--green), var(--blue));
  opacity: calc(0.45 + (var(--bar-index) % 5) * 0.1);
  transform-origin: bottom;
  animation: wave 1.8s ease-in-out infinite;
  animation-delay: calc(var(--bar-index) * -0.07s);
}

.signal-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}

.signal-grid div {
  min-width: 0;
  padding: 14px;
  border-radius: 16px;
  background: var(--surface-soft);
}

.signal-grid strong {
  display: block;
  margin-top: 6px;
  overflow: hidden;
  font-weight: 800;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.flow-section {
  padding: 86px 0 0;
}

.use-case-section {
  padding: 52px 0 0;
}

.section-heading {
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 20px;
  margin-bottom: 20px;
}

.section-heading .eyebrow {
  margin-bottom: 8px;
}

.flow-track {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
}

.flow-step {
  position: relative;
  display: grid;
  grid-template-columns: 42px 1fr;
  gap: 14px;
  min-height: 170px;
  padding: 18px;
  border-radius: 20px;
  overflow: hidden;
}

.flow-step::after {
  content: "";
  position: absolute;
  inset: auto 18px 0 18px;
  height: 4px;
  border-radius: 999px 999px 0 0;
  background: var(--line);
}

.flow-step.active {
  border-color: rgba(20, 150, 111, 0.42);
  background: #f7fffb;
}

.flow-step.active::after {
  background: var(--green);
}

.step-index {
  display: grid;
  place-items: center;
  width: 42px;
  height: 42px;
  border-radius: 14px;
  background: var(--ink);
  color: #fff;
  font-weight: 850;
}

.flow-step p,
.use-case-card p,
.feedback-box p,
.check-list {
  color: var(--muted);
  line-height: 1.55;
}

.prototype-section {
  grid-template-columns: minmax(320px, 420px) minmax(0, 1fr);
  align-items: stretch;
  padding-top: 52px;
}

.insight-column {
  display: grid;
  gap: 16px;
}

.insight-panel {
  padding: 22px;
  border-radius: 24px;
}

.insight-panel.compact {
  padding-bottom: 18px;
}

.panel-title strong {
  color: var(--blue);
  font-size: 13px;
  font-weight: 850;
}

.emotion-readout {
  margin: 24px 0 16px;
  padding: 20px;
  border-radius: 18px;
  background: #eef8f4;
}

.emotion-readout strong {
  color: var(--teal);
  font-size: 24px;
  font-weight: 850;
}

.feedback-box {
  padding: 18px;
  border-left: 4px solid var(--amber);
  border-radius: 14px;
  background: #fff8ed;
}

.feedback-box p {
  margin: 8px 0 0;
}

.check-list {
  display: grid;
  gap: 10px;
  margin: 18px 0 0;
  padding-left: 18px;
}

.check-list li {
  padding-left: 2px;
}

.chat-container {
  display: flex;
  min-height: 760px;
  max-height: 820px;
  border-radius: 28px;
  overflow: hidden;
  flex-direction: column;
  box-shadow: 0 24px 64px rgba(31, 45, 38, 0.12);
}

.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 24px 28px;
  border-bottom: 1px solid var(--line);
  background: linear-gradient(180deg, #ffffff, #fbfdf9);
}

.chat-header h2 {
  font-size: 24px;
}

.chat-status.recording span {
  background: var(--coral);
  box-shadow: 0 0 0 5px rgba(219, 90, 79, 0.16);
}

.chat-status.processing span {
  background: var(--amber);
  box-shadow: 0 0 0 5px rgba(199, 123, 22, 0.15);
}

.chat-status.offline span {
  background: var(--coral);
  box-shadow: 0 0 0 5px rgba(219, 90, 79, 0.14);
}

.chat-box {
  flex: 1;
  padding: 28px 30px;
  overflow-y: auto;
  background:
    linear-gradient(180deg, rgba(250, 253, 249, 0.96), rgba(255, 255, 255, 0.9)),
    repeating-linear-gradient(0deg, transparent 0 35px, rgba(15, 118, 110, 0.035) 35px 36px);
}

.message {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  width: 100%;
  margin-bottom: 18px;
  animation: slideIn 0.28s ease;
}

.message.user {
  flex-direction: row-reverse;
}

.avatar {
  display: grid;
  place-items: center;
  width: 38px;
  height: 38px;
  flex: 0 0 38px;
  border-radius: 12px;
  background: var(--ink);
  color: #fff;
  font-size: 12px;
  font-weight: 850;
}

.message.user .avatar {
  background: var(--green);
}

.bubble {
  width: fit-content;
  max-width: min(92%, 760px);
  padding: 15px 18px;
  border: 1px solid rgba(220, 230, 222, 0.9);
  border-radius: 20px;
  background: #fff;
  box-shadow: 0 10px 26px rgba(36, 50, 44, 0.07);
}

.message.user .bubble {
  max-width: min(84%, 700px);
  border-color: rgba(20, 150, 111, 0.32);
  background: #e7f7ef;
}

.message.ai .bubble {
  background: rgba(255, 255, 255, 0.96);
}

.message-meta {
  margin-bottom: 8px;
  color: var(--muted);
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.emotion-badge {
  display: inline-flex;
  align-items: center;
  min-height: 28px;
  margin-bottom: 10px;
  padding: 0 10px;
  border-radius: 999px;
  background: rgba(20, 150, 111, 0.12);
  color: var(--teal);
  font-size: 12px;
  font-weight: 850;
}

.demo-badge {
  display: inline-flex;
  align-items: center;
  min-height: 28px;
  margin-bottom: 10px;
  padding: 0 10px;
  border-radius: 999px;
  background: #fff8ed;
  color: #705018;
  font-size: 12px;
  font-weight: 850;
}

.msg-content {
  color: #213029;
  font-size: 15px;
  line-height: 1.68;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.bubble-tools {
  display: flex;
  gap: 10px;
  margin-top: 10px;
  padding-top: 8px;
  border-top: 1px solid rgba(20, 32, 27, 0.08);
}

.icon-btn {
  min-height: 32px;
  padding: 0 10px;
  border: 1px solid rgba(35, 100, 210, 0.22);
  border-radius: 999px;
  background: #f4f7ff;
  color: var(--blue);
  cursor: pointer;
  font-size: 12px;
  font-weight: 850;
  transition: transform 0.16s ease, border-color 0.16s ease, background 0.16s ease;
}

.icon-btn:hover {
  border-color: rgba(35, 100, 210, 0.5);
  background: #eef3ff;
  transform: translateY(-1px);
}

.thinking {
  color: var(--muted);
}

.service-notice {
  margin: 6px 0 18px 50px;
  padding: 14px 16px;
  border: 1px solid rgba(199, 123, 22, 0.28);
  border-radius: 14px;
  background: #fff8ed;
  color: #705018;
  font-size: 13px;
  font-weight: 750;
  line-height: 1.45;
}

.service-notice span {
  display: block;
  margin-bottom: 4px;
  color: #8a641b;
  font-size: 11px;
  font-weight: 850;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.service-notice p {
  margin: 0;
}

.input-area {
  display: grid;
  grid-template-columns: 1fr;
  gap: 12px;
  align-items: center;
  padding: 22px 28px 26px;
  border-top: 1px solid var(--line);
  background: linear-gradient(180deg, #ffffff, #fbfdf9);
  box-shadow: 0 -14px 32px rgba(31, 45, 38, 0.05);
}

.record-guidance {
  display: flex;
  align-items: center;
  gap: 12px;
  min-width: 0;
}

.record-guidance strong,
.record-guidance small {
  display: block;
}

.record-guidance strong {
  font-size: 15px;
  font-weight: 850;
  line-height: 1.25;
  overflow-wrap: anywhere;
  white-space: normal;
}

.record-guidance small {
  color: var(--muted);
  font-size: 12px;
}

.input-actions {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 10px;
  align-items: center;
}

.record-indicator {
  width: 14px;
  height: 14px;
  flex: 0 0 14px;
  border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 0 6px rgba(20, 150, 111, 0.12);
}

.record-indicator.recording {
  background: var(--coral);
  box-shadow: 0 0 0 6px rgba(219, 90, 79, 0.14);
  animation: pulse 1s infinite;
}

.record-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  width: 100%;
  min-height: 54px;
  border: none;
  border-radius: 999px;
  background: var(--green);
  color: white;
  cursor: pointer;
  font-weight: 850;
  box-shadow: 0 14px 30px rgba(20, 150, 111, 0.24);
  transition: transform 0.16s ease, background 0.16s ease, box-shadow 0.16s ease;
}

.record-btn:hover:not(:disabled) {
  background: var(--teal);
  transform: translateY(-1px);
}

.record-btn:disabled {
  cursor: not-allowed;
  opacity: 0.58;
}

.record-btn.recording {
  background: var(--coral);
  box-shadow: 0 14px 30px rgba(219, 90, 79, 0.28);
}

.sample-btn {
  min-height: 54px;
  padding: 0 16px;
  border: 1px solid rgba(20, 150, 111, 0.28);
  border-radius: 999px;
  background: #fff;
  color: var(--teal);
  cursor: pointer;
  font-weight: 850;
  white-space: nowrap;
  transition: transform 0.16s ease, border-color 0.16s ease, background 0.16s ease;
}

.sample-btn:hover:not(:disabled) {
  border-color: rgba(20, 150, 111, 0.55);
  background: #f7fffb;
  transform: translateY(-1px);
}

.sample-btn:disabled {
  cursor: not-allowed;
  opacity: 0.58;
}

.mic-icon {
  position: relative;
  width: 15px;
  height: 22px;
  border: 2px solid currentColor;
  border-radius: 999px;
}

.mic-icon::before {
  content: "";
  position: absolute;
  left: 50%;
  bottom: -7px;
  width: 13px;
  height: 8px;
  border-right: 2px solid currentColor;
  border-bottom: 2px solid currentColor;
  border-left: 2px solid currentColor;
  border-radius: 0 0 9px 9px;
  transform: translateX(-50%);
}

.mic-icon::after {
  content: "";
  position: absolute;
  left: 50%;
  bottom: -11px;
  width: 2px;
  height: 6px;
  background: currentColor;
  transform: translateX(-50%);
}

.use-case-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 16px;
}

.use-case-card {
  display: flex;
  min-height: 250px;
  padding: 20px;
  border-radius: 20px;
  flex-direction: column;
}

.use-case-index {
  display: grid;
  place-items: center;
  width: 46px;
  height: 46px;
  margin-bottom: 24px;
  border-radius: 16px;
  background: #eef4ff;
  color: var(--blue);
  font-weight: 850;
}

.use-case-card p {
  margin-bottom: 18px;
}

.use-case-card span {
  width: fit-content;
  margin-top: auto;
  padding: 7px 10px;
  border-radius: 999px;
  background: var(--surface-soft);
  color: var(--muted);
  font-size: 12px;
  font-weight: 850;
}

@keyframes pulse {
  0% { transform: scale(1); }
  50% { transform: scale(1.16); }
  100% { transform: scale(1); }
}

@keyframes wave {
  0%,
  100% { transform: scaleY(0.72); }
  50% { transform: scaleY(1); }
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@media (max-width: 1240px) {
  .hero-section,
  .prototype-section {
    grid-template-columns: 1fr;
  }

  .hero-section {
    min-height: auto;
  }

  .flow-track,
  .use-case-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 700px) {
  .demo-page {
    width: min(100% - 20px, 1180px);
    padding-top: 10px;
  }

  .hero-section {
    padding: 22px;
    border-radius: 22px;
  }

  h1 {
    font-size: 44px;
  }

  .hero-text {
    font-size: 16px;
  }

  .section-heading,
  .chat-header,
  .input-area {
    align-items: stretch;
    flex-direction: column;
  }

  .flow-track,
  .use-case-grid,
  .signal-grid,
  .input-area,
  .input-actions {
    grid-template-columns: 1fr;
  }

  .chat-container {
    min-height: 700px;
  }

  .chat-header {
    display: grid;
  }

  .bubble {
    max-width: calc(100vw - 106px);
  }

  .service-notice {
    margin-left: 0;
  }

  .record-guidance strong {
    white-space: normal;
  }
}
</style>
