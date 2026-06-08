from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import requests
import os
from pydub import AudioSegment  # 新加的格式转换工具

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

print("正在加载语音识别模型 (medium)...")
model = whisper.load_model("medium")

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio"}), 400
    
    audio_file = request.files['audio']
    # 1. 先存一个临时文件（浏览器原始格式）
    temp_raw_path = os.path.join(UPLOAD_FOLDER, "raw_voice.webm").replace('\\', '/')
    audio_file.save(temp_raw_path)

    # 2. 【核心修复】：把原始格式转为标准 WAV
    print("正在进行格式标准化转换...")
    try:
        # 读取原始录音（不管它是 webm 还是 ogg）
        audio = AudioSegment.from_file(temp_raw_path)
        # 强制转为标准 WAV (16000Hz, 单声道, 这是 SER 模型最喜欢的)
        standard_wav_path = os.path.join(UPLOAD_FOLDER, "current_voice.wav").replace('\\', '/')
        audio.set_frame_rate(16000).set_channels(1).export(standard_wav_path, format="wav")
        print("格式转换成功！")
    except Exception as e:
        print(f"格式转换失败: {e}")
        return jsonify({"error": "音频转换失败"}), 500

    # 3. Whisper 识别文字
    print("开始识别文字内容...")
    asr_result = model.transcribe(standard_wav_path, language="zh", fp16=False)
    recognized_text = asr_result['text'].strip() 

    if not recognized_text:
        return jsonify({"status": "success", "text": "（没听清哦）", "emotion": "neutral"})

    print(f"识别到: {recognized_text}。正在调动情感大脑...")

    # 4. 拿着标准 WAV 去问队友的模型
    try:
        ser_res = requests.post(
            "http://localhost:5000/predict", 
            json={"audio_path": standard_wav_path, "text": recognized_text},
            timeout=15
        )
        if ser_res.status_code == 200:
            detected_emotion = ser_res.json().get("emotion", "neutral")
            print(f"情感大脑结果：{detected_emotion}")
        else:
            print(f"情感大脑报 500 了，检查一下它的日志：{ser_res.text}")
            detected_emotion = "neutral"
    except Exception as e:
        detected_emotion = "neutral"

    return jsonify({
        "status": "success",
        "text": recognized_text,
        "emotion": detected_emotion
    })

if __name__ == '__main__':
    app.run(port=8000)