# SER Inference API — 接口文档

> 粤语语音情感识别推理接口，供后端服务调用。传入音频文件 + 转录文本，返回情感标签。

## 基本信息

| 项目 | 内容 |
|------|------|
| 服务地址 | `http://{host}:{port}`（默认 `http://localhost:5000`） |
| 协议 | HTTP/1.1 |
| 数据格式 | JSON |
| 编码 | UTF-8 |

---

## 启动服务

```bash
python ser_api.py --checkpoint checkpoints/best_model.pt --allowed_dir ./uploads
```

启动日志会输出服务地址和设备信息，如：

```
SER Inference API ready
  Endpoint: http://0.0.0.0:5000
  Device: cuda
  Modality: both
  Allowed dir: /path/to/uploads
```

**启动参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--checkpoint` | str | **必填** | 模型 checkpoint 文件路径 (`best_model.pt`) |
| `--allowed_dir` | str | `./uploads` | 允许读取音频的目录（防路径遍历） |
| `--host` | str | `0.0.0.0` | 服务绑定地址 |
| `--port` | int | `5000` | 服务端口 |
| `--max_file_size_mb` | int | `10` | 单个音频文件大小上限 |
| `--max_audio_sec` | int | `60` | 单个音频时长上限 |

---

## 端点

### 1. POST /predict — 情感推理

传入音频文件路径和 ASR 转录文本，返回识别到的情感标签。

**Request**

```http
POST /predict HTTP/1.1
Content-Type: application/json
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `audio_path` | string | ✅ | 音频文件路径，必须在 `allowed_dir` 目录内 |
| `text` | string | ✅ | ASR 转录的中文/粤语文本 |

**示例请求**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "audio_path": "uploads/user_20260529_001.mp3",
    "text": "我好開心今日見到你"
  }'
```

**成功响应**

```http
HTTP/1.1 200 OK
Content-Type: application/json
```

```json
{
  "emotion": "happy"
}
```

`emotion` 字段为以下 8 种情感标签之一：

| 标签 | 含义 |
|------|------|
| `angry` | 愤怒 |
| `disgusted` | 厌恶 |
| `fearful` | 恐惧 |
| `happy` | 开心 |
| `neutral` | 中性 |
| `other` | 其他 |
| `sad` | 悲伤 |
| `surprised` | 惊讶 |

**错误响应**

所有错误返回格式统一：

```json
{
  "error": "具体错误描述"
}
```

| HTTP 状态码 | 场景 | 示例 |
|------------|------|------|
| `400` | 参数缺失 | `{"error": "Missing required field: text"}` |
| `400` | 空文本 | `{"error": "Text input is empty"}` |
| `400` | 路径越界 | `{"error": "Audio path is outside the allowed directory..."}` |
| `400` | 文件过大 | `{"error": "Audio file too large: 15.2 MB (max 10 MB)"}` |
| `400` | JSON 格式错误 | `{"error": "Request body must be valid JSON"}` |
| `400` | 音频预处理失败 | `{"error": "Audio preprocessing failed: ..."}` |
| `400` | 文本预处理失败 | `{"error": "Text preprocessing failed: ..."}` |
| `404` | 音频文件不存在 | `{"error": "Audio file not found: uploads/xxx.mp3"}` |
| `500` | 模型推理异常 | `{"error": "Inference failed: ..."}` |

---

### 2. GET /health — 健康检查

检查服务是否正常运行。

**Request**

```http
GET /health HTTP/1.1
```

**示例请求**

```bash
curl http://localhost:5000/health
```

**响应**

```http
HTTP/1.1 200 OK
Content-Type: application/json
```

```json
{
  "status": "ok",
  "device": "cuda",
  "modality": "both"
}
```

| 字段 | 说明 |
|------|------|
| `status` | `"ok"` 表示服务正常 |
| `device` | 推理设备：`"cuda"` 或 `"cpu"` |
| `modality` | 模型模态：`"both"` / `"audio"` / `"text"` |

---

## 集成示例

### Python

```python
import requests

def get_emotion(audio_path: str, text: str) -> str:
    """调用 SER API 获取情感标签。"""
    resp = requests.post(
        "http://localhost:5000/predict",
        json={"audio_path": audio_path, "text": text},
    )
    resp.raise_for_status()
    return resp.json()["emotion"]

# 使用示例
emotion = get_emotion("uploads/voice.mp3", "我好開心今日見到你")
# → "happy"

# 用情感作为 LLM 提示词
llm_prompt = f"用户的当前情绪是 {emotion}，请据此调整回复语气。" + user_message
```

### JavaScript / Node.js

```javascript
async function getEmotion(audioPath, text) {
  const resp = await fetch("http://localhost:5000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ audio_path: audioPath, text }),
  });
  if (!resp.ok) {
    const err = await resp.json();
    throw new Error(err.error);
  }
  const data = await resp.json();
  return data.emotion;
}

// 使用示例
const emotion = await getEmotion("uploads/voice.mp3", "我好開心今日見到你");
// → "happy"
```

### curl

```bash
# 健康检查
curl http://localhost:5000/health

# 情感推理
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"audio_path": "uploads/voice.mp3", "text": "ASR转录文本"}'
```

---

## 注意事项

1. **音频格式**：通过 `soundfile` 解码，支持以下常见格式（完整列表 27 种）：
   - ✅ **MP3** (MPEG-1/2 Audio) — 推荐
   - ✅ **WAV** (Microsoft)
   - ✅ **FLAC** (Free Lossless Audio Codec)
   - ✅ **OGG** (OGG Container)
   - ✅ AIFF, CAF, AU 等
   - 内部自动转 22050Hz 单声道，采样率和声道不一致时自动处理
2. **路径安全**：`audio_path` 必须在 `--allowed_dir` 目录内，否则返回 400
3. **文本语言**：支持中文和粤语文本，内部使用 jieba 分词 + Tencent Word2Vec 编码
4. **并发**：开发模式为单线程串行处理，同一时刻只处理一个请求
5. **GPU 内存**：模型常驻 GPU 显存，启动时加载一次，后续请求直接推理
6. **依赖**：服务需安装 `flask`、`torch`、`torchaudio`、`soundfile`、`text2vec`、`jieba`

---

## 部署建议

```bash
# 1. 安装依赖
pip install flask torch torchaudio soundfile text2vec jieba numpy

# 2. 确保模型文件存在
ls checkpoints/best_model.pt

# 3. 创建音频目录
mkdir -p uploads

# 4. 启动服务
python ser_api.py --checkpoint checkpoints/best_model.pt --allowed_dir ./uploads
```

> 生产环境建议使用 `gunicorn`、`systemd` 或 Docker 部署，并添加反向代理（如 nginx）。
