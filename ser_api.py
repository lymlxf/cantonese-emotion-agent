#!/usr/bin/env python
"""
Flask HTTP inference API for Cantonese Speech Emotion Recognition (SER).

Accepts MP3 audio file path + ASR transcription text, runs inference with a
pre-trained MultimodalSER model, and returns the predicted emotion label.

Usage:
    python ser_api.py --checkpoint checkpoints/best_model.pt --allowed_dir ./uploads

Endpoints:
    POST /predict  - Infer emotion from audio+text
    GET  /health   - Model and device status
"""

import argparse
import logging
import os
import sys
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from flask import Flask, jsonify, request

# == Add project root to path for sibling imports ==============================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import MultimodalSER
from utils.audio_utils import AudioPreprocessor
from utils.text_utils import TextPreprocessor, CantoBertTextPreprocessor

# == Logging ===================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ser_api")

# == Emotion Labels ============================================================
LABEL_NAMES = {
    0: "angry",
    1: "disgusted",
    2: "fearful",
    3: "happy",
    4: "neutral",
    5: "other",
    6: "sad",
    7: "surprised",
}

# == Global singletons (populated at startup) ==================================
_model: MultimodalSER = None
_device: torch.device = None
_audio_preprocessor: AudioPreprocessor = None
_text_preprocessor = None  # TextPreprocessor or CantoBertTextPreprocessor
_text_encoder_type: str = "word2vec"


# =============================================================================
# Task 1: Model Loading & Device Detection
# =============================================================================

def detect_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("CUDA available: %s (%s)", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available, falling back to CPU")
    return device


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint and set to eval mode.

    Returns (model, config_dict).
    """
    logger.info("Loading checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ckpt["config"]
    modality = config.get("modality", "both")

    model = MultimodalSER(
        d_model=config.get("d_model", 256),
        num_heads=config.get("num_heads", 8),
        num_classes=config.get("num_classes", 8),
        dropout=config.get("dropout", 0.2),
        modality=modality,
        text_encoder_type=config.get("text_encoder_type", _text_encoder_type),
    ).to(device)

    # Load with strict=False: frozen BART weights are loaded from HuggingFace,
    # not stored in checkpoint (they never change during training).
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    logger.info("  Modality: %s", modality)
    logger.info("  Parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")
    logger.info("  Val acc (training): %.2f%%", ckpt.get("val_acc", float("nan")))
    logger.info("  Epoch: %s", ckpt.get("epoch", "N/A"))

    return model, config


# =============================================================================
# Task 2: Audio Preprocessing
# =============================================================================

def preprocess_audio(audio_path: str):
    """Read MP3 file and convert to Mel spectrogram tensor.

    Returns (mel_spec, length_tensor):
        mel_spec:  Tensor of shape (1, 80, time_steps)
        length_tensor:  Tensor of shape (1,) containing the number of time frames
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
    except OSError as e:
        raise OSError(f"Failed to read audio file: {audio_path}") from e

    preprocessor = AudioPreprocessor()  # eval mode by default, no augmentations
    mel_spec, length = preprocessor(audio_bytes)

    if length == 0 or mel_spec.numel() == 0:
        raise ValueError(f"Audio preprocessing produced empty output for: {audio_path}")

    # Add batch dimension: (80, T) → (1, 80, T)
    mel_spec = mel_spec.unsqueeze(0)
    length_tensor = torch.tensor([length])

    return mel_spec, length_tensor


# =============================================================================
# Task 3: Text Preprocessing
# =============================================================================

def preprocess_text(text: str):
    """Convert raw Chinese/Cantonese text to model-ready tensor.

    Word2Vec mode: returns (embeddings, length_tensor) where embeddings is (1, seq_len, 200) float.
    CantoBERT mode: returns (token_ids, length_tensor) where token_ids is (1, seq_len) long.
    """
    global _text_preprocessor

    if _text_preprocessor is None:
        if _text_encoder_type == "cantobert":
            logger.info("Loading CantoBERT TextPreprocessor...")
            _text_preprocessor = CantoBertTextPreprocessor()
        else:
            logger.info("Loading TextPreprocessor (Word2Vec model)...")
            _text_preprocessor = TextPreprocessor()

    if not text or not text.strip():
        raise ValueError("Text input is empty")

    data, length = _text_preprocessor(text)

    if length == 0 or data.shape[0] == 0:
        raise ValueError(f"Text preprocessing produced empty output for: {text!r}")

    if _text_encoder_type == "cantobert":
        # Token IDs: (seq_len,) int64 → (1, seq_len) long
        token_ids = torch.from_numpy(data).long().unsqueeze(0)
        length_tensor = torch.tensor([length]).clamp(min=1)
        return token_ids, length_tensor
    else:
        # Word2Vec embeddings: (seq_len, 200) float → (1, seq_len, 200) float
        embeddings = torch.from_numpy(data).float().unsqueeze(0)
        length_tensor = torch.tensor([length]).clamp(min=1)
        return embeddings, length_tensor


# =============================================================================
# Task 4: Flask Application
# =============================================================================

app = Flask(__name__)

# Runtime config (populated at startup via CLI args)
_allowed_dir: str = None
_max_file_size_mb: int = 10
_max_audio_sec: int = 60


def _validate_audio_path(audio_path: str) -> str:
    """Validate that audio_path is within the allowed directory and exists."""
    if not _allowed_dir:
        return audio_path  # no restriction configured

    abs_allowed = os.path.abspath(_allowed_dir)
    abs_path = os.path.abspath(audio_path)

    if os.path.commonpath([abs_allowed, abs_path]) != abs_allowed:
        raise ValueError(
            f"Audio path is outside the allowed directory. "
            f"Allowed: {abs_allowed}, Got: {abs_path}"
        )

    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    file_size_mb = os.path.getsize(abs_path) / (1024 * 1024)
    if file_size_mb > _max_file_size_mb:
        raise ValueError(
            f"Audio file too large: {file_size_mb:.1f} MB "
            f"(max {_max_file_size_mb} MB)"
        )

    return abs_path


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "device": str(_device),
        "modality": _model.modality if _model else "not_loaded",
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Infer emotion from audio file path and text.

    Expects JSON: {"audio_path": "...", "text": "..."}
    Returns JSON:  {"emotion": "happy"}
    """
    # 1. Parse request body ------------------------------------------------
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    audio_path = body.get("audio_path")
    text = body.get("text")

    if not audio_path:
        return jsonify({"error": "Missing required field: audio_path"}), 400
    if not text:
        return jsonify({"error": "Missing required field: text"}), 400

    # 2. Validate audio path ------------------------------------------------
    try:
        audio_path = _validate_audio_path(audio_path)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # 3. Validate text ------------------------------------------------------
    if not text.strip():
        return jsonify({"error": "Text input is empty"}), 400

    # 4. Preprocess ---------------------------------------------------------
    try:
        audio_tensor, audio_length = preprocess_audio(audio_path)
    except (FileNotFoundError, OSError, ValueError) as e:
        return jsonify({"error": f"Audio preprocessing failed: {e}"}), 400

    try:
        text_tensor, text_length = preprocess_text(text)
    except ValueError as e:
        return jsonify({"error": f"Text preprocessing failed: {e}"}), 400

    # 5. Run inference ------------------------------------------------------
    try:
        with torch.no_grad():
            audio_tensor = audio_tensor.to(_device)
            text_tensor = text_tensor.to(_device)
            audio_length = audio_length.to(_device)
            text_length = text_length.to(_device)

            if _model.modality == "both":
                logits = _model(audio_tensor, text_tensor, audio_length, text_length)
            elif _model.modality == "audio":
                logits = _model(audio_tensor, audio_lengths=audio_length)
            elif _model.modality == "text":
                logits = _model(text_input=text_tensor, text_lengths=text_length)
            else:
                return jsonify({"error": f"Unknown modality: {_model.modality}"}), 500

            pred = torch.argmax(logits, dim=1).item()
            emotion = LABEL_NAMES[pred]

    except Exception as e:
        logger.exception("Inference failed")
        return jsonify({"error": f"Inference failed: {e}"}), 500

    return jsonify({"emotion": emotion})


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Flask inference API for Cantonese SER model"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to best_model.pt checkpoint file",
    )
    parser.add_argument(
        "--allowed_dir", default="./uploads",
        help="Directory containing allowed audio files (default: ./uploads)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--text_encoder",
        default="word2vec",
        choices=["word2vec", "cantobert"],
        help="Text encoder backend (default: word2vec)",
    )
    parser.add_argument(
        "--port", type=int, default=5000,
        help="Port to bind (default: 5000)",
    )
    parser.add_argument(
        "--max_file_size_mb", type=int, default=10,
        help="Maximum MP3 file size in MB (default: 10)",
    )
    parser.add_argument(
        "--max_audio_sec", type=int, default=60,
        help="Maximum audio duration in seconds (default: 60)",
    )
    return parser.parse_args()


def main():
    global _model, _device, _audio_preprocessor
    global _allowed_dir, _max_file_size_mb, _max_audio_sec
    global _text_encoder_type

    args = parse_args()

    # Store runtime config
    _allowed_dir = args.allowed_dir
    _max_file_size_mb = args.max_file_size_mb
    _max_audio_sec = args.max_audio_sec
    _text_encoder_type = args.text_encoder

    # Ensure allowed directory exists
    os.makedirs(_allowed_dir, exist_ok=True)

    # Device detection
    _device = detect_device()

    # Load model (warm-load at startup)
    _model, config = load_model(args.checkpoint, _device)

    # Pre-load text preprocessor (Word2Vec) at startup
    logger.info("Pre-loading TextPreprocessor...")
    _ = preprocess_text("预热")  # triggers global singleton init
    logger.info("TextPreprocessor ready")

    logger.info("=" * 50)
    logger.info("SER Inference API ready")
    logger.info("  Endpoint: http://%s:%s", args.host, args.port)
    logger.info("  Device: %s", _device)
    logger.info("  Modality: %s", _model.modality)
    logger.info("  Allowed dir: %s", os.path.abspath(_allowed_dir))
    logger.info("=" * 50)

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
