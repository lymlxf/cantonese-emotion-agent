import pandas as pd
import librosa
import numpy as np
import io
import matplotlib.pyplot as plt


# ==========================================
# 超参数配置 (Hyperparameters)
# ==========================================
class AudioConfig:
    # --- 物理参数 ---
    target_sr = 22050
    duration = 8
    target_samples = target_sr * duration

    # --- 频谱图参数 ---
    n_mels = 128
    n_fft = 2048
    hop_length = 512

    # --- 标签参数 ---
    num_classes = 5  # 情感类别的总数

    # --- [增强] 基础参数 ---
    noise_factor = 0.005  # 白噪声振幅比例
    mixup_alpha = 0.2  # Mixup 的 Beta 分布参数

    # --- [增强] 注意力窗口 (SpecAugment) 参数 ---
    freq_mask_max = 15  # 最多遮住多少个 Mel 频段 (纵向高度)
    time_mask_max = 30  # 最多遮住多少个时间帧 (横向宽度)


def to_one_hot(label, num_classes):
    one_hot = np.zeros(num_classes, dtype=np.float32)
    one_hot[int(label)] = 1.0
    return one_hot


def apply_mixup(spec1, label1_oh, spec2, label2_oh, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    mixed_spec = lam * spec1 + (1 - lam) * spec2
    mixed_label = lam * label1_oh + (1 - lam) * label2_oh
    return mixed_spec, mixed_label, lam


def apply_spec_augment(spec, num_mask=2, freq_masking_max=15, time_masking_max=30):
    augmented_spec = spec.copy()
    n_mels, n_steps = augmented_spec.shape
    fill_value = augmented_spec.min()

    for i in range(num_mask):
        f_mask_size = np.random.randint(1, freq_masking_max)
        f0 = np.random.randint(0, n_mels - f_mask_size)
        augmented_spec[f0: f0 + f_mask_size, :] = fill_value

        t_mask_size = np.random.randint(1, time_masking_max)
        t0 = np.random.randint(0, n_steps - t_mask_size)
        augmented_spec[:, t0: t0 + t_mask_size] = fill_value

    return augmented_spec


def preprocess_audio_to_spec(audio_bytes, config, is_train=False):
    y, _ = librosa.load(io.BytesIO(audio_bytes), sr=config.target_sr)

    if len(y) < config.target_samples:
        pad_total = config.target_samples - len(y)
        if is_train:
            pad_left = np.random.randint(0, pad_total)
            pad_right = pad_total - pad_left
            y = np.pad(y, (pad_left, pad_right), mode='constant')
        else:
            y = np.pad(y, (0, pad_total), mode='constant')
    else:
        y = y[:config.target_samples]

    if is_train:
        noise = np.random.randn(len(y))
        y = y + config.noise_factor * noise

    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=config.target_sr, n_mels=config.n_mels,
        n_fft=config.n_fft, hop_length=config.hop_length
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    if is_train:
        mel_spec_db = apply_spec_augment(
            mel_spec_db,
            num_mask=2,
            freq_masking_max=config.freq_mask_max,
            time_masking_max=config.time_mask_max
        )

    return mel_spec_db


if __name__ == "__main__":
    file_path = r'D:\yue_emo_speech\data\train-00000-of-00045.parquet'
    df = pd.read_parquet(file_path)
    cfg = AudioConfig()

    unique_labels = sorted(df['label'].unique().tolist())
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}

    cfg.num_classes = len(unique_labels)

    print("=" * 30)
    print(f"✅ 自动发现 {cfg.num_classes} 种情绪类别！")
    print(f"类别映射表: {label_to_int}")
    print("=" * 30)

    row_A = df.iloc[0]
    row_B = df.iloc[1]

    spec_A = preprocess_audio_to_spec(row_A['audio_file']['bytes'], cfg, is_train=True)
    spec_B = preprocess_audio_to_spec(row_B['audio_file']['bytes'], cfg, is_train=True)

    int_label_A = label_to_int[row_A['label']]
    int_label_B = label_to_int[row_B['label']]

    label_A_oh = to_one_hot(int_label_A, cfg.num_classes)
    label_B_oh = to_one_hot(int_label_B, cfg.num_classes)

    mixed_spec, mixed_label, lam = apply_mixup(spec_A, label_A_oh, spec_B, label_B_oh, alpha=cfg.mixup_alpha)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mixed_spec, sr=cfg.target_sr, hop_length=cfg.hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Fully Augmented Spec (Shift + Noise + SpecAugment + Mixup)")
    plt.tight_layout()
    plt.show()