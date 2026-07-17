import pandas as pd
from io import BytesIO
import soundfile as sf

df = pd.read_parquet("train-00000-of-00045.parquet")
for i in range(5):
    audio_bytes = df['audio_file'].iloc[i]['bytes']

    with BytesIO(audio_bytes) as f:
        data, samplerate = sf.read(f)

    sf.write(f"{df['text'].iloc[i]} {df['label'].iloc[i]} {df['confidence'].iloc[i]}.wav", data, samplerate)
    print(f"音频已保存为 {df['text'].iloc[i]} {df['label'].iloc[i]} {df['confidence'].iloc[i]}.wav，采样率: {samplerate} Hz")