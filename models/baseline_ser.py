import argparse
import json
import random
import time
from collections import Counter
from io import BytesIO
from pathlib import Path

import joblib
import librosa
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, SVC
from tqdm import tqdm


VALID_LABELS = [
    "neutral",
    "happy",
    "angry",
    "surprised",
    "sad",
    "disgusted",
    "fearful",
]

ACOUSTIC_CONFIG = {
    "sample_rate": 16000,
    "max_duration": 6.0,
}

TEAMMATE_MEL_CONFIG = {
    "sample_rate": 22050,
    "max_duration": 8.0,
    "n_mels": 128,
    "n_fft": 2048,
    "hop_length": 512,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cantonese speech emotion recognition baseline using acoustic features."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--pattern", default="train-*.parquet")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-duration", type=float, default=6.0)
    parser.add_argument(
        "--preprocess-profile",
        choices=["acoustic", "teammate_mel", "fused"],
        default="acoustic",
        help="Use the original, read.py log-mel, or combined audio feature front end.",
    )
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Drop examples with confidence below this value.",
    )
    parser.add_argument(
        "--max-samples-per-label",
        type=int,
        default=1000,
        help="Cap samples per class for a quick reproducible baseline. Use 0 for all data.",
    )
    parser.add_argument(
        "--model",
        choices=["logreg", "linear_svm", "rbf_svm", "sgd"],
        default="logreg",
    )
    parser.add_argument(
        "--svc-c",
        type=float,
        default=3.0,
        help="Regularization parameter for rbf_svm; ignored for other models.",
    )
    parser.add_argument(
        "--svc-gamma-multiplier",
        type=float,
        default=1.0,
        help="RBF gamma multiplier relative to 1 / feature_count after scaling.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Recompute features even if the cached feature file exists.",
    )
    return parser.parse_args()


def apply_preprocess_profile(args):
    if args.preprocess_profile == "teammate_mel":
        for key, value in TEAMMATE_MEL_CONFIG.items():
            setattr(args, key, value)
    return args


def frontend_settings(args):
    if args.preprocess_profile == "fused":
        return {"acoustic": ACOUSTIC_CONFIG, "teammate_mel": TEAMMATE_MEL_CONFIG}
    return {
        "sample_rate": args.sample_rate,
        "max_duration": args.max_duration,
        "n_mels": args.n_mels,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
    }


def find_parquet_files(data_dir, pattern):
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {data_dir} with pattern {pattern!r}")
    return files


def feature_cache_path(args):
    cap = args.max_samples_per_label if args.max_samples_per_label > 0 else "all"
    if args.preprocess_profile == "fused":
        front_end = "fused_acoustic_sr16000_dur6_teammate_sr22050_dur8_mel128_fft2048_hop512"
    else:
        front_end = (
            f"{args.preprocess_profile}_sr{args.sample_rate}_dur{args.max_duration:g}_"
            f"mel{args.n_mels}_fft{args.n_fft}_hop{args.hop_length}"
        )
    name = f"features_{front_end}_cap{cap}_conf{args.min_confidence:g}_{args.pattern.replace('*', 'star')}.npz"
    return args.output_dir / name


def summarize_parquets(files):
    total_rows = 0
    row_groups = 0
    for path in files:
        pf = pq.ParquetFile(path)
        total_rows += pf.metadata.num_rows
        row_groups += pf.num_row_groups
    return total_rows, row_groups


def file_manifest(files):
    return [
        {
            "name": path.name,
            "size": path.stat().st_size,
            "mtime": path.stat().st_mtime,
        }
        for path in files
    ]


def safe_to_float_array(values):
    arr = np.asarray(values, dtype=np.float32)
    arr[~np.isfinite(arr)] = 0.0
    return arr


def stats(values):
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    mean = np.mean(arr, axis=1)
    std = np.std(arr, axis=1)
    return np.concatenate([mean, std]).astype(np.float32)


def extract_acoustic_features(audio_bytes, sample_rate, max_duration):
    y, sr = librosa.load(
        BytesIO(audio_bytes),
        sr=sample_rate,
        mono=True,
        duration=max_duration,
    )
    if y.size == 0:
        raise ValueError("empty audio")

    y = librosa.util.normalize(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    feature_parts = [
        stats(mfcc),
        stats(delta),
        stats(delta2),
        stats(chroma),
        stats(spectral_centroid),
        stats(spectral_bandwidth),
        stats(spectral_rolloff),
        stats(zero_crossing_rate),
        stats(rms),
        np.asarray([len(y) / sr], dtype=np.float32),
    ]
    return safe_to_float_array(np.concatenate(feature_parts))


def extract_teammate_mel_features(audio_bytes, sample_rate, max_duration, n_mels, n_fft, hop_length):
    y, sr = librosa.load(BytesIO(audio_bytes), sr=sample_rate, mono=True)
    if y.size == 0:
        raise ValueError("empty audio")

    target_samples = int(sample_rate * max_duration)
    original_duration = min(len(y), target_samples) / sr
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)), mode="constant")
    else:
        y = y[:target_samples]

    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    feature_parts = [
        stats(mel_db),
        stats(mfcc),
        stats(delta),
        stats(delta2),
        np.asarray([original_duration], dtype=np.float32),
    ]
    return safe_to_float_array(np.concatenate(feature_parts))


def extract_features(audio_bytes, args):
    if args.preprocess_profile == "fused":
        acoustic = extract_acoustic_features(
            audio_bytes,
            ACOUSTIC_CONFIG["sample_rate"],
            ACOUSTIC_CONFIG["max_duration"],
        )
        teammate_mel = extract_teammate_mel_features(
            audio_bytes,
            TEAMMATE_MEL_CONFIG["sample_rate"],
            TEAMMATE_MEL_CONFIG["max_duration"],
            TEAMMATE_MEL_CONFIG["n_mels"],
            TEAMMATE_MEL_CONFIG["n_fft"],
            TEAMMATE_MEL_CONFIG["hop_length"],
        )
        return safe_to_float_array(np.concatenate([acoustic, teammate_mel]))
    if args.preprocess_profile == "teammate_mel":
        return extract_teammate_mel_features(
            audio_bytes,
            args.sample_rate,
            args.max_duration,
            args.n_mels,
            args.n_fft,
            args.hop_length,
        )
    return extract_acoustic_features(audio_bytes, args.sample_rate, args.max_duration)


def should_take(label, counts, cap):
    if label not in VALID_LABELS:
        return False
    if cap <= 0:
        return True
    return counts[label] < cap


def all_caps_filled(counts, cap):
    if cap <= 0:
        return False
    return all(counts[label] >= cap for label in VALID_LABELS)


def iter_rows(files, batch_size):
    for path in files:
        pf = pq.ParquetFile(path)
        columns = ["audio_file", "label", "duration", "confidence", "text"]
        for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
            for row in batch.to_pylist():
                yield path, row


def build_features(args, files):
    cache_path = feature_cache_path(args)
    manifest = file_manifest(files)
    if cache_path.exists() and not args.force_extract:
        data = np.load(cache_path, allow_pickle=True)
        cached_manifest = json.loads(str(data["file_manifest"].item())) if "file_manifest" in data else None
        if cached_manifest == manifest:
            print(f"Loaded cached features: {cache_path}")
            return (
                data["X"],
                data["y"],
                data["paths"].tolist(),
                data["texts"].tolist(),
                json.loads(str(data["label_counts"].item())),
            )
        print(f"Cache exists but parquet files changed, rebuilding features: {cache_path}")

    rng = random.Random(args.seed)
    shuffled_files = list(files)
    rng.shuffle(shuffled_files)

    X = []
    y = []
    paths = []
    texts = []
    counts = Counter()
    skipped = Counter()
    accepted = 0
    total_rows, _ = summarize_parquets(files)
    cap = args.max_samples_per_label

    with tqdm(total=total_rows, desc="extract", unit="row", mininterval=5.0, miniters=5000) as progress:
        for source_file, row in iter_rows(shuffled_files, args.batch_size):
            progress.update(1)
            label = row.get("label")
            confidence = row.get("confidence")

            if label not in VALID_LABELS:
                skipped["invalid_label"] += 1
                continue
            if confidence is not None and float(confidence) < args.min_confidence:
                skipped["low_confidence"] += 1
                continue
            if not should_take(label, counts, cap):
                skipped["cap_reached"] += 1
                if all_caps_filled(counts, cap):
                    break
                continue

            audio = row.get("audio_file") or {}
            audio_bytes = audio.get("bytes")
            if not audio_bytes:
                skipped["missing_audio"] += 1
                continue

            try:
                feats = extract_features(audio_bytes, args)
            except Exception:
                skipped["decode_or_feature_error"] += 1
                continue

            X.append(feats)
            y.append(VALID_LABELS.index(label))
            paths.append(f"{source_file.name}:{audio.get('path', '')}")
            texts.append(row.get("text") or "")
            counts[label] += 1
            accepted += 1
            if accepted % 500 == 0:
                progress.set_postfix({k: counts[k] for k in VALID_LABELS}, refresh=False)
            if all_caps_filled(counts, cap):
                break

    if not X:
        raise RuntimeError("No usable audio examples were extracted.")

    X = np.vstack(X).astype(np.float32)
    y = np.asarray(y, dtype=np.int64)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        X=X,
        y=y,
        paths=np.asarray(paths, dtype=object),
        texts=np.asarray(texts, dtype=object),
        label_counts=json.dumps(dict(counts), ensure_ascii=False),
        skipped=json.dumps(dict(skipped), ensure_ascii=False),
        file_manifest=json.dumps(manifest, ensure_ascii=False),
    )
    print(f"Saved features: {cache_path}")
    print(f"Label counts: {dict(counts)}")
    print(f"Skipped rows: {dict(skipped)}")
    return X, y, paths, texts, dict(counts)


def make_model(name, seed, svc_c, svc_gamma_multiplier, feature_count):
    if name == "logreg":
        classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
        )
    elif name == "linear_svm":
        classifier = LinearSVC(class_weight="balanced", random_state=seed, dual="auto")
    elif name == "rbf_svm":
        classifier = SVC(
            C=svc_c,
            kernel="rbf",
            gamma=svc_gamma_multiplier / feature_count,
            class_weight="balanced",
            random_state=seed,
        )
    else:
        classifier = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=1e-4,
            l1_ratio=0.15,
            class_weight="balanced",
            max_iter=2000,
            tol=1e-4,
            random_state=seed,
        )
    return make_pipeline(StandardScaler(), classifier)


def train_and_evaluate(args, X, y, paths, texts, label_counts):
    X_train, X_test, y_train, y_test, paths_train, paths_test, _, texts_test = train_test_split(
        X,
        y,
        paths,
        texts,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    model = make_model(
        args.model,
        args.seed,
        args.svc_c,
        args.svc_gamma_multiplier,
        X.shape[1],
    )
    start = time.time()
    model.fit(X_train, y_train)
    train_seconds = time.time() - start
    pred = model.predict(X_test)

    metrics = {
        "model": args.model,
        "svc_c": args.svc_c if args.model == "rbf_svm" else None,
        "svc_gamma_multiplier": args.svc_gamma_multiplier if args.model == "rbf_svm" else None,
        "preprocess_profile": args.preprocess_profile,
        "frontend_settings": frontend_settings(args),
        "sample_rate": args.sample_rate,
        "max_duration": args.max_duration,
        "n_mels": args.n_mels,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "max_samples_per_label": args.max_samples_per_label,
        "min_confidence": args.min_confidence,
        "num_examples": int(len(y)),
        "num_train": int(len(y_train)),
        "num_test": int(len(y_test)),
        "label_counts": label_counts,
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted")),
        "train_seconds": train_seconds,
        "classification_report": classification_report(
            y_test,
            pred,
            target_names=VALID_LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / f"baseline_{args.model}.joblib"
    report_json = args.output_dir / f"baseline_{args.model}_report.json"
    report_txt = args.output_dir / f"baseline_{args.model}_report.txt"
    predictions_csv = args.output_dir / f"baseline_{args.model}_predictions.csv"

    joblib.dump({"model": model, "labels": VALID_LABELS, "args": vars(args)}, model_path)
    report_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    text_report = classification_report(
        y_test,
        pred,
        target_names=VALID_LABELS,
        zero_division=0,
    )
    report_txt.write_text(
        "\n".join(
            [
                f"Accuracy: {metrics['accuracy']:.4f}",
                f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}",
                f"Macro F1: {metrics['macro_f1']:.4f}",
                f"Weighted F1: {metrics['weighted_f1']:.4f}",
                "",
                text_report,
                "",
                "Confusion matrix rows=true cols=pred:",
                np.array2string(np.asarray(metrics["confusion_matrix"]), separator=", "),
            ]
        ),
        encoding="utf-8",
    )

    with predictions_csv.open("w", encoding="utf-8") as f:
        f.write("path,true_label,pred_label,text\n")
        for path, true_id, pred_id, text in zip(paths_test, y_test, pred, texts_test):
            clean_text = str(text).replace('"', '""').replace("\n", " ")
            f.write(f'"{path}",{VALID_LABELS[true_id]},{VALID_LABELS[pred_id]},"{clean_text}"\n')

    print("\n=== Baseline result ===")
    print(f"examples: {len(y)} train: {len(y_train)} test: {len(y_test)}")
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"balanced_accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"macro_f1: {metrics['macro_f1']:.4f}")
    print(f"weighted_f1: {metrics['weighted_f1']:.4f}")
    print(text_report)
    print(f"Saved model: {model_path}")
    print(f"Saved report: {report_txt}")
    print(f"Saved json: {report_json}")
    print(f"Saved predictions: {predictions_csv}")


def main():
    args = apply_preprocess_profile(parse_args())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = find_parquet_files(args.data_dir, args.pattern)
    total_rows, row_groups = summarize_parquets(files)
    print(f"Found {len(files)} parquet files, {total_rows} rows, {row_groups} row groups.")
    print(f"Valid labels: {VALID_LABELS}")
    if args.preprocess_profile == "teammate_mel":
        print(f"Using read.py mel preprocessing settings: {TEAMMATE_MEL_CONFIG}")
    elif args.preprocess_profile == "fused":
        print(f"Using fused acoustic settings: {ACOUSTIC_CONFIG} + {TEAMMATE_MEL_CONFIG}")
    X, y, paths, texts, label_counts = build_features(args, files)
    train_and_evaluate(args, X, y, paths, texts, label_counts)


if __name__ == "__main__":
    main()
