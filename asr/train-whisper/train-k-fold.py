import os
import io
import re
import torch
import torchaudio
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from scipy.io import wavfile
from scipy.signal import firwin, filtfilt
import noisereduce as nr

import evaluate
from dataclasses import dataclass
from datasets import Dataset, Audio
from sklearn.model_selection import KFold
from typing import Any, Dict, List, Union
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

# === ASR Manager for Denoising ===
class ASRManager:
    SAMPLING_RATE = 16000
    CUTOFF_FREQ = 750
    NUMTAPS = 4096

    def fir_lowpass_filter(self, data: np.ndarray, cutoff_hz: int, fs: int) -> np.ndarray:
        nyq = 0.5 * fs
        fir_coeff = firwin(self.NUMTAPS, cutoff_hz / nyq, window='hamming')
        return filtfilt(fir_coeff, [1.0], data)

    def denoise_audio(self, audio_bytes: bytes) -> tuple[np.ndarray, int]:
        rate, data = wavfile.read(io.BytesIO(audio_bytes))
        data = data.astype(np.float32)

        if data.ndim == 2:
            filtered_channels = [
                self.fir_lowpass_filter(data[:, ch], self.CUTOFF_FREQ, rate)
                for ch in range(data.shape[1])
            ]
            filtered_data = np.stack(filtered_channels, axis=1)
        else:
            filtered_data = self.fir_lowpass_filter(data, self.CUTOFF_FREQ, rate)

        reduced_noise = nr.reduce_noise(y=filtered_data, sr=rate, use_torch=True, device="cuda" if torch.cuda.is_available() else "cpu")
        return reduced_noise, rate

# === Constants ===
MODEL = "whisper-small.en"
AUDIO_BASE_PATH = "/home/jupyter/advanced/asr/"
DATA_PATH = "/home/jupyter/advanced/asr/asr.jsonl"
OUTPUT_DIR = f"./k-fold-denoised/{MODEL}"
BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, 'best')
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

# === Load & Augment Data ===
df = pd.read_json(DATA_PATH, lines=True)[:4000]
df["audio_filepath"] = df["audio"].apply(lambda x: os.path.join(AUDIO_BASE_PATH, x))
df["sentence"] = df["transcript"]
asr_manager = ASRManager()

# Save denoised WAVs to a separate folder
denoised_output_dir = os.path.join('./', "denoised_audios")
os.makedirs(denoised_output_dir, exist_ok=True)

def denoise_and_save(filepath):
    with open(filepath, 'rb') as f:
        audio_bytes = f.read()
    denoised_audio, rate = asr_manager.denoise_audio(audio_bytes)

    norm = np.max(np.abs(denoised_audio)) or 1e-6
    normalized = (denoised_audio / norm) * 32767
    pcm_audio = normalized.astype(np.int16)

    base_name = os.path.basename(filepath).replace(".wav", "_denoised.wav")
    denoised_path = os.path.join(denoised_output_dir, base_name)

    wavfile.write(denoised_path, rate, pcm_audio)
    return denoised_path

# Create denoised dataset
denoised_paths = []
for path in tqdm(df["audio_filepath"], desc="Denoising audio"):
    denoised_paths.append(denoise_and_save(path))

df_denoised = df.copy()
df_denoised["audio_filepath"] = denoised_paths

df_combined = pd.concat([df, df_denoised], ignore_index=True)

# === Whisper Components ===
feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/{MODEL}")
tokenizer = WhisperTokenizer.from_pretrained(f"openai/{MODEL}", language="English", task="transcribe")
processor = WhisperProcessor.from_pretrained(f"openai/{MODEL}", language="English", task="transcribe")
forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")

def prepare_dataset(batch):
    audio = batch["audio_filepath"]
    waveform, sr = torchaudio.load(audio)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    batch["input_features"] = feature_extractor(waveform.squeeze().numpy(), sampling_rate=16000).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=tokenizer.pad_token_id,
)

metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# === Training Arguments ===
DEEPSPEED_CONFIG_PATH = "/home/jupyter/brainhack-til-2025/til-25-main/asr/train/dsconfig.json"
base_training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    dataloader_num_workers=4,
    learning_rate=1e-5,
    warmup_steps=100,
    max_steps=2000,
    warmup_ratio=0.22,
    weight_decay=1e-4,
    adam_beta1=0.9,
    adam_beta2=0.98,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=200,
    eval_steps=200,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    deepspeed=DEEPSPEED_CONFIG_PATH
)

# === K-Fold Cross-Validation ===
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(df_combined)):
    print(f"\n=== Fold {fold+1}/{k} ===")

    train_df = df_combined.iloc[train_idx].reset_index(drop=True)
    val_df = df_combined.iloc[val_idx].reset_index(drop=True)

    train_ds = Dataset.from_pandas(train_df).cast_column("audio_filepath", Audio(sampling_rate=16000))
    val_ds = Dataset.from_pandas(val_df).cast_column("audio_filepath", Audio(sampling_rate=16000))

    train_ds = train_ds.map(prepare_dataset, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(prepare_dataset, remove_columns=val_ds.column_names)

    model = WhisperForConditionalGeneration.from_pretrained(f"openai/{MODEL}")
    model.config.forced_decoder_ids = forced_decoder_ids

    training_args = deepcopy(base_training_args)
    training_args.output_dir = os.path.join(OUTPUT_DIR, f"fold{fold+1}")
    os.makedirs(training_args.output_dir, exist_ok=True)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    result = trainer.evaluate()
    fold_results.append(result)

    trainer.save_model(os.path.join(BEST_MODEL_DIR, f"fold{fold+1}"))
    torch.cuda.empty_cache()

avg_wer = np.mean([res["eval_wer"] for res in fold_results])
print(f"\n=== Average WER over {k} folds: {avg_wer:.2f}% ===")
