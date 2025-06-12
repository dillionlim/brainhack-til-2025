import evaluate
import os
import pandas as pd
import torch

from dataclasses import dataclass
from datasets import Dataset, DatasetDict, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from typing import Any, Dict, List, Union

# === Model Used ===
MODEL = "whisper-base.en"

# === Paths ===
AUDIO_BASE_PATH = "/home/jupyter/advanced/asr/"
DATA_PATH = "/home/jupyter/advanced/asr/asr.jsonl"
OUTPUT_DIR = f"./{MODEL}"

BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, 'best')
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

# === Load and Prepare Dataset ===
df = pd.read_json(DATA_PATH, lines=True)[:4500]

# Map your custom field names to expected ones
df["audio_filepath"] = df["audio"].apply(lambda x: os.path.join(AUDIO_BASE_PATH, x))
df["sentence"] = df["transcript"]

# Create Hugging Face Dataset and cast audio column
dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=16000))

# Train/Test Split
split = dataset.train_test_split(test_size=0.1)
common_voice = DatasetDict({
    "train": split["train"],
    "test": split["test"]
})

# === Whisper Setup ===
feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/{MODEL}")
tokenizer = WhisperTokenizer.from_pretrained(f"openai/{MODEL}", language="English", task="transcribe")
processor = WhisperProcessor.from_pretrained(f"openai/{MODEL}", language="English", task="transcribe")

# === Dataset Processing Function ===
def prepare_dataset(batch):
    audio = batch["audio_filepath"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

common_voice = common_voice.map(
    prepare_dataset,
    remove_columns=common_voice["train"].column_names,
    num_proc=1
)

# === Load Whisper Model ===
model = WhisperForConditionalGeneration.from_pretrained(f"openai/{MODEL}")
forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
model.config.forced_decoder_ids = forced_decoder_ids

# === Data Collator ===
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
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# === Evaluation Metric ===
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
# DEEPSPEED_CONFIG_PATH = "/home/jupyter/brainhack-til-2025/til-25-main/asr/train/dsconfig.json"
training_args = Seq2SeqTrainingArguments(
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
    eval_steps=100,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    # deepspeed=DEEPSPEED_CONFIG_PATH
)

# === Initialize Trainer ===
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# === Train ===
trainer.train()

trainer.save_model(BEST_MODEL_DIR)
