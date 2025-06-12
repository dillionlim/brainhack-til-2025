import os
import json
import time
import random
import torch
import soundfile as sf
import lightning.pytorch as pl
from omegaconf import OmegaConf

# For resuming training or to initialize from a non-NVIDIA model, use
# init_from_nemo_model: "<path/to/nemo/model.nemo>"

yaml_str = """
init_from_pretrained_model: "nvidia/parakeet-tdt-0.6b-v2"

trainer:
  devices: 1
  accelerator: gpu
  max_epochs: 12
  precision: 32
  accumulate_grad_batches: 8

exp_manager:
  exp_dir: /tmp/asr_finetune_exp
  name: finetune_run
  create_tensorboard_logger: true
  create_checkpoint_callback: false

model:
  logit_normalization: true
  train_ds:
    manifest_filepath: /tmp/train_manifest.json
    batch_size: 1
    shuffle: true
    is_tarred: false
    num_workers: 1
    sample_rate: 16000
    max_duration: 40.0
    min_duration: 0.1

  validation_ds:
    manifest_filepath: /tmp/val_manifest.json"
    batch_size: 1
    shuffle: false
    is_tarred: false
    num_workers: 1
    sample_rate: 16000
    max_duration: 40.0
    min_duration: 0.1

  optim:
    name: adam
    lr: 1e-5  # LOWERED
    betas: [0.9, 0.999]
    weight_decay: 1e-5
    sched:
      name: CosineAnnealing
      min_lr: 1e-5
      warmup_steps: 500

  tokenizer:
    update_tokenizer: false
  char_labels:
    update_labels: false

  decoding:
    strategy: greedy
    preserve_alignments: false
    compute_timestamps: false
    max_symbols_per_step: 30
"""

from nemo.core.config import hydra_runner
from nemo.utils import logging, model_utils
from nemo.utils.exp_manager import exp_manager
from nemo.utils.get_rank import is_global_rank_zero
from nemo.utils.trainer_utils import resolve_trainer_cfg
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import ASRModel

# os.environ['TORCH_SHOW_CPP_STACKTRACES'] = "1"
# torch.utils.data._utils.worker.IS_ENABLED = False

# Paths
INPUT_JSONL = "./data/asr.jsonl"
AUDIO_BASE_PATH = "./data/"
TRAIN_MANIFEST = "/tmp/train_manifest.json"
VAL_MANIFEST = "/tmp/val_manifest.json"
SAVE_PATH = "./finetuned_parakeet.nemo"
NUM_EXAMPLES = 4500
TRAIN_SPLIT_RATIO = 0.9

def prepare_manifests():
    print("Preparing manifests...")
    examples = []
    os.makedirs(os.path.dirname(TRAIN_MANIFEST), exist_ok=True)
    with open(INPUT_JSONL, "r") as fin:
        for i, line in enumerate(fin):
            if i >= NUM_EXAMPLES:
                break
            item = json.loads(line)
            audio_path = os.path.join(AUDIO_BASE_PATH, item["audio"])
            try:
                info = sf.info(audio_path)
                duration = info.frames / info.samplerate
            except RuntimeError:
                print(f"Skipping unreadable file: {audio_path}")
                continue
            examples.append({
                "audio_filepath": audio_path,
                "duration": duration,
                "text": item["transcript"]
            })

    random.shuffle(examples)
    split_index = int(len(examples) * TRAIN_SPLIT_RATIO)
    write_manifest(examples[:split_index], TRAIN_MANIFEST)
    write_manifest(examples[split_index:], VAL_MANIFEST)
    print(f"Prepared {len(examples[:split_index])} training and {len(examples[split_index:])} validation examples.")

def write_manifest(data, path):
    with open(path, "w") as fout:
        for item in data:
            fout.write(json.dumps(item) + "\n")

def get_base_model(trainer, cfg):
    model_path = cfg.get("init_from_nemo_model", None)
    pretrained = cfg.get("init_from_pretrained_model", None)
    if model_path and pretrained:
        raise ValueError("Specify only one: `init_from_nemo_model` or `init_from_pretrained_model`.")
    if not model_path and not pretrained:
        raise ValueError("Specify one of `init_from_nemo_model` or `init_from_pretrained_model`.")

    if model_path:
        model = ASRModel.restore_from(restore_path=model_path)
    else:
        num_ranks = trainer.num_devices * trainer.num_devices
        if num_ranks > 1 and is_global_rank_zero():
            model = ASRModel.from_pretrained(model_name=pretrained)
        else:
            wait_time = max(cfg.get("exp_manager", {}).get("seconds_to_sleep", 10), 10)
            logging.info(f"Waiting {wait_time}s for model download sync.")
            time.sleep(wait_time)
            model = ASRModel.from_pretrained(model_name=pretrained)

    model.set_trainer(trainer)
    return model

def check_vocabulary(model, cfg):
    tokenizer_cfg = OmegaConf.select(cfg, "model.tokenizer", default=None)
    char_labels_cfg = OmegaConf.select(cfg, "model.char_labels", default=None)

    if tokenizer_cfg and tokenizer_cfg.get("update_tokenizer", False):
        if char_labels_cfg and char_labels_cfg.get("update_labels", False):
            raise ValueError("Can't update tokenizer and char_labels at the same time.")
        else:
            model = update_tokenizer(model, tokenizer_cfg.get("dir"), tokenizer_cfg.get("type"))
    elif char_labels_cfg and char_labels_cfg.get("update_labels", False):
        model.change_vocabulary(new_vocabulary=char_labels_cfg.get("labels"))
        logging.warning("Vocabulary updated with char labels.")
    else:
        logging.info("Reusing vocabulary from pre-trained model.")
    return model

def update_tokenizer(model, tokenizer_dir, tokenizer_type):
    if not tokenizer_dir:
        raise ValueError("Must provide tokenizer_dir.")
    vocab_size = model.tokenizer.vocab_size
    decoder = model.decoder.state_dict()
    joint_state = model.joint.state_dict() if hasattr(model, 'joint') else None
    model.change_vocabulary(new_tokenizer_dir=tokenizer_dir, new_tokenizer_type=tokenizer_type)

    if model.tokenizer.vocab_size == vocab_size:
        model.decoder.load_state_dict(decoder)
        if joint_state:
            model.joint.load_state_dict(joint_state)
    else:
        logging.warning("Tokenizer vocab changed. Decoder reinitialized.")
    return model

def setup_dataloaders(model, cfg):
    cfg = model_utils.convert_model_config_to_dict_config(cfg)
    cfg.model.train_ds.sample_rate = 16000
    cfg.model.validation_ds.sample_rate = 16000
    cfg.model.train_ds.text_field = "text"
    cfg.model.validation_ds.text_field = "text"
    model.setup_training_data(cfg.model.train_ds)
    model.setup_multiple_validation_data(cfg.model.validation_ds)
    return model

def main():
    prepare_manifests()
    cfg = OmegaConf.create(yaml_str)
    cfg.model.train_ds.manifest_filepath = TRAIN_MANIFEST
    cfg.model.validation_ds.manifest_filepath = VAL_MANIFEST

    trainer_cfg = resolve_trainer_cfg(cfg.trainer)
    trainer_cfg["logger"] = False
    trainer_cfg["num_sanity_val_steps"] = 0
    trainer_cfg["precision"] = 32
    trainer = pl.Trainer(**trainer_cfg)
    exp_manager(trainer, cfg.get("exp_manager", None))

    model = get_base_model(trainer, cfg)
    model = check_vocabulary(model, cfg)
    model = setup_dataloaders(model, cfg)
    model.setup_optimization(cfg.model.optim)

    # Disable CUDA graph decoding for RNNT
    OmegaConf.set_struct(model.decoding.cfg, False)
    model.decoding.cfg.preserve_alignments = False
    OmegaConf.set_struct(model.cfg, False)
    model.cfg.preserve_alignments = False
    if hasattr(model, "_wer"):
        model._wer.preserve_alignments = False

    print(f"Model type: {type(model)}")
    trainer.fit(model)

    model.save_to(SAVE_PATH)
    print(f"Model saved to: {SAVE_PATH}")

if __name__ == "__main__":
    main()