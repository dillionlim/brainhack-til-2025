import io
import os
import torch
import torchaudio
import numpy as np

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import re

class ASRManager:
    MODEL_PATH = os.path.join(os.getcwd(), "whisper")
    SAMPLING_RATE = 16000
    CUTOFF_FREQ = 750
    NUMTAPS = 4096

    def __init__(self):
        
        self.processor = WhisperProcessor.from_pretrained(ASRManager.MODEL_PATH)
        self.model = WhisperForConditionalGeneration.from_pretrained(ASRManager.MODEL_PATH)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        self.model.generation_config.forced_decoder_ids = None
        self.model.generation_config.use_cache = False

    def asr(self, audio_bytes: bytes) -> str:
        """Performs ASR transcription on an audio file with denoising."""

        buffer = io.BytesIO(audio_bytes)
        
        speech_array, sampling_rate = torchaudio.load(buffer)
        if sampling_rate != self.SAMPLING_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=self.SAMPLING_RATE)
            speech_array = resampler(speech_array)

        input_features = self.processor(
            speech_array[0],
            sampling_rate=self.SAMPLING_RATE,
            return_tensors="pt"
        ).input_features.to(self.device)

        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return transcription
