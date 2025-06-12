import io
import os
import torch
import torchaudio
import numpy as np
from scipy.io import wavfile
from scipy.signal import firwin, filtfilt
import noisereduce as nr

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

    def fir_lowpass_filter(self, data: np.ndarray, cutoff_hz: int, fs: int) -> np.ndarray:
        nyq = 0.5 * fs
        fir_coeff = firwin(self.NUMTAPS, cutoff_hz / nyq, window='hamming')
        return filtfilt(fir_coeff, [1.0], data)

    def denoise_audio(self, audio_bytes: bytes) -> tuple[np.ndarray, int]:
        # Read audio from bytes using scipy
        rate, data = wavfile.read(io.BytesIO(audio_bytes))
        data = data.astype(np.float32)

        # Apply lowpass filter
        if data.ndim == 2:  # Stereo
            filtered_channels = [
                self.fir_lowpass_filter(data[:, ch], self.CUTOFF_FREQ, rate)
                for ch in range(data.shape[1])
            ]
            filtered_data = np.stack(filtered_channels, axis=1)
        else:
            filtered_data = self.fir_lowpass_filter(data, self.CUTOFF_FREQ, rate)

        # Apply noise reduction
        reduced_noise = nr.reduce_noise(y=filtered_data, sr=rate, use_torch=True, device="cuda")

        return reduced_noise, rate
    
    def remove_duplicate_phrases(self, text):
        """
        Removes repeated phrases from a sentence using regex.
        This targets repeated word sequences with minor separators.
        """
        return re.sub(r'((\b\w+\b.{1,2}\w+\b)+).+\1', r'\1', text, flags=re.IGNORECASE)

    def asr(self, audio_bytes: bytes) -> str:
        """Performs ASR transcription on an audio file with denoising."""

        # Denoise the audio bytes
        denoised_audio, rate = self.denoise_audio(audio_bytes)

        # Normalize to 16-bit PCM and load via torchaudio
        norm = np.max(np.abs(denoised_audio)) or 1e-6
        normalized = (denoised_audio / norm) * 32767
        pcm_audio = normalized.astype(np.int16)

        buffer = io.BytesIO()
        wavfile.write(buffer, rate, pcm_audio)
        buffer.seek(0)

        # Load with torchaudio
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
        
        cleaned_transcription = self.remove_duplicate_phrases(transcription)
        
        return cleaned_transcription
