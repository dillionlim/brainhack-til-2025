import io
import os
import torch
import torchaudio
import numpy as np
import soundfile as sf
import tempfile
import nemo.collections.asr as nemo_asr

class ASRManager:
    SAMPLING_RATE = 16000

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nemo_asr.models.ASRModel.restore_from("./finetuned_parakeet_12.nemo")
        self.model = self.model.to(self.device)
        self.model.eval()

    def preprocess(self, audio_bytes: bytes) -> torch.Tensor:
        audio_buffer = io.BytesIO(audio_bytes)
        waveform, sample_rate = sf.read(audio_buffer, dtype='float32')

        if sample_rate != self.SAMPLING_RATE:
            waveform = torch.tensor(waveform).unsqueeze(0)
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.SAMPLING_RATE)(waveform)
            waveform = waveform.squeeze(0).numpy()

        if waveform.ndim == 2:
            waveform = np.mean(waveform, axis=1)

        waveform = waveform.astype(np.float32)
        if np.isnan(waveform).any() or np.isinf(waveform).any():
            raise ValueError("Waveform contains NaNs or Infs")

        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val

        return torch.tensor(waveform, dtype=torch.float32)

    def asr_batch(self, audio_bytes_list: list[bytes]) -> list[str]:
        try:
            tmp_files = []

            # Save all audio bytes to temporary WAV files
            for audio_bytes in audio_bytes_list:
                waveform = self.preprocess(audio_bytes)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                    sf.write(tmp_wav.name, waveform, self.SAMPLING_RATE)
                    tmp_files.append(tmp_wav.name)
            
            # Run batch transcription
            transcriptions = self.model.transcribe(tmp_files,batch_size=4)
            
            results = [t.text for t in transcriptions]

            # Clean up temp files
            for f in tmp_files:
                os.remove(f)
            return results
        except Exception as e:
            print(f"ASR BATCH ERROR: {e}")
            raise
