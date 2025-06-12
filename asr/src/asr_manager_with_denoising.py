import io
import os
import torch
import torchaudio
import noisereduce as nr
import numpy as np
import soundfile as sf
import tempfile
from noisereduce.torchgate import TorchGate as TG
import nemo.collections.asr as nemo_asr

class ASRManager:
    SAMPLING_RATE = 16000

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nemo_asr.models.ASRModel.restore_from("./finetuned_parakeet_12.nemo")
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tg = TG(sr=self.SAMPLING_RATE, nonstationary=True).to(self.device)
    
    def preprocess(self, audio_bytes: bytes) -> np.ndarray:
        buf = io.BytesIO(audio_bytes)
        wav_np, sr = sf.read(buf, dtype="float32")
        # make mono + normalize safely
        if wav_np.ndim == 2:
            wav_np = wav_np.mean(axis=1)
        eps = 1e-8
        wav_np = wav_np / max(eps, np.max(np.abs(wav_np)))

        # resample if needed
        if sr != self.SAMPLING_RATE:
            wav_t = torch.from_numpy(wav_np).unsqueeze(0)
            wav_t = torchaudio.transforms.Resample(sr, self.SAMPLING_RATE)(wav_t)
            wav_np = wav_t.squeeze(0).numpy()

        # to tensor & device
        wav_t = torch.from_numpy(wav_np).unsqueeze(0).to(self.device)

        # denoise with clamp & fallback
        try:
            with torch.no_grad():
                denoised_t = self.tg(wav_t)
                denoised_t = torch.nan_to_num(
                    denoised_t,
                    nan=0.0, posinf=0.0, neginf=0.0
                ).clamp(-1.0, 1.0)
        except Exception:
            # if TorchGate still fails, use raw audio
            denoised_t = wav_t

        denoised = denoised_t.squeeze(0).cpu().numpy()

        # final normalize
        den_max = max(eps, np.max(np.abs(denoised)))
        denoised = denoised / den_max

        return denoised.astype(np.float32)

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