import io
import os
import torch
import torchaudio
import numpy as np

import nemo.collections.asr as nemo_asr
import soundfile as sf
import tempfile
import faulthandler

faulthandler.enable()

fp = os.path.join(os.path.expanduser('~'), 'advanced', 'asr', 'sample_0.wav')

model = nemo_asr.models.ASRModel.restore_from("./src/finetuned_parakeet.nemo")
model.eval()
if torch.cuda.is_available():
    model.cuda()
    
print("MODEL LOADED!")

try:
    with open(fp, 'rb') as f:
        audio_bytes = f.read()
    
    audio_buffer = io.BytesIO(audio_bytes)
    waveform, sample_rate = sf.read(audio_buffer, dtype='float32')

    # Resample if necessary
    if sample_rate != 16000:
        print(f"RESAMPLING from {sample_rate} to 16000 Hz")
        waveform = torch.tensor(waveform).unsqueeze(0)
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        waveform = waveform.squeeze(0).numpy()
        
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        sf.write(tmp_wav.name, waveform, 16000)
        tmp_filepath = tmp_wav.name
    
    print("TEMPFILE GENERATED")
    
    # Transcribe using RNNT model's `transcribe()` method
    transcription_list = model.transcribe([tmp_filepath])
    transcription = transcription_list[0]
    print("TRANSCRIBED")
    print(transcription)

except Exception as e:
    print(f"EXCEPTION: {e}")
    
print("TESTING ENDED!")
