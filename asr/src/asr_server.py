"""Runs the ASR server."""

# Unless you want to do something special with the server, you shouldn't need
# to change anything in this file.


import base64
from fastapi import FastAPI, Request
from asr_manager import ASRManager


app = FastAPI()
manager = ASRManager()


@app.post("/asr")
async def asr(request: Request) -> dict[str, list[str]]:
    """Performs ASR on audio files.

    Args:
        request: The API request. Contains a list of audio files, encoded in
            base-64.

    Returns:
        A `dict` with a single key, `"predictions"`, mapping to a `list` of
        `str` transcriptions, in the same order as which appears in `request`.
    """
    try:
        inputs_json = await request.json()

        predictions = []
        
        audio_bytes_list = [base64.b64decode(instance["b64"]) for instance in inputs_json["instances"]]
        predictions = manager.asr_batch(audio_bytes_list)
        
        # print(predictions)
        return {"predictions": predictions}
    except Exception as e:
        print("ASR endpoint error:", str(e))
        return


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint for the server."""
    return {"message": "health ok"}
