"""Runs the OCR server."""

# Unless you want to do something special with the server, you shouldn't need
# to change anything in this file.


import base64

from fastapi import FastAPI, Request
from ocr_manager import OCRManager

app = FastAPI()
manager = OCRManager()


@app.post("/ocr")
async def ocr(request: Request) -> dict[str, list[str]]:
    """Performs OCR on images of documents.

    Args:
        request: The API request. Contains a list of images, encoded in
            base-64.

    Returns:
        A `dict` with a single key, `"predictions"`, mapping to a `list` of
        `str` document text, in the same order as which appears in `request`.
    """

    inputs_json = await request.json()

    image_bytes_ls = []
    for i, instance in enumerate(inputs_json["instances"]):
        # Reads the base-64 encoded image and decodes it into bytes.
        image_bytes_ls.append(base64.b64decode(instance["b64"]))

    # Performs OCR on batch.    
    predictions = manager.ocr(image_bytes_ls)

    return {"predictions": predictions}


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint for your model."""
    return {"message": "health ok"}
