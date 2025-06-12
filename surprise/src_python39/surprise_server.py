"""Runs the surprise challenge server."""

# Unless you want to do something special with the server, you shouldn't need
# to change anything in this file.


from fastapi import FastAPI, Request
from surprise_manager_tsp import SurpriseManager
import base64


app = FastAPI()
manager = SurpriseManager()


@app.post("/surprise")
async def surprise(request: Request) -> dict[str, list[list[int]]]:
    input_json = await request.json()

    predictions = []
    for instance in input_json["instances"]:
        permutations = manager.surprise([
            base64.b64decode(slice) for slice in instance["slices"]
        ])
        predictions.append(permutations)
        
    return {"predictions": predictions}


@app.get("/health")
def health() -> dict[str, str]:
    """Health check function for your model."""
    return {"message": "health ok"}
