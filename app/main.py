from fastapi import FastAPI, UploadFile, File, Form
import shutil
import uuid
from pathlib import Path

from app.inference import DiaVocInferenceSystem

app = FastAPI(title="Voice-to-Diabetes API")

# Load model ONCE at startup
diavoc = DiaVocInferenceSystem(
    model_dir="models",
    audio_checkpoint="models/serab-byols/checkpoints/default2048_BYOLAs64x96-2105311814-e100-bs256-lr0003-rs42.pth"
)

UPLOAD_DIR = Path("temp_audio")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/predict")
async def predict_diabetes(
    audio: UploadFile = File(...),
    age: int = Form(...),
    gender: str = Form(...),
    bmi: float = Form(...),
    ethnicity: str = Form("asian")
):
    # Save uploaded audio temporarily
    file_id = f"{uuid.uuid4()}.wav"
    file_path = UPLOAD_DIR / file_id

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    try:
        result = diavoc.predict_from_wav(
            wav_path=str(file_path),
            age=age,
            gender=gender,
            bmi=bmi,
            ethnicity=ethnicity
        )
        return result
    finally:
        file_path.unlink(missing_ok=True)
