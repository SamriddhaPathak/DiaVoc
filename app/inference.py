# app/inference.py
import numpy as np
import joblib
import torch
import librosa
from pathlib import Path
import warnings
from models.sarab_byols import load_model, get_scene_embeddings

warnings.filterwarnings('ignore')

class DiaVocInferenceSystem:
    def __init__(self, model_dir='models', audio_checkpoint=None):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.scaler = joblib.load(self.model_dir / 'scaler.pkl')
        self.pca = joblib.load(self.model_dir / 'pca.pkl')
        self.feature_selector = joblib.load(self.model_dir / 'feature_selector.pkl')

        model_path = self.model_dir / 'global_model_improved.pkl'
        self.model = joblib.load(model_path)

        self.audio_encoder = load_model(audio_checkpoint)
        self.audio_encoder.to(self.device)
        self.audio_encoder.eval()

        self.sample_rate = 16000

    def _extract_voice_embedding(self, wav_path):
        audio, _ = librosa.load(wav_path, sr=self.sample_rate)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = get_scene_embeddings(audio_tensor, self.audio_encoder)

        return embedding.cpu().numpy().flatten()

    def _extract_handcrafted(self, embedding):
        chunk_size = len(embedding) // 16
        means = [np.mean(embedding[i*chunk_size:(i+1)*chunk_size]) for i in range(16)]
        stds = [np.std(embedding[i*chunk_size:(i+1)*chunk_size]) for i in range(16)]
        return np.array(means + stds)

    def predict_from_wav(self, wav_path, age, gender, bmi, ethnicity='asian'):
        raw_embedding = self._extract_voice_embedding(wav_path)

        gender_bin = 1 if str(gender).lower() in ['male', '1'] else 0
        ethnicity_bin = 1 if str(ethnicity).lower() in ['asian', 'asia'] else 0

        tabular = np.array([age, gender_bin, bmi, ethnicity_bin])
        handcrafted = self._extract_handcrafted(raw_embedding)
        reduced_emb = self.pca.transform(raw_embedding.reshape(1, -1))

        X = np.hstack([
            tabular.reshape(1, -1),
            handcrafted.reshape(1, -1),
            reduced_emb
        ])

        X_scaled = self.scaler.transform(X)
        X_final = self.feature_selector.transform(X_scaled)

        prob = self.model.predict_proba(X_final)[0][1]
        prediction = int(prob >= 0.5)

        return {
            "diagnosis": "DIABETIC" if prediction else "HEALTHY",
            "probability": float(prob)
        }
