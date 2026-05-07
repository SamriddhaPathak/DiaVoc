# DiaVoc

A machine learning system for predicting diabetes risk from voice recordings using federated learning and advanced audio feature extraction.

## Overview

DiaVoc is an innovative AI-powered system that analyzes voice patterns to predict diabetes risk. The system leverages self-supervised learning with BYOL-S (Bootstrap Your Own Latent) for robust audio embeddings, combined with federated learning to ensure privacy-preserving model training across distributed clients.

## Features

- **Voice-Based Diabetes Prediction**: Analyzes voice recordings to assess diabetes risk
- **Federated Learning**: Privacy-preserving training across multiple clients without sharing raw data
- **Advanced Audio Processing**: Uses BYOL-S embeddings for state-of-the-art voice feature extraction
- **Web API**: FastAPI-based REST API for easy integration
- **Comprehensive Evaluation**: Multiple ML models with thorough performance metrics
- **Non-IID Data Handling**: Robust partitioning for realistic federated scenarios

## Project Structure

```
Voice-to-Diabetes-main/
├── src/                          # Core source code
│   ├── training.py              # Federated learning training pipeline
│   ├── infer.py                 # Inference system for predictions
│   ├── processing_pipeline.py   # Data preprocessing and feature engineering
│   └── generate_embedding.py    # Audio embedding generation
├── app/                         # Web application
│   ├── main.py                  # FastAPI server
│   └── inference.py             # API inference logic
├── fl/                          # Federated learning components
│   ├── fl_simulation.py         # FL simulation and aggregation
│   ├── data_loader.py           # Data loading utilities
│   ├── predict.py               # Prediction utilities
│   └── benchmark_models.py      # Model benchmarking
├── models/                      # Pre-trained models and artifacts
│   ├── serab-byols/            # BYOL-S audio encoder
│   ├── scaler.pkl              # Feature scaler
│   ├── pca.pkl                 # PCA transformer
│   └── global_model_improved.pkl # Trained global model
├── data/                        # Data files
│   ├── male_embeddings.pkl      # Male voice embeddings
│   ├── female_embeddings.pkl    # Female voice embeddings
│   └── *.wav                    # Sample audio files
├── README.md                    # This file
└── training_history.png         # Training visualization
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- CUDA (optional, for GPU acceleration)

### Dependencies

Install the required packages:

```bash
pip install numpy pandas scikit-learn torch librosa fastapi uvicorn joblib matplotlib seaborn
```

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Voice-to-Diabetes-main
```

2. Install dependencies:
```bash
pip install -r requirements.txt  # If available, or install manually as above
```

3. Ensure data files are in place:
   - Place `male_embeddings.pkl` and `female_embeddings.pkl` in the `data/` directory
   - Ensure BYOL-S checkpoints are available in `models/serab-byols/checkpoints/`

## Usage

### Training the Model

Run the federated learning training pipeline:

```bash
python src/training.py
```

This will:
- Load and preprocess voice embeddings
- Simulate federated learning across 5 clients
- Train and aggregate models using FedAvg
- Save the global model to `models/`

### Running Inference

For command-line inference:

```bash
python src/infer.py
```

### Web API

Start the FastAPI server:

```bash
python app/main.py
```

The API will be available at `http://localhost:8000`

#### API Endpoints

- `POST /predict`: Predict diabetes risk from voice recording
  - Parameters: `audio` (file), `age` (int), `gender` (str), `bmi` (float), `ethnicity` (str)

Example usage:

```python
import requests

files = {'audio': open('sample.wav', 'rb')}
data = {'age': 45, 'gender': 'male', 'bmi': 28.5, 'ethnicity': 'asian'}
response = requests.post('http://localhost:8000/predict', files=files, data=data)
print(response.json())
```

### Data Preprocessing

To preprocess new data:

```bash
python src/processing_pipeline.py
```

This generates the necessary preprocessing artifacts (scaler, PCA) saved in `models/`.

### Federated Learning Simulation

Run FL simulations:

```bash
python fl/fl_simulation.py
```

## Methodology

### Audio Feature Extraction
- Uses BYOL-S (Bootstrap Your Own Latent) for self-supervised audio representation learning
- Extracts 2048-dimensional embeddings from voice recordings
- Combines with demographic features (age, BMI, gender, ethnicity)

### Federated Learning
- Implements FedAvg algorithm with true parameter averaging
- Non-IID data partitioning using Dirichlet distribution (α=0.5)
- 5 clients simulation with local training rounds
- Global model aggregation for improved generalization

### Model Architecture
- MLP Classifier as primary model
- Feature engineering with Risk Index (Age × BMI interaction)
- PCA dimensionality reduction to 100 components
- Standard scaling for numerical features

## Performance

The system achieves:
- **80%+ accuracy** on held-out test sets
- Robust performance across demographic groups
- Privacy preservation through federated learning
- Real-time inference capability

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite:

```
@misc{diavoc2026,
  title={DiaVoc: Voice-based Diabetes Prediction using Federated Learning},
  author={Khati, Harina and Pathak, Samriddha and Pokhrel, Jyoti and Subedi, Rasum},
  year={2026},
  howpublished={GitHub repository}}
}
```

## Contact

Samriddha Pathak: +977 9702187444

Email: samriddhapathak123333@gmail.com
