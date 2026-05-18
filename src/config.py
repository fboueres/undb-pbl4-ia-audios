from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"
AUDIO_DIR = DATASET_DIR / "AudioWAV"

MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

SAMPLE_RATE = 22050
N_MFCC = 40
MAX_PAD_LEN = 174

EMOTION_TO_SENTIMENT = {
    "HAP": "positivo",
    "NEU": "neutro",
    "ANG": "negativo",
    "SAD": "negativo",
    "FEA": "negativo",
    "DIS": "negativo",
}

SENTIMENT_LABELS = [
    "negativo",
    "neutro",
    "positivo",
]
