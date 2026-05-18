import pandas as pd
from config import AUDIO_DIR, EMOTION_TO_SENTIMENT

def load_audio_metadata():
    records = []

    for file_path in AUDIO_DIR.glob("*.wav"):
        parts = file_path.stem.split("_")

        if len(parts) < 3:
            continue

        emotion_code = parts[2]

        if emotion_code not in EMOTION_TO_SENTIMENT:
            continue

        sentiment = EMOTION_TO_SENTIMENT[emotion_code]

        records.append({
            "file_name": file_path.name,
            "file_path": str(file_path),
            "emotion_code": emotion_code,
            "sentiment": sentiment
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = load_audio_metadata()

    print(df.head())
    print()
    print("Total de áudios:", len(df))
    print()
    print("Distribuição por sentimento:")
    print(df["sentiment"].value_counts())
