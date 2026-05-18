import os
import subprocess

from dotenv import load_dotenv

load_dotenv()

username = os.getenv("KAGGLE_USERNAME")
key = os.getenv("KAGGLE_KEY")

if not username or not key:
    raise ValueError(
        "KAGGLE_USERNAME ou KAGGLE_KEY não encontrados no .env"
    )

os.environ["KAGGLE_USERNAME"] = username
os.environ["KAGGLE_KEY"] = key

DATASET = "ejlok1/cremad"

print("Baixando dataset CREMA-D...")

subprocess.run([
    "kaggle",
    "datasets",
    "download",
    "-d",
    DATASET,
    "-p",
    "dataset"
])

print("Download concluído.")
