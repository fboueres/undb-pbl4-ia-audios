from load_dataset import load_audio_metadata
from preprocess import extract_mfcc

df = load_audio_metadata()

sample = df.iloc[0]

mfcc = extract_mfcc(sample["file_path"])

print("Arquivo:")
print(sample["file_name"])

print()
print("Shape do MFCC:")
print(mfcc.shape)
