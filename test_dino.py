import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "models"))

from models.dinov3 import DINOv3Encoder

from PIL import Image
import torch

# Utwórz obiekt DINOv3Encoder
encoder = DINOv3Encoder()

# Zainicjalizuj model — podaj poprawne ścieżki
encoder.setup(
    dino_size="b",
    repo_dir="dinov3",  # lokalny folder z DINOv3
    dino_ckpt=None#"model.safetensors"  # plik wag
)

# Wczytaj przykładowy obraz
img = Image.new("RGB", (256, 256), color=(255, 0, 0))  # czerwony prostokąt
x = encoder.transform(img).unsqueeze(0)  #  transform z klasy dinov3encoder, unsqueeze dodaje wymiar batcha na początek tensora

# Wylicz embedding
with torch.no_grad():
    emb = encoder.model(x)

print("Kształt wyjścia:", emb.shape)
print("emb:", emb)
