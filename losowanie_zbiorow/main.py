import os
import shutil
import random
import numpy as np
from pathlib import Path


def collect_all_images(root):
    """Zbiera wszystkie obrazy z podfolderów 0–9."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    files = []

    for sub in os.listdir(root):
        sub_path = os.path.join(root, sub)
        if os.path.isdir(sub_path):
            for f in os.listdir(sub_path):
                if Path(f).suffix.lower() in exts:
                    files.append(os.path.join(sub_path, f))

    return files


def resolve_name_collision(dst_folder, filename):
    """Zmień nazwę, jeśli plik już istnieje."""
    base = Path(filename).stem
    ext = Path(filename).suffix
    counter = 1
    new_name = filename

    while os.path.exists(os.path.join(dst_folder, new_name)):
        new_name = f"{base}_{counter}{ext}"
        counter += 1

    return new_name


def sample_indices_normal(n, k):
    """
    Losuje k indeksów z przedziału 0...n-1 wg rozkładu normalnego.
    Środek ustawiony na n/2, odchylenie: n/6 (bezpiecznie dla 99.7% danych).
    """
    mean = n / 2
    std = n / 6

    indices = np.random.normal(loc=mean, scale=std, size=k)
    indices = np.clip(indices, 0, n - 1)  # żeby nie wyjść poza zakres
    return indices.astype(int)
def sample_indices_normal_no_replacement(n, k):
    """
    Losuje k unikalnych indeksów (bez powtórzeń) z wagami wynikającymi
    z rozkładu normalnego.
    """
    x = np.arange(n)

    # Parametry rozkładu normalnego
    mean = n / 2
    std = n / 6

    # Gęstość PDF dla każdego indeksu
    pdf = np.exp(-0.5 * ((x - mean) / std) ** 2)
    pdf /= pdf.sum()  # normalizacja do sumy = 1

    # Losowanie bez powtórzeń
    indices = np.random.choice(n, size=k, replace=False, p=pdf)
    return indices



def main(input_root, output_root, sample_size=10000):
    print("⏳ Zbieram wszystkie obrazy...")
    all_images = collect_all_images(input_root)
    n = len(all_images)

    print(f"📁 Znaleziono {n} obrazów.")

    if n < sample_size:
        raise ValueError(f"Za mało obrazów ({n}) aby wylosować {sample_size}.")

    print("🎲 Losuję obrazy wg rozkładu normalnego...")
    indices = sample_indices_normal_no_replacement(n, sample_size)
    selected = [all_images[i] for i in indices]

    os.makedirs(output_root, exist_ok=True)

    print("📦 Kopiuję obrazy do folderu output...")
    for src in selected:
        filename = os.path.basename(src)
        filename = resolve_name_collision(output_root, filename)
        dst = os.path.join(output_root, filename)
        shutil.copy2(src, dst)

    print("✅ Gotowe! Obrazy znajdują się w:", output_root)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Losowanie 10k obrazów wg rozkładu normalnego.")
    parser.add_argument("input_root", help="Ścieżka do katalogu z folderami 0–9")
    parser.add_argument("output_root", help="Ścieżka do katalogu na wynik")
    parser.add_argument("--sample_size", type=int, default=10000, help="Ile obrazów wylosować (domyślnie 10k)")

    args = parser.parse_args()
    main(args.input_root, args.output_root, args.sample_size)
