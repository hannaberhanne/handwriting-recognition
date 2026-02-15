import argparse
import os

from predict_yamfo_matt import AHAFO_TOWNS, match_closest_town, predict_letter

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def predict_from_folder(folder_name: str):
    folder_path = os.path.join(BASE_DIR, "data", folder_name)
    if not os.path.isdir(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    images = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith(".png")]
    )
    if not images:
        print(f"No PNG images found in {folder_name} folder.")
        return

    predictions = []
    for filename in images:
        path = os.path.join(folder_path, filename)
        _, letter = predict_letter(path)
        predictions.append(letter)
        print(f"{filename} -> {letter}")

    predicted_name = "".join(predictions)
    best_match = match_closest_town(predicted_name, AHAFO_TOWNS)

    print("")
    print(f"Folder: {folder_name}")
    print(f"Predicted name: {predicted_name}")
    print(f"Closest town match: {best_match}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict a town name from letter PNGs in data/<folder>."
    )
    parser.add_argument("--folder", default="YAMFO", help="Folder under data/ (default: YAMFO)")
    args = parser.parse_args()
    predict_from_folder(args.folder)


if __name__ == "__main__":
    main()
