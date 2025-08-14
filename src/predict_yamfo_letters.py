import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
import difflib

DEBUG_SAVE = False
DEBUG_DIR = os.path.join(os.path.dirname(__file__), "../debug_preprocessed")

# ---- Centering helper: crop by bbox, fit into 28x28 with aspect ----
def _center_and_fit(img_gray: np.ndarray, pad: int = 2) -> np.ndarray:
    g = cv2.GaussianBlur(img_gray, (3, 3), 0)
    _, binary = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(binary > 0)
    if len(xs) == 0 or len(ys) == 0:
        return cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, img_gray.shape[1]-1)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, img_gray.shape[0]-1)
    crop = img_gray[y0:y1+1, x0:x1+1]
    h, w = crop.shape
    if h == 0 or w == 0:
        return cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
    if h > w:
        new_h, new_w = 20, max(1, int(round(20 * w / h)))
    else:
        new_h, new_w = max(1, int(round(20 * h / w))), 20
    small = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_off = (28 - new_h) // 2
    x_off = (28 - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = small
    return canvas

# ---- Build a single 28x28 variant according to orientation/centering ----
def _build_variant(img_gray: np.ndarray, rotate_code=None, center=False) -> np.ndarray:
    img = img_gray
    if rotate_code is not None:
        img = cv2.rotate(img, rotate_code)
    if center:
        img28 = _center_and_fit(img)
    else:
        img28 = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    x = img28.astype(np.float32) / 255.0
    x = 1.0 - x  # invert to match training (1 - x)
    return x

# ---- Optionally save debug images ----
def _maybe_save_debug(arr28: np.ndarray, out_path: str):
    if not DEBUG_SAVE:
        return
    os.makedirs(DEBUG_DIR, exist_ok=True)
    arr = (arr28 * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(out_path, arr)

# CNN model for classifying handwritten capital letters A-Z (26 classes)
class LetterCNN(nn.Module):
    def __init__(self, num_classes=26):  # Model outputs one of 26 capital letters
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LetterCNN().to(device)

# Path to the trained EMNIST model
model_path = os.path.join(os.path.dirname(__file__), "../emnist_cnn_corrected.pth")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Predict a single capital letter from a given image
def predict_letter(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read {img_path}")
        return "?"

    # Try multiple reasonable variants; pick the one with highest confidence
    variants = [
        (None, False, "norot_raw"),
        (cv2.ROTATE_90_CLOCKWISE, False, "rot_cw_raw"),
        (cv2.ROTATE_90_COUNTERCLOCKWISE, False, "rot_ccw_raw"),
        (None, True, "norot_center"),
        (cv2.ROTATE_90_CLOCKWISE, True, "rot_cw_center"),
        (cv2.ROTATE_90_COUNTERCLOCKWISE, True, "rot_ccw_center"),
    ]

    best_letter = "?"
    best_conf = -1.0
    best_tag = ""

    with torch.no_grad():
        for rot, center, tag in variants:
            norm = _build_variant(img, rotate_code=rot, center=center)
            tensor = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            output = model(tensor)
            probs = F.softmax(output, dim=1)
            conf, pred = torch.max(probs, dim=1)
            conf_val = float(conf.item())
            pred_val = int(pred.item())
            letter = chr(pred_val + 65)

            if DEBUG_SAVE:
                dbg_name = f"{os.path.basename(img_path)}__{tag}__{letter}_{conf_val:.2f}.png"
                _maybe_save_debug(norm, os.path.join(DEBUG_DIR, dbg_name))

            if conf_val > best_conf:
                best_conf = conf_val
                best_letter = letter
                best_tag = tag

    # Also print a short tag + confidence so you can see which variant won
    print(f"    chosen={best_tag}, conf={best_conf:.2f}")
    return best_letter

# List of known town names in Ahafo Region
AHAFO_TOWNS = [
    "AHYIAIYEM", "AMANFROM", "ANIWIAM", "ATANEATA", "BECHEM", "BEDIAKO",
    "EDASO", "GOASO", "KINAADAIKROM", "KUKUOM", "KWANTENE", "MFRERIKUM",
    "NEW SAWERESO", "NKASEIM", "NTOTROSU", "SUBINSU", "WURUMUMUSO", "YAMFO"
]

# Compare predicted name to known town names and return the closest match
def match_closest_town(predicted, town_list):
    match = difflib.get_close_matches(predicted, town_list, n=1, cutoff=0.6)
    return match[0] if match else "No close match found"

# Predict all letters from image files in a given folder and reconstruct the full name
def predict_from_folder(folder_name):
    folder_path = os.path.join(os.path.dirname(__file__), "../data", folder_name)
    if not os.path.isdir(folder_path):
        print("Folder not found:", folder_path)
        return

    images = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(".png")
    ])

    if not images:
        print(f"No PNG images found in {folder_name} folder.")
        return

    predictions = []
    for filename in images:
        path = os.path.join(folder_path, filename)
        predicted = predict_letter(path)
        predictions.append(predicted)
        print(f"{filename} → {predicted}")

    name = "".join(predictions)
    print(f"\nFinal predicted name in {folder_name}:", name)

    # Attempt to match predicted name with known towns
    best_match = match_closest_town(name, AHAFO_TOWNS)
    print("Closest town match:", best_match)
    print("-" * 40)

if __name__ == "__main__":
    for folder in ["YAMFO", "Bechem", "ATANEATA"]:
        predict_from_folder(folder)