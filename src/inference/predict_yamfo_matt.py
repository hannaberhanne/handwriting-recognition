import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
import difflib
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Corrected model (26 classes)
class LetterCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(7 * 7 * 64, 128)  # Fixed dimensions
        self.fc2 = nn.Linear(128, 26)  # 26 classes A-Z

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 28x28 -> 28x28
        x = F.max_pool2d(x, 2)     # 14x14
        x = F.relu(self.conv2(x))  # 14x14 -> 14x14
        x = F.max_pool2d(x, 2)     # 7x7
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LetterCNN().to(device)

# Path to the trained EMNIST model
model_path = os.path.join(BASE_DIR, "emnist_cnn_corrected.pth")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- Inference config ---
PRESENTATION_WHITE_BG = True   # keep images as black-on-white for your visuals
MODEL_WHITE_ON_BLACK = True    # set True if your trained model expects white-on-black tensors

# --- Inference helpers (no model changes) ---
import torchvision.transforms.functional as TF
import math

def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-8)

# Try several thresholding configs; pick the one with highest top-1 prob after TTA
BIN_CONFIGS = [
    ("adaptive_31_5", {"blockSize":31, "C":5}),
    ("adaptive_25_3", {"blockSize":25, "C":3}),
    ("adaptive_41_7", {"blockSize":41, "C":7}),
    ("otsu", {}),
]

def binarize_with_config(img_gray: np.ndarray, name: str, params: dict) -> np.ndarray:
    if name == "otsu":
        _, th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th
    # adaptive gaussian with CLAHE boost
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(img_gray)
    bs = params.get("blockSize", 31)
    C = params.get("C", 5)
    if bs % 2 == 0:
        bs += 1
    th = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, bs, C)
    return th


def binarize_adaptive(img_gray: np.ndarray) -> np.ndarray:
    """Boost contrast and binarize uneven/faded ink."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(img_gray)
    th = cv2.adaptiveThreshold(eq, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY,
                               31, 5)
    return th


def center_pad_28x28(binary: np.ndarray) -> np.ndarray:
    """MNIST-style centering on *black ink / white paper*."""
    # mask of black ink
    mask = (binary == 0).astype(np.uint8)
    if mask.sum() == 0:
        return np.full((28, 28), 255, np.uint8)  # pure white canvas

    x, y, w, h = cv2.boundingRect(mask)
    roi = binary[y:y+h, x:x+w]  # still black-on-white

    # keep aspect, fit inside 20x20
    if w >= h:
        new_w, new_h = 20, max(1, int(round(h * (20.0 / w))))
    else:
        new_h, new_w = 20, max(1, int(round(w * (20.0 / h))))
    roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    roi = cv2.medianBlur(roi, 3)

    canvas = np.full((28, 28), 255, np.uint8)  # white background
    x_off = (28 - new_w) // 2
    y_off = (28 - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = roi
    return canvas


def deskew_around_center(img28: np.ndarray) -> np.ndarray:
    """Tiny deskew to reduce slant; safe no-op if moments are degenerate."""
    m = cv2.moments(img28)
    if abs(m['mu02']) < 1e-2:
        return img28
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5*28*skew], [0, 1, 0]])
    return cv2.warpAffine(img28, M, (28, 28), flags=cv2.INTER_LINEAR)


def predict_with_tta(tensor_1x1x28x28: torch.Tensor, model: nn.Module, device: torch.device) -> torch.Tensor:
    """Return mean logits over light rotations + one flip (on CPU)."""
    logits = []
    with torch.no_grad():
        for deg in (-4, -2, 0, 2, 4):
            rotated = TF.rotate(tensor_1x1x28x28, deg)
            out = model(rotated.to(device))  # (1,26)
            logits.append(out.detach().cpu())
        # Optional single horizontal flip vote (kept to increase robustness)
        flipped = torch.flip(tensor_1x1x28x28, dims=[3])
        logits.append(model(flipped.to(device)).detach().cpu())
    mean_logits = torch.stack(logits, dim=0).mean(dim=0).squeeze(0)  # (26,)
    return mean_logits

def circularity_score(img28_bw: np.ndarray) -> float:
    """Estimate how circle-like the glyph is (1.0 ~ perfect circle)."""
    # black = ink
    mask = (img28_bw == 0).astype(np.uint8)
    if mask.sum() < 10:
        return 0.0
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    if peri <= 0:
        return 0.0
    return float((4.0 * np.pi * area) / (peri * peri))


# --- O / Q / D structural disambiguation ---
# All functions expect a 28x28 **black-ink-on-white** binary image (0=ink, 255=paper)

def count_holes_bw(img28_bw: np.ndarray) -> int:
    """Return number of holes (white regions fully enclosed by black ink)."""
    # Find contours with hierarchy to count inner holes
    mask = (img28_bw == 0).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    holes = 0
    for i, h in enumerate(hierarchy[0]):
        parent = h[3]
        # A contour with a parent is a hole
        if parent != -1:
            holes += 1
    return holes


def q_tail_score_bw(img28_bw: np.ndarray) -> float:
    """Heuristic: measure presence of a down-right spur typical for 'Q'.
    Returns a score in [0,1] (higher -> more Q-like)."""
    # Focus on lower-right quadrant
    roi = img28_bw[14:28, 14:28]
    # Edge map to highlight slender strokes
    edges = cv2.Canny(roi, 50, 150)
    # Hough segments at ~45°
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=5, minLineLength=3, maxLineGap=2)
    score = 0.0
    if lines is not None:
        for l in lines:
            x1,y1,x2,y2 = l[0]
            dx, dy = x2-x1, y2-y1
            angle = abs(np.degrees(np.arctan2(dy, dx)))
            if 20 <= angle <= 70:  # diagonal-ish
                score += 0.25  # accumulate small evidence
    # Additional density cue: extra ink near corner beyond circular ring
    # Build an approximate annulus by distance transform
    inv = 255 - img28_bw  # white ring on black
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    ring = (dist > 1.0).astype(np.uint8) * 255
    ring = cv2.morphologyEx(ring, cv2.MORPH_ERODE, np.ones((3,3), np.uint8))
    ink = (img28_bw == 0).astype(np.uint8)
    extras = (ink * (1 - (ring>0).astype(np.uint8)))[14:28,14:28]
    dens = extras.sum() / (14*14)  # normalized by area
    score += float(min(1.0, dens * 0.2))
    return float(min(1.0, score))


def override_oqd(letter_top: str, probs: np.ndarray, img28_bw: np.ndarray) -> str:
    """Rule-based override using holes and tail score to fix O/Q/D confusions."""
    letters = [chr(i+65) for i in range(26)]
    if letter_top not in ("O", "Q", "D"):
        return letter_top
    holes = count_holes_bw(img28_bw)
    if holes == 0:
        return "D"  # cannot be O/Q without a hole
    # holes >=1 -> likely O or Q
    tail = q_tail_score_bw(img28_bw)
    if tail >= 0.35:
        return "Q"
    else:
        return "O"


# Predict a single capital letter from a given image
def predict_letter(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read {img_path}")
        return None, "?"

    best = {"prob": -1.0, "letter": "?", "tensor": None}

    for name, params in BIN_CONFIGS:
        try:
            binary = binarize_with_config(img, name, params)
            canvas = center_pad_28x28(binary)
            canvas = deskew_around_center(canvas)
            # keep black-on-white for presentation
            disp_canvas = canvas

            # feed the model the polarity it was trained on
            model_input = (255 - canvas) if MODEL_WHITE_ON_BLACK else canvas
            norm = model_input.astype("float32") / 255.0
            tensor = torch.tensor(norm).unsqueeze(0).unsqueeze(0)

            mean_logits = predict_with_tta(tensor, model, device)  # (26,)
            probs = softmax_np(mean_logits.numpy())
            letters = [chr(i + 65) for i in range(26)]
            top_idx = int(np.argmax(probs))
            top_p = float(probs[top_idx])
            pred_letter = letters[top_idx]

            # Structural override for O/Q/D using holes + tail
            oqd_letter = override_oqd(pred_letter, probs, disp_canvas)
            if oqd_letter != pred_letter:
                # trust override if model is uncertain (<0.75) or if second-best is close (<0.12 margin)
                margin = top_p - float(np.partition(probs, -2)[-2])
                if top_p < 0.75 or margin < 0.12:
                    pred_letter = oqd_letter

            if top_p > best["prob"]:
                best.update({
                    "prob": top_p,
                    "letter": pred_letter,
                    "tensor": torch.tensor(disp_canvas.astype("float32")/255.0).unsqueeze(0).unsqueeze(0),
                })
        except Exception as e:
            # be permissive; move on to next config
            continue

    return best["tensor"], best["letter"]

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

# Predict all letters from image files in the YAMFO folder and reconstruct the full name
def predict_yamfo():
    yamfo_dir = os.path.join(BASE_DIR, "data", "YAMFO")
    if not os.path.isdir(yamfo_dir):
        print("Folder not found:", yamfo_dir)
        return

    images = sorted([
        f for f in os.listdir(yamfo_dir)
        if f.lower().endswith(".png")
    ])

    if not images:
        print("No PNG images found in YAMFO folder.")
        return

    predictions = []
    for filename in images:
        path = os.path.join(yamfo_dir, filename)
        _, letter = predict_letter(path)
        predictions.append(letter)
        print(f"{filename} → {letter}")

    name = "".join(predictions)
    print("\nFinal predicted name:", name)
    best_match = match_closest_town(name, AHAFO_TOWNS)
    print("Lexicon-corrected (closest match):", best_match)

if __name__ == "__main__":
    test_image_path = os.path.join(BASE_DIR, "data", "YAMFO", "O.png")
    image, pred  = predict_letter(test_image_path)

    # Recompute confidence for visualization if image tensor available
    conf = None
    if image is not None:
        mean_logits = predict_with_tta(image, model, device)
        probs = softmax_np(mean_logits.numpy())
        top_idx = int(np.argmax(probs))
        top_p = float(probs[top_idx])
        conf = top_p

    print(f"The Answer is: {pred}")

    if image is not None:
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        for i in range(5):
            img = image.cpu().squeeze()
            if conf is not None:
                axes[i].set_title(f"True: O \nPred: {pred}\nConf: {conf:.2f}")
            else:
                axes[i].set_title(f"True: O \nPred: {pred}")
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
        plt.savefig("yamfo_predictions.png")
        print("Sample predictions saved as yamfo_predictions.png")
    else:
        print("Skipping visualization because input image could not be read.")
    
    
    #predict_yamfo()
