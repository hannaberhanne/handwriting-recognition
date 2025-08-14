import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
import difflib

# CNN model for classifying handwritten capital letters A-Z (26 usable labels from EMNIST)
class LetterCNN(nn.Module):
    def __init__(self, num_classes=27):  # EMNIST 'letters' split uses 27 classes (1-26 for A-Z)
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(5 * 5 * 64, 128)
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

# Load trained CNN model from file
model_path = os.path.join(os.path.dirname(__file__), "../ind_study_emnist/models/emnist_cnn.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LetterCNN().to(device)

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

    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    resized = cv2.resize(binary, (28, 28))
    normalized = resized.astype("float32") / 255.0
    tensor = torch.tensor(normalized).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1).item()
        return chr(pred + 64) if pred > 0 else "?"  # Map class index to A-Z, skip 0

# List of known Ahafo Region towns
AHAFO_TOWNS = [
    "AHYIAIYEM", "AMANFROM", "ANIWIAM", "ATANEATA", "BECHEM", "BEDIAKO",
    "EDASO", "GOASO", "KINAADAIKROM", "KUKUOM", "KWANTENE", "MFRERIKUM",
    "NEW SAWERESO", "NKASEIM", "NTOTROSU", "SUBINSU", "WURUMUMUSO", "YAMFO"
]

# Match predicted name to known towns using fuzzy string matching
def match_closest_town(predicted, town_list):
    match = difflib.get_close_matches(predicted, town_list, n=1, cutoff=0.6)
    return match[0] if match else "No close match found"

# Predict all letters from the ATANEATA folder and reconstruct the full name
def predict_ataneata():
    folder = os.path.join(os.path.dirname(__file__), "../data/A-T-A-N-E-A-T-A")
    if not os.path.isdir(folder):
        print("Folder not found:", folder)
        return

    images = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(".png")
    ])

    if not images:
        print("No PNG images found in A-T-A-N-E-A-T-A folder.")
        return

    predictions = []
    for filename in images:
        path = os.path.join(folder, filename)
        predicted = predict_letter(path)
        predictions.append(predicted)
        print(f"{filename} → {predicted}")

    name = "".join(predictions)
    print("\nFinal predicted name:", name)
    best_match = match_closest_town(name, AHAFO_TOWNS)
    print("Closest town match:", best_match)

if __name__ == "__main__":
    predict_ataneata()
