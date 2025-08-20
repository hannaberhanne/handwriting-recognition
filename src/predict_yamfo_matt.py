import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
import difflib
import matplotlib.pyplot as plt

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

    #img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    resized = cv2.resize(binary, (28, 28))

    normalized = resized.astype("float32") / 255.0
    
    # REMOVED THE EXTRA INVERSION
    tensor = torch.tensor(normalized).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1).item()
        # Map predicted class index (0-25) to corresponding ASCII uppercase letter (A-Z)

        answer = chr(pred + 65)  # EMNIST: 0 = 'A'
        return tensor, answer

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
    yamfo_dir = os.path.join(os.path.dirname(__file__), "../data/YAMFO")
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
        predicted = predict_letter(path)
        predictions.append(predicted)
        print(f"{filename} → {predicted}")

    name = "".join(predictions)
    print("\nFinal predicted name:", name)

    # Attempt to match predicted name with known towns
    best_match = match_closest_town(name, AHAFO_TOWNS)
    print("Closest town match:", best_match)

if __name__ == "__main__":
    test_image_path = os.path.join(os.path.dirname(__file__), "../data/YAMFO/M.png")
    image, pred  = predict_letter(test_image_path)
    print(f"The Answer is: {pred}")

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        img = image.cpu().squeeze()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"True: M \nPred: {pred}")
        axes[i].axis('off')
    plt.savefig("yamfo_predictions.png")
    print("Sample predictions saved as sample_predictions.png")
    
    
    #predict_yamfo()
