import torch
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from collections import OrderedDict

# === Paths ===
image_path = r"C:/Users/anmol/Downloads/Leaf rust_wheat_2.jpg"
model_path = r"C:/Users/anmol/OneDrive/Desktop/AI_Krishidoot/backend/models/PestPredict/best_model.pth"

# === Device ===
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Classes ===
class_names = [
    'Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
    'Blueberry__healthy', 'Cherry(including_sour)Powdery_mildew', 'Cherry(including_sour)_healthy',
    'Corn_(maize)Cercospora_leaf_spot_Gray_leaf_spot', 'Corn(maize)Common_rust', 'Corn_(maize)_Northern_Leaf_Blight',
    'Corn_(maize)healthy', 'Grape_Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)',
    'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot', 'Peach__healthy',
    'Pepper_bell__Bacterial_spot', 'Pepper_bell_healthy', 'Potato_Early_blight', 'Potato__Late_blight',
    'Potato__healthy', 'Raspberry_healthy', 'Rice_Bacterial_leaf_blight', 'RiceBrown_spot', 'Rice_Hispa',
    'Rice_Leaf_blast', 'RiceLeaf_scald', 'RiceNarrow_brown_leaf_spot', 'RiceNeck_blast', 'Rice_Sheath_blight',
    'Rice_healthy', 'Soybean_healthy', 'Squash_Powdery_mildew', 'Strawberry_Leaf_scorch', 'Strawberry__healthy',
    'Tomato__Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato__Leaf_Mold',
    'Tomato__Septoria_leaf_spot', 'Tomato_Spider_mites_Two-spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus', 'Tomato__healthy',
    'Wheat_Leaf_rust', 'Wheathealthy', 'Wheat_septoria'
]

# === Model Setup ===
# ⚠️ Change this line if needed (try efficientnet_b0, b2, v2_s, v2_m, etc.)
model = models.efficientnet_v2_m(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_names))
model = model.to(device)

# === Load checkpoint ===
try:
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:  
        # Case 1: checkpoint with 'state_dict'
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):  
        # Case 2: plain state_dict (maybe with 'module.' prefix)
        state_dict = checkpoint
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")  # remove 'module.' if present
            new_state_dict[name] = v
        state_dict = new_state_dict
        model.load_state_dict(state_dict, strict=False)
    else:
        # Case 3: full model was saved
        model = checkpoint

except Exception as e:
    print(" Error loading checkpoint:", e)
    exit()

model.eval()

# === Image Transform ===
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# === Inference ===
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)
    _, pred = torch.max(output, 1)
    predicted_class = class_names[pred.item()]

# Draw label on image
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()
draw.text((10, 10), f"Predicted: {predicted_class}", fill="red", font=font)

# Show image
plt.figure(figsize=(8,8))
plt.imshow(image)
plt.axis("off")
plt.show()

print(f" Prediction: {predicted_class}")
