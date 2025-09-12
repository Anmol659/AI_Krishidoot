import torch
from torchvision import transforms, models
from PIL import Image
from collections import OrderedDict

# === Device ===
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Classes ===
CLASS_NAMES = [
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

# === Image Transform ===
TRANSFORM = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# === Load Model Once ===
def load_model(model_path="best_model.pth"):
    model = models.efficientnet_v2_m(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        # Handle DataParallel prefixes
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            new_state_dict[k.replace("module.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
    else:
        model = checkpoint

    model = model.to(device)
    model.eval()
    return model

MODEL = load_model()  # load once when module is imported

# === Inference Function ===
def get_pest_prediction(image_path: str) -> dict:
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = TRANSFORM(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = MODEL(input_tensor)
            _, pred = torch.max(output, 1)
            predicted_class = CLASS_NAMES[pred.item()]

        return {"prediction": predicted_class, "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}
