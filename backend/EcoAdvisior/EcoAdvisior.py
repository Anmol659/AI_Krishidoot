# =============================================================================
# EcoAdvisor: An Integrated AI System for Agricultural Advisory
# =============================================================================
import os
import json
import sys
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta

# Core Libraries
import requests
import requests_cache
import pandas as pd
from dotenv import load_dotenv

# ML & Data Processing Libraries
import joblib
import torch
from torchvision import transforms, models
from PIL import Image
from collections import OrderedDict

# =============================================================================
# 0) SETUP & CONFIGURATION
# =============================================================================
# --- Basic Setup ---
requests_cache.install_cache('api_cache', backend='memory', expire_after=900)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
load_dotenv()

# --- API Key Configuration ---
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

# --- Gemini Model Configuration ---
USE_GEMINI = False
genai = None
if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        USE_GEMINI = True
        print(" Gemini API Key configured successfully.")
    except Exception:
        print(" Gemini disabled: configuration failed.")
else:
    print(" Gemini disabled: GOOGLE_API_KEY missing.")

# =============================================================================
# 0.5) LOAD ALL ML MODELS AT STARTUP
# =============================================================================

# --- A) PestPredict (Image Diagnosis Model) ---
PESTPREDICT_MODEL_PATH = r"C:/Users/anmol/OneDrive/Desktop/AI_Krishidoot/backend/models/PestPredict/best_model.pth"
PESTPREDICT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PESTPREDICT_CLASS_NAMES = [
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
PESTPREDICT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
PESTPREDICT_MODEL = None
try:
    model_instance = models.efficientnet_v2_m(weights=None)
    model_instance.classifier[1] = torch.nn.Linear(model_instance.classifier[1].in_features, len(PESTPREDICT_CLASS_NAMES))
    checkpoint = torch.load(PESTPREDICT_MODEL_PATH, map_location=PESTPREDICT_DEVICE)
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        new_state_dict[k.replace("module.", "")] = v
    model_instance.load_state_dict(new_state_dict, strict=False)
    model_instance = model_instance.to(PESTPREDICT_DEVICE)
    model_instance.eval()
    PESTPREDICT_MODEL = model_instance
    print(" PestPredict image diagnosis model loaded successfully.")
except Exception as e:
    print(f" Warning: Could not load PestPredict model: {e}")

# --- B) ProfitPilot (Price Prediction Model) ---
PROFITPILOT_MODEL_PATH = r"C:/Users/anmol/OneDrive/Desktop/AI_Krishidoot/backend/models/ProfitPilot/punjab_crop_price_model.pkl"
PROFITPILOT_ENCODER_PATH = r"C:/Users/anmol/OneDrive/Desktop/AI_Krishidoot/backend/models/ProfitPilot/label_encoders.pkl"
PROFIT_MODEL = None
PROFIT_ENCODERS = None
try:
    PROFIT_MODEL = joblib.load(PROFITPILOT_MODEL_PATH)
    PROFIT_ENCODERS = joblib.load(PROFITPILOT_ENCODER_PATH)
    print(" ProfitPilot ML model and encoders loaded successfully.")
except Exception as e:
    print(f" Warning: Could not load ProfitPilot model: {e}")

# --- C) FertilizerAdviser (Recommendation Model) ---
FERTILIZER_MODEL_PATH = r"C:/Users/anmol/OneDrive/Desktop/AI_Krishidoot/backend/models/FertilizerAdviser/fertilizer_model.pkl"
FERTILIZER_ENCODER_PATH = r"C:/Users/anmol/OneDrive/Desktop/AI_Krishidoot/backend/models/FertilizerAdviser/label_encoders.pkl"
FERTILIZER_MODEL = None
FERTILIZER_ENCODERS = None
try:
    FERTILIZER_MODEL = joblib.load(FERTILIZER_MODEL_PATH)
    FERTILIZER_ENCODERS = joblib.load(FERTILIZER_ENCODER_PATH)
    print(" FertilizerAdviser ML model and encoders loaded successfully.")
except Exception as e:
    print(f" Warning: Could not load FertilizerAdviser model: {e}")

# =============================================================================
# 1) UTILITY FUNCTIONS
# =============================================================================

CITY_COORDS = {
    "Amritsar": (31.63, 74.87), "Ludhiana": (30.90, 75.86), "Jalandhar": (31.33, 75.58),
    "Patiala": (30.34, 76.38), "Bathinda": (30.21, 74.94), "Firozpur": (30.92, 74.60)
}

def gemini_parser(query: str) -> Dict[str, Any]:
    """Uses Gemini to extract structured parameters from a user query."""
    print(f"[NLU-PARSER]: Analyzing query: '{query}'")
    base_params = {
        "query": query, "intent": "get_current_info", "location": "Ludhiana", "crop": "Wheat",
        "area_acres": 1.0, "timeframe": "next month", "soil_type": "Loamy",
        "nitrogen": 40, "phosphorous": 50, "potassium": 50, "temperature": 25,
        "humidity": 60, "soil_moisture": 30
    }
    if not USE_GEMINI: return base_params
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
    Analyze the farmer's query to extract structured information into a clean JSON object.
    - intent: Can be 'get_current_info', 'get_pest_risk', 'get_future_profit_forecast', 'get_fertilizer_advice', or 'diagnose_from_image'.
    - crop: Normalized crop name, default to 'Wheat'.
    - location: Normalized location name, default to 'Ludhiana'.
    - area_acres: Number of acres mentioned, default to 1.0.
    - timeframe: Timeframe for forecast (e.g., 'December'), default to 'next month'.
    - soil_type: Type of soil mentioned (e.g., Loamy, Sandy, Clayey), default to 'Loamy'.
    - Extract nutrient values (nitrogen, phosphorous, potassium), temperature, humidity, and soil_moisture if mentioned.
    
    Query: "{query}"
    JSON Output:
    """
    try:
        response = model.generate_content(prompt)
        json_str = response.text.strip().replace("```json", "").replace("```", "").strip()
        parsed_data = json.loads(json_str)
        # Merge parsed data with defaults to ensure all keys exist
        final_params = {**base_params, **parsed_data}
        final_params['query'] = query
        return final_params
    except Exception as e:
        print(f"      [NLU Error]: Could not parse query with Gemini: {e}")
        return base_params

def parse_timeframe_to_date(timeframe: str) -> str:
    """Converts natural language timeframe to a 'dd/mm/yyyy' string for the model."""
    today = datetime.now()
    timeframe = timeframe.lower()
    if "next week" in timeframe: future_date = today + timedelta(weeks=1)
    elif "next month" in timeframe: future_date = (today.replace(day=1) + timedelta(days=32)).replace(day=15)
    elif "december" in timeframe: future_date = datetime(today.year, 12, 15)
    elif "january" in timeframe: future_date = datetime(today.year + 1, 1, 15)
    else: future_date = (today.replace(day=1) + timedelta(days=32)).replace(day=15)
    return future_date.strftime("%d/%m/%Y")

# =============================================================================
# 2) SPECIALIST AGENT FUNCTIONS
# =============================================================================

def get_weather(location: str) -> Dict[str, Any]:
    """ClimaScout: Fetches current weather data."""
    if not OPENWEATHER_API_KEY: return {"error": "OPENWEATHER_API_KEY is not configured."}
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": f"{location},IN", "appid": OPENWEATHER_API_KEY, "units": "metric"}
    try:
        r = requests.get(url, params=params, timeout=10); r.raise_for_status(); data = r.json()
        return {"location": location.title(), "description": data["weather"][0]["description"].title(), "temperature_c": data["main"]["temp"], "humidity_pct": data["main"]["humidity"]}
    except Exception as e: return {"error": f"Weather API error: {e}"}

def get_market(crop: str, state: str = "Punjab") -> Dict[str, Any]:
    """MarketPulse: Fetches and parses market prices from the Punjab backend."""
    url = "https://backend-sewa.onrender.com/price"
    params = {"state": state.lower(), "commodity": crop.lower()}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()

        # Find the first real data row (it has 9 elements)
        first_valid_record = None
        for item in data:
            if isinstance(item, list) and len(item) == 9:
                first_valid_record = item
                break # Stop after finding the first one

        if not first_valid_record:
            return {"error": f"No valid mandi data found for {crop} in {state}."}

        # Extract data by position (index) since it's a list
        # Example row: ['Mohali', 'Kurali', 'Wheat', 'Other', '₹ 2,650', '₹ 2,620', '₹ 2,600', '29 Aug 2025', ...]
        market_name = first_valid_record[1]
        modal_price = first_valid_record[6].replace('₹ ', '').strip() # Clean the price string
        price_date = first_valid_record[7]

        return {
            "commodity": crop.title(),
            "market": market_name,
            "modal_price_inr_per_quintal": modal_price,
            "date": price_date
        }
    except Exception as e:
        return {"error": f"Market API error: {e}"}

def get_pest_risk(weather_data: Dict[str, Any], crop: str) -> Dict[str, Any]:
    """PestPredict (Risk): Forecasts pest risk based on weather."""
    if "error" in weather_data: return {"error": "Cannot compute pest risk due to unavailable weather data."}
    temp = float(weather_data.get("temperature_c", 0)); hum = float(weather_data.get("humidity_pct", 0))
    risk, notes = "Low", [f"General conditions for {crop} appear stable."]
    crop_lower = crop.lower()
    if crop_lower == "wheat":
        if temp > 20 and hum > 85: risk, notes = "High", ["Warm, highly humid conditions increase Rust fungus risk."]
    elif crop_lower == "rice":
        if temp > 25 and hum > 90: risk, notes = "High", ["Hot, very humid weather is ideal for Bacterial and Sheath Blight."]
    return {"crop": crop.title(), "risk": risk, "notes": notes}

def diagnose_pest_from_image(image_path: str) -> Dict[str, Any]:
    """PestPredict (Diagnosis): Uses the ML model to identify disease from an image."""
    if not PESTPREDICT_MODEL: return {"error": "PestPredict model not loaded."}
    if not image_path: return {"error": "No image provided for diagnosis."}
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = PESTPREDICT_TRANSFORM(image).unsqueeze(0).to(PESTPREDICT_DEVICE)
        with torch.no_grad():
            output = PESTPREDICT_MODEL(input_tensor)
            _, pred = torch.max(output, 1)
            predicted_class = PESTPREDICT_CLASS_NAMES[pred.item()].replace("__", " ").replace("_", " ")
        return {"prediction": predicted_class, "status": "success"}
    except Exception as e: return {"error": str(e), "status": "failed"}

def get_price_and_profit_forecast(crop: str, location: str, area_acres: float, timeframe: str) -> Dict[str, Any]:
    """ProfitPilot: Uses a RandomForest ML model to predict future prices and estimate revenue."""
    if not PROFIT_MODEL: return {"error": "ProfitPilot model not loaded."}
    try:
        prediction_date_str = parse_timeframe_to_date(timeframe); date_obj = datetime.strptime(prediction_date_str, "%d/%m/%Y")
        district_enc = PROFIT_ENCODERS["District"].transform([location])[0]
        market_enc = PROFIT_ENCODERS["Market"].transform([location])[0]
        commodity_enc = PROFIT_ENCODERS["Commodity"].transform([crop])[0]
        variety_enc = PROFIT_ENCODERS["Variety"].transform(["Other"])[0]
        features = pd.DataFrame([[district_enc, market_enc, commodity_enc, variety_enc, 0, 0, date_obj.day, date_obj.month, date_obj.year]], columns=["District", "Market", "Commodity", "Variety", "Minimum Price (INR)", "Maximum Price (INR)", "day", "month", "year"])
        predicted_price = PROFIT_MODEL.predict(features)[0]
        
        yields = {"wheat": 20.0, "rice": 24.0}; avg_yield = yields.get(crop.lower(), 15.0)
        est_revenue = avg_yield * area_acres * predicted_price
        loan_advice = "Based on your estimated revenue, you appear to be a good candidate for a Kisan Credit Card (KCC) loan."
        return {"forecast_for_date": prediction_date_str, "predicted_price_inr_per_quintal": round(predicted_price, 2), "estimated_revenue_inr": round(est_revenue, 2), "area_acres": area_acres, "loan_advice": loan_advice}
    except KeyError as e: return {"error": f"Cannot make prediction. The value '{str(e)}' is unknown to the model."}
    except Exception as e: return {"error": f"An error occurred during profit forecasting: {e}"}

def get_fertilizer_recommendation(params: Dict[str, Any]) -> Dict[str, Any]:
    """FertilizerAdviser: Uses an ML model to recommend fertilizer."""
    if not FERTILIZER_MODEL: return {"error": "FertilizerAdviser model not loaded."}
    try:
        soil_enc = FERTILIZER_ENCODERS['Soil Type'].transform([params['soil_type']])[0]
        crop_enc = FERTILIZER_ENCODERS['Crop Type'].transform([params['crop']])[0]
        
        features = pd.DataFrame([[
            params['temperature'], params['humidity'], params['soil_moisture'],
            soil_enc, crop_enc, params['nitrogen'], params['phosphorous'], params['potassium']
        ]], columns=['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Phosphorous', 'Potassium'])
        
        prediction = FERTILIZER_MODEL.predict(features)[0]
        return {"recommendation": prediction}
    except KeyError as e: return {"error": f"Cannot make recommendation. The value '{str(e)}' is unknown to the model."}
    except Exception as e: return {"error": f"An error occurred during fertilizer recommendation: {e}"}

# =============================================================================
# 3) ROUTER
# =============================================================================
def local_router(params: Dict[str, Any]) -> List[str]:
    """Uses the intent and keywords from the NLU to select agents."""
    q = params.get("query", "").lower()
    intent = params.get("intent", "get_current_info")
    models = set()

    if intent == 'diagnose_from_image' or any(k in q for k in ["photo", "image", "sick plant"]):
        models.add("PestDiagnostician"); return list(models)
    if intent == 'get_fertilizer_advice' or any(k in q for k in ["fertilizer", "khad", "खाद", "ખાતર"]):
        models.add("FertilizerAdviser"); return list(models)
    if intent == 'get_future_profit_forecast' or any(k in q for k in ["profit", "income", "मुनाफा", "નફો"]):
        models.add("ProfitPilot")
    if intent == 'get_pest_risk' or "risk" in q:
        models.add("PestPredictRisk")
    
    if any(k in q for k in ["price", "market", "bhav", "भाव", "ભાવ"]): models.add("MarketPulse")
    if any(k in q for k in ["weather", "rain", "मौसम", "હવામાન"]): models.add("ClimaScout")
    
    if not models: return ["ClimaScout", "MarketPulse", "PestPredictRisk"] # Default for greetings
    return list(models)

# =============================================================================
# 4) EXECUTION ORCHESTRATOR
# =============================================================================
def execute_models(models_to_call: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
    """Calls the selected specialist agents and collects their data."""
    print(f"\n[EXECUTOR]: Calling: {models_to_call}")
    collected: Dict[str, Any] = {}
    location = params.get("location", "Ludhiana"); crop = params.get("crop", "Wheat")
    
    # Dependency management
    if "PestPredictRisk" in models_to_call and "ClimaScout" not in models_to_call: models_to_call.insert(0, "ClimaScout")
    
    for name in models_to_call:
        if name == "ClimaScout": collected["weather"] = get_weather(location)
        elif name == "MarketPulse": collected["market"] = get_market(crop)
        elif name == "PestPredictRisk": collected["pest_risk"] = get_pest_risk(collected.get("weather", {}), crop)
        elif name == "PestDiagnostician": collected["diagnosis"] = diagnose_pest_from_image(params.get("image_path"))
        elif name == "ProfitPilot": collected["forecast"] = get_price_and_profit_forecast(crop, location, float(params.get("area_acres", 1.0)), params.get("timeframe", "next month"))
        elif name == "FertilizerAdviser": collected["fertilizer"] = get_fertilizer_recommendation(params)
    return collected

# =============================================================================
# 5) SYNTHESIZER
# =============================================================================
def gemini_synthesizer(query: str, context_data: Dict[str, Any], lang: str) -> str:
    """Uses Gemini to generate a final, natural language response."""
    if not USE_GEMINI: return "Gemini is not configured. Raw data: " + json.dumps(context_data)
    print(f"\n[GEMINI-SYNTHESIZER]: Generating final response in {lang}...")
    context_str = json.dumps(context_data, indent=2, ensure_ascii=False)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
    You are EcoAdvisor, an expert AI assistant for Indian farmers. Your response MUST be in {lang}.
    Use the structured data below to provide a concise, actionable, and friendly advisory. Do not mention internal model names like 'ProfitPilot'.
    - If there's an "error" key, state that the information is currently unavailable.
    - For market price, do not mention the date.
    - Structure the answer with clear headings (like Weather, Market Price) and short bullet points. Start with a friendly greeting in {lang}.

    Farmer's Query: "{query}"
    Structured Data:
    {context_str}

    Your expert response in {lang}:
    """
    try:
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        print(f"      [Synthesizer Error] {e}")
        return "Sorry, there was an error generating the response. Raw data: " + json.dumps(context_data)

# =============================================================================
# 6) MAIN HANDLER
# =============================================================================
def get_ai_response(query_text: str, lang: str = "English", image_path: Optional[str] = None) -> str:
    """Main function to process a user query and return an AI-generated response."""
    print("="*60 + f"\n🚀 New Job Started for Query: '{query_text}'")
    
    params = gemini_parser(query_text)
    if image_path: params["image_path"] = image_path
    print(f"[NLU PARAMS] {params}")
    
    models_needed = local_router(params)
    collected_data = execute_models(models_needed, params)
    
    if not collected_data: return "Sorry, I was unable to gather any data for your query."
    
    final_answer = gemini_synthesizer(query_text, collected_data, lang)
    
    print("\n Job Complete." + "\n" + "="*60 + "\n")
    return final_answer

# =============================================================================
# 7) QUICK DEMOS
# =============================================================================
if __name__ == "__main__":
    
    # --- Demo 1: General Info Query ---
    print("--- DEMO 1: General Info ---")
    answer1 = get_ai_response("what is the weather and wheat price in Amritsar?")
    print("\n--- FINAL ANSWER 1 ---\n", answer1)

    # --- Demo 2: Profit Forecast Query ---
    print("\n--- DEMO 2: Profit Forecast ---")
    answer2 = get_ai_response("What is the profit forecast for 5 acres of rice in Patiala for next month?")
    print("\n--- FINAL ANSWER 2 ---\n", answer2)

    # --- Demo 3: Fertilizer Advice Query ---
    print("\n--- DEMO 3: Fertilizer Advice ---")
    answer3 = get_ai_response("Suggest a fertilizer for wheat in loamy soil. Nitrogen is 50, phosphorous 40.")
    print("\n--- FINAL ANSWER 3 ---\n", answer3)
    
    # --- Demo 4: Image Diagnosis (Simulated) ---
    print("\n--- DEMO 4: Image Diagnosis (Simulated) ---")
    # In a real app, 'image.jpg' would be a path from a file upload.
    # Create a dummy image file for the test to run without error.
    try:
        dummy_image = Image.new('RGB', (100, 100), color = 'red')
        dummy_image_path = "dummy_image.jpg"
        dummy_image.save(dummy_image_path)
        
        answer4 = get_ai_response("what is wrong with my plant photo", image_path=dummy_image_path)
        print("\n--- FINAL ANSWER 4 ---\n", answer4)
        
        os.remove(dummy_image_path) # Clean up the dummy file
    except Exception as e:
        print(f"Could not run image demo: {e}")