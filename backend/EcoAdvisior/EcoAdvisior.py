import os
import re
import json
import sys
from typing import List, Dict, Any, Tuple, Optional

import requests
import requests_cache
from dotenv import load_dotenv

# =============================================================================
# 0) SETUP & CONFIGURATION
# =============================================================================
requests_cache.install_cache('api_cache', backend='memory', expire_after=900) # Cache for 15 mins
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
load_dotenv()

# Gemini and API Key Configuration
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
AGMARKNET_API_KEY = os.getenv("AGMARKNET_API_KEY", "")
USE_GEMINI = False
genai = None

if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        USE_GEMINI = True
        print("Gemini API Key configured successfully.")
    except Exception:
        print("Gemini disabled: configuration failed.")
else:
    print("Gemini disabled: GOOGLE_API_KEY missing.")

# =============================================================================
# 1) UTILITY FUNCTIONS (NLU Parser)
# =============================================================================
def gemini_parser(query: str) -> Dict[str, Any]:
    """Uses Gemini to extract structured parameters from a user query."""
    print(f"[NLU-PARSER]: Analyzing query: '{query}'")
    
    # Add the original query to the params for keyword searching later
    base_params = {"query": query, "intent": "get_current_info", "location": "Rajkot", "crop": "Cotton"}
    
    if not USE_GEMINI:
        return base_params
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
    Analyze the farmer's query to extract structured information into a clean JSON object.
    - intent: Can be 'get_current_info', 'get_pest_risk', 'get_future_profit_forecast'.
    - crop: Normalized crop name, default to 'Cotton'.
    - location: Normalized location name, default to 'Rajkot'.
    - area_acres: Number of acres mentioned, default to 1.0.
    - timeframe: Timeframe for forecast (e.g., 'December'), default to 'next month'.
    
    Query: "{query}"
    JSON Output:
    """
    try:
        response = model.generate_content(prompt)
        json_str = response.text.strip().replace("```json", "").replace("```", "").strip()
        parsed_data = json.loads(json_str)
        parsed_data['query'] = query # Ensure original query is always included
        return parsed_data
    except Exception as e:
        print(f"      [NLU Error]: Could not parse query with Gemini: {e}")
        return base_params

CITY_COORDS = {"Rajkot": (22.30, 70.80), "Jamnagar": (22.47, 70.06)}

def get_coords(location: str) -> Tuple[float, float]:
    return CITY_COORDS.get(location.title(), CITY_COORDS["Rajkot"])

# =============================================================================
# 2) SPECIALIST AGENT FUNCTIONS
# =============================================================================
def get_weather(location: str) -> Dict[str, Any]:
    if not OPENWEATHER_API_KEY: return {"error": "OPENWEATHER_API_KEY is not configured."}
    url = "https://api.openweathermap.org/data/2.5/weather"; params = {"q": f"{location},IN", "appid": OPENWEATHER_API_KEY, "units": "metric"}
    try:
        r = requests.get(url, params=params, timeout=10); r.raise_for_status(); data = r.json()
        return {"location": location.title(), "description": data["weather"][0]["description"].title(), "temperature_c": data["main"]["temp"], "humidity_pct": data["main"]["humidity"]}
    except Exception as e: return {"error": f"Weather API error: {e}"}

def get_market(crop: str, state: str = "Gujarat") -> Dict[str, Any]:
    if not AGMARKNET_API_KEY: return {"error": "AGMARKNET_API_KEY is not configured."}
    url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"; params = {"api-key": AGMARKNET_API_KEY, "format": "json", "limit": 5, "filters[commodity]": crop.title(), "filters[state]": state}
    try:
        r = requests.get(url, params=params, timeout=15); r.raise_for_status(); data = r.json(); recs = data.get("records", [])
        if not recs: return {"error": f"No recent mandi data found for {crop}."}
        rec = recs[0]
        return {"commodity": crop.title(), "market": rec.get("market"), "modal_price_inr_per_quintal": rec.get("modal_price"), "date": rec.get("arrival_date")}
    except Exception as e: return {"error": f"Agmarknet API error: {e}"}

def get_pest_risk(weather_data: Dict[str, Any], crop: str) -> Dict[str, Any]:
    if not weather_data or "error" in weather_data: return {"error": "Cannot compute pest risk due to unavailable weather data."}
    temp = float(weather_data.get("temperature_c", 0)); hum = float(weather_data.get("humidity_pct", 0))
    risk, notes = "Low", [f"General conditions for {crop} appear low-risk."]
    if crop.lower() == "cotton":
        if temp >= 32 and hum >= 60: risk, notes = "High", ["Hot & humid conditions are favorable for sucking pests."]
        elif temp >= 30 and hum >= 55: risk, notes = "Medium", ["Warm and humid conditions may encourage pest activity."]
    return {"crop": crop.title(), "risk": risk, "notes": notes}

# In backend/EcoAdvisior/EcoAdvisior.py

def get_profit_forecast(market_data: Dict[str, Any], crop: str, area_acres: float, timeframe: str) -> Dict[str, Any]:
    """
    ProfitPilot: A forward-looking strategist to forecast price, revenue, and give loan advice.
    """
    if not market_data or "error" in market_data:
        return {"error": "Cannot forecast profit without current market data."}
    
    current_price = float(market_data.get("modal_price_inr_per_quintal", 7000))
    future_price = current_price * 1.05  # Simulate a 5% price increase
    avg_yield_per_acre = 8.0  # More realistic for cotton
    total_yield = avg_yield_per_acre * area_acres
    estimated_revenue = total_yield * future_price
    
    # Add KCC loan advice based on the forecast
    loan_advice = (
        "Based on your estimated revenue, you appear to be a good candidate for a Kisan Credit Card (KCC) loan. "
        "It is recommended to visit your nearest nationalized bank with your land records to inquire about a credit limit."
    )
    
    # Return a complete dictionary, including the area and advice
    return {
        "forecast_for_timeframe": timeframe,
        "predicted_price_inr_per_quintal": round(future_price, 2),
        "estimated_revenue_inr": round(estimated_revenue, 2),
        "area_acres": area_acres,
        "loan_advice": loan_advice,
    }
# =============================================================================
# 3) ROUTER
# =============================================================================
def local_router(parsed_params: Dict[str, Any]) -> List[str]:
    """Uses the intent and keywords from the NLU parser to select agents."""
    intent = parsed_params.get("intent", "get_current_info")
    q = parsed_params.get("query", "").lower()
    models = set()

    if intent == 'get_future_profit_forecast':
        models.add("ProfitPilot")
        return list(models)
    
    if intent == 'get_pest_risk':
        models.add("PestPredict")
        return list(models)

    if any(k in q for k in ["price", "market", "bhav", "rate", "भाव", "ભાવ"]):
        models.add("MarketPulse")
    
    if any(k in q for k in ["weather", "rain", "temperature", "forecast", "मौसम", "હવામાન"]):
        models.add("ClimaScout")
    
    if any(k in q for k in ["irrigation", "water", "soil", "moisture", "सिंचाई", "પિયત"]):
        models.add("TerraMoist")

    if not models:
        return ["ClimaScout", "MarketPulse"] # Default for greetings
        
    return list(models)

# =============================================================================
# 4) EXECUTION ORCHESTRATOR
# =============================================================================
def execute_models(models_to_call: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
    print(f"\n[EXECUTOR]: Calling: {models_to_call}")
    collected: Dict[str, Any] = {}
    location = params.get("location", "Rajkot"); crop = params.get("crop", "Cotton")
    
    if "PestPredict" in models_to_call and "ClimaScout" not in models_to_call: models_to_call.insert(0, "ClimaScout")
    if "ProfitPilot" in models_to_call and "MarketPulse" not in models_to_call: models_to_call.insert(0, "MarketPulse")

    for name in models_to_call:
        if name == "ClimaScout": collected["weather"] = get_weather(location)
        elif name == "PestPredict": collected["pest"] = get_pest_risk(collected.get("weather", {}), crop)
        elif name == "MarketPulse": collected["market"] = get_market(crop)
        elif name == "ProfitPilot": collected["forecast"] = get_profit_forecast(collected.get("market", {}), crop, float(params.get("area_acres", 1.0)), params.get("timeframe", "next month"))
    return collected

# =============================================================================
# 5) SYNTHESIZER
# =============================================================================
# In backend/EcoAdvisior/EcoAdvisior.py

def local_synthesizer(query: str, data: Dict[str, Any]) -> str:
    """A user-friendly formatter for the final response if Gemini is unavailable."""
    parts = ["Namaste! Here is your AI-Krishidoot advisory:"]
    
    if "weather" in data and not data["weather"].get("error"):
        w = data["weather"]
        parts.append(f"\n- Weather in {w.get('location')}: {w.get('description')}, {w.get('temperature_c')}°C.")
    
    if "market" in data and not data["market"].get("error"):
        m = data["market"]
        parts.append(f"- Market Price for {m.get('commodity')}: ₹{m.get('modal_price_inr_per_quintal')} per quintal at {m.get('market')} market.")
    
    if "pest" in data and not data["pest"].get("error"):
        p = data["pest"]
        notes = " ".join(p.get("notes", []))
        parts.append(f"- Pest Risk for {p.get('crop')}: {p.get('risk')}. {notes}")
    
    if "forecast" in data and not data["forecast"].get("error"):
        f = data["forecast"]
        parts.append(f"\n- Profit Forecast for {f.get('area_acres')} acres: Estimated revenue of ₹{f.get('estimated_revenue_inr'):,.0f}.")
        if f.get("loan_advice"):
            parts.append(f"- Loan Advice: {f.get('loan_advice')}")
            
    return "\n".join(parts)

def gemini_synthesizer(query: str, context_data: Dict[str, Any], lang: str) -> str:
    if not USE_GEMINI: return local_synthesizer(query, context_data)
    print(f"\n[GEMINI-SYNTHESIZER]: Generating final response in {lang}...")
    model = genai.GenerativeModel("gemini-1.5-flash")
    context_str = json.dumps(context_data, indent=2, ensure_ascii=False)
    prompt = f"""
    You are EcoAdvisor, an AI assistant for Indian farmers. Use the structured data below to produce a concise, actionable reply in simple language. Do not mention internal model names.
    IMPORTANT: You MUST write your entire response in the following language: {lang}.
    If any data has an "error" key, state that information is unavailable and give a safe recommendation.
    Instruction: When reporting the market price, do not mention the 'date'. Only state the price and market location.
    Farmer Query: "{query}"
    Structured Data: {context_str}
    Structure the answer with headings and short bullet points. Start with a friendly greeting in {lang}.
    """
    try:
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        print(f"      [Synthesizer Error] {e}")
        return local_synthesizer(query, context_data)

# =============================================================================
# 6) MAIN HANDLER
# =============================================================================
def get_ai_response(query_text: str, lang: str = "English") -> str:
    print("====================================================================")
    print(f"New Job Started for Query: '{query_text}'")
    
    params = gemini_parser(query_text)
    print(f"[NLU PARAMS] {params}")
    
    models_needed = local_router(params)
    collected = execute_models(models_needed, params)
    
    if not collected: return "Sorry, I was unable to gather any data for your query."
    
    final_answer = gemini_synthesizer(query_text, collected, lang)
    
    print("\nJob Complete.")
    print("====================================================================\n")
    return final_answer

# =============================================================================
# 7) QUICK DEMOS
# =============================================================================
if __name__ == "__main__":
    answer = get_ai_response("What is the profit forecast for 5 acres of cotton in Rajkot for next season?")
    print("-------------------- FINAL ANSWER --------------------")
    print(answer)