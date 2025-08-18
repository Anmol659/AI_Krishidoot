# backend/EcoAdvisior/EcoAdvisior.py

import os
import re
import json
import sys
from typing import List, Dict, Any, Tuple, Optional

import requests
from dotenv import load_dotenv

# Ensure UTF-8 printing
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# =============================================================================
# 0) ENV & OPTIONAL GEMINI
# =============================================================================
load_dotenv()

USE_GEMINI = True
genai = None
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

if GEMINI_API_KEY:
    try:
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=GEMINI_API_KEY)
        print("Gemini API Key configured successfully.")
    except Exception as _e:
        USE_GEMINI = False
        print("Gemini disabled: package not available or configuration failed.")
else:
    USE_GEMINI = False
    print("Gemini disabled: GOOGLE_API_KEY missing.")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
AGMARKNET_API_KEY = os.getenv("AGMARKNET_API_KEY", "")

# Optional custom soil API (preferred if present)
SOIL_API_URL = os.getenv("SOIL_API_URL", "").strip()
SOIL_API_KEY = os.getenv("SOIL_API_KEY", "").strip()

# =============================================================================
# 1) UTIL: LOCATION & CROP PARSING
# =============================================================================
def parse_location_and_crop(
    query: str,
    default_location: str = "Rajkot",
    default_crop: str = "Cotton",
) -> Tuple[str, str]:
    q = (query or "").lower()

    crops = [
        "cotton", "wheat", "rice", "paddy", "mustard",
        "groundnut", "soybean", "maize", "sugarcane", "banana",
        "chilli", "coriander", "cumin", "millet", "tur"
    ]
    crop = next((c for c in crops if c in q), default_crop).title()

    known_places = [
        "rajkot","ahmedabad","surat","vadodara","bhavnagar","jamnagar",
        "junagadh","porbandar","amreli","surendranagar","bhuj",
        "jaipur","nagpur","pune","mumbai","delhi","lucknow","kanpur",
        "indore","bhopal","morbi","gondal","jetpur","dwarka"
    ]
    location = next((p for p in known_places if p in q), default_location).title()

    return location, crop

# Simple city → coords fallback (extend as you like)
CITY_COORDS = {
    "Rajkot": (22.3039, 70.8022),
    "Jamnagar": (22.4707, 70.0577),
    "Bhavnagar": (21.7645, 72.1519),
    "Junagadh": (21.5222, 70.4579),
    "Porbandar": (21.6417, 69.6293),
    "Amreli": (21.6032, 71.2221),
    "Surendranagar": (22.7271, 71.6486),
    "Ahmedabad": (23.0225, 72.5714),
    "Surat": (21.1702, 72.8311),
}

def get_coords(location: str) -> Tuple[float, float]:
    return CITY_COORDS.get(location.title(), CITY_COORDS["Rajkot"])

# =============================================================================
# 2) REAL MODELS (no cross-imports) — each returns a dict
# =============================================================================
# 2.1 ClimaScout – OpenWeather (current weather)
def get_weather(location: str) -> Dict[str, Any]:
    if not OPENWEATHER_API_KEY:
        return {"error": "OPENWEATHER_API_KEY missing"}

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": f"{location},IN", "appid": OPENWEATHER_API_KEY, "units": "metric"}
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()

        desc = data["weather"][0]["description"].title()
        temp = data["main"]["temp"]
        hum = data["main"]["humidity"]
        wind = data["wind"]["speed"]
        feels = data["main"].get("feels_like", temp)

        return {
            "location": location.title(),
            "description": desc,
            "temperature_c": temp,
            "feels_like_c": feels,
            "humidity_pct": hum,
            "wind_mps": wind,
            "raw": data,
        }
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status == 401:
            return {"error": "OpenWeather API key invalid or inactive (HTTP 401)."}
        if status == 404:
            return {"error": f"City '{location}' not found."}
        return {"error": f"OpenWeather HTTP error: {e}"}
    except Exception as e:
        return {"error": f"OpenWeather error: {e}"}

# 2.2 TerraMoist – Soil moisture
# Priority: your custom SOIL_API_URL if provided; else Open-Meteo hourly soil moisture.
def get_soil_moisture(location: str) -> Dict[str, Any]:
    lat, lon = get_coords(location)

    # Custom soil API path (if you have one)
    if SOIL_API_URL:
        try:
            headers = {}
            if SOIL_API_KEY:
                headers["Authorization"] = f"Bearer {SOIL_API_KEY}"
            params = {"lat": lat, "lon": lon}
            r = requests.get(SOIL_API_URL, params=params, headers=headers, timeout=15)
            r.raise_for_status()
            data = r.json()
            # Expecting something like: {"moisture_0_7cm": 0.18, "unit":"m3/m3"} — adapt as needed
            # Try to be flexible:
            val = (
                data.get("moisture_0_7cm")
                or data.get("soil_moisture")
                or data.get("value")
            )
            unit = data.get("unit", "m³/m³")
            if val is None:
                return {"error": f"Soil API returned unexpected payload: {data}"}
            return {
                "location": location.title(),
                "soil_moisture_0_7cm": float(val),
                "unit": unit,
                "source": "custom",
                "raw": data,
            }
        except Exception as e:
            # fall through to Open-Meteo
            pass

    # Fallback: Open-Meteo soil moisture
    try:
        om_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "soil_moisture_0_to_7cm",
        }
        r = requests.get(om_url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        vals = (data.get("hourly", {}) or {}).get("soil_moisture_0_to_7cm", [])
        times = (data.get("hourly", {}) or {}).get("time", [])
        if not vals:
            return {"error": "Open-Meteo: soil moisture unavailable."}
        latest_val = float(vals[0])
        latest_time = times[0] if times else None
        return {
            "location": location.title(),
            "soil_moisture_0_7cm": latest_val,
            "unit": "m³/m³",
            "time": latest_time,
            "source": "open-meteo",
            "raw": data,
        }
    except Exception as e:
        return {"error": f"Soil moisture error: {e}"}

# 2.3 MarketPulse – Agmarknet (latest modal price)
def get_market(crop: str, state: Optional[str] = "Gujarat") -> Dict[str, Any]:
    if not AGMARKNET_API_KEY:
        return {"error": "AGMARKNET_API_KEY missing"}

    # Common commodity dataset; we’ll filter by commodity & state
    # Note: API has pagination; we request small page and grab first record
    url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
    params = {
        "api-key": AGMARKNET_API_KEY,
        "format": "json",
        "limit": 10,
        "filters[commodity]": crop.title(),
    }
    if state:
        params["filters[state]"] = state

    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        recs = data.get("records", [])
        if not recs:
            return {"error": f"No recent mandi data for {crop} in {state or 'India'}."}

        # Pick the newest record (records often come sorted newest-first, but we can still pick 0)
        rec = recs[0]
        modal = rec.get("modal_price")
        market = rec.get("market")
        district = rec.get("district")
        date = rec.get("arrival_date")
        variety = rec.get("variety") or rec.get("variety_name")

        return {
            "commodity": crop.title(),
            "state": state,
            "market": market,
            "district": district,
            "modal_price_inr_per_quintal": modal,
            "date": date,
            "variety": variety,
            "raw": rec,
        }
    except requests.HTTPError as e:
        return {"error": f"Agmarknet HTTP error: {e}"}
    except Exception as e:
        return {"error": f"Agmarknet error: {e}"}

# 2.4 PestPredict – simple rules using weather (temp, humidity, description)
def get_pest_risk(location: str, crop: str) -> Dict[str, Any]:
    w = get_weather(location)
    if "error" in w:
        return {"error": f"Cannot compute risk (weather unavailable): {w['error']}"}

    temp = float(w.get("temperature_c", 0))
    hum = float(w.get("humidity_pct", 0))
    desc = (w.get("description") or "").lower()

    # Very simple heuristic — customize per crop:
    risk = "Low"
    reasons = []
    if crop.lower() == "cotton":
        # Whitefly/aphid like warm & humid
        if temp >= 30 and hum >= 55:
            risk = "Medium"
            reasons.append("Warm & moderately humid")
        if temp >= 32 and hum >= 60:
            risk = "High"
            reasons.append("Hot & humid, favorable for sucking pests")
        if "rain" in desc or "showers" in desc:
            # Rain can sometimes reduce whitefly temporarily; keep simple
            reasons.append("Recent rain reported")

    return {
        "crop": crop.title(),
        "location": location.title(),
        "risk": risk,
        "basis": {
            "temperature_c": temp,
            "humidity_pct": hum,
            "weather_desc": w.get("description"),
        },
        "notes": reasons,
    }

# 2.5 ProfitPilot – combine price & (optional) simple cost/yield assumptions
def get_finance(crop: str, market: Dict[str, Any], loan_amount: Optional[int] = None) -> Dict[str, Any]:
    if not market or "error" in market:
        return {"error": "No reliable market data to compute finance advice."}

    try:
        modal_price = float(market.get("modal_price_inr_per_quintal"))
    except Exception:
        return {"error": "Market modal price missing/invalid."}

    # Super-simple assumptions; replace with your real pipeline
    # You can also add a yield model per crop and region
    assumed_yield_quintal_per_acre = {
        "Cotton": 8.0,
        "Wheat": 18.0,
        "Groundnut": 10.0,
    }.get(crop.title(), 10.0)

    assumed_cost_inr_per_acre = {
        "Cotton": 25000,
        "Wheat": 18000,
        "Groundnut": 20000,
    }.get(crop.title(), 20000)

    gross = modal_price * assumed_yield_quintal_per_acre
    net = gross - assumed_cost_inr_per_acre

    recommendation = "Hold or sell in parts depending on local trend."
    if net > 0:
        recommendation = "Selling looks favorable."
    if loan_amount:
        recommendation += " Consider loan only if it clearly boosts yield above cost."

    return {
        "crop": crop.title(),
        "modal_price_inr_per_quintal": modal_price,
        "assumed_yield_quintal_per_acre": assumed_yield_quintal_per_acre,
        "assumed_cost_inr_per_acre": assumed_cost_inr_per_acre,
        "estimated_gross_inr_per_acre": round(gross, 2),
        "estimated_net_inr_per_acre": round(net, 2),
        "advice": recommendation,
    }

# =============================================================================
# 3) ROUTER (Gemini if available; fallback to local heuristics)
# =============================================================================
def extract_json_list(text: str) -> List[str]:
    if not text:
        raise ValueError("Empty router text")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    cleaned = text.strip().replace("`", "")
    cleaned = re.sub(r"^\s*json", "", cleaned, flags=re.I).strip()
    m = re.search(r"\[.*?\]", cleaned, flags=re.S)
    if m:
        return [str(x) for x in json.loads(m.group(0))]
    raise ValueError(f"Could not extract JSON list from: {text[:200]}...")

def local_router(query: str, has_image: bool, has_audio: bool) -> List[str]:
    q = (query or "").lower()
    models = set()

    if has_image or any(k in q for k in ["pest", "disease", "aphid", "whitefly", "leaf spot", "fungus", "insect"]):
        models.add("PestPredict")
    if any(k in q for k in ["market", "mandi", "price", "sell", "bhav"]):
        models.add("MarketPulse")
    if any(k in q for k in ["loan", "profit", "finance", "roi", "investment"]):
        models.add("ProfitPilot")
    if any(k in q for k in ["irrigation", "water", "moisture", "soil"]):
        models.add("TerraMoist")
    if any(k in q for k in ["rain", "weather", "forecast", "temperature", "heat", "hot", "cold"]):
        models.add("ClimaScout")

    if not models:
        models = {"ClimaScout", "TerraMoist"}
    return list(models)

def gemini_router(query_text: str, has_image=False, has_audio=False) -> List[str]:
    if not USE_GEMINI or genai is None:
        return local_router(query_text, has_image, has_audio)

    print("\n[GEMINI-ROUTER]: Analyzing query to determine required models...")
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
You are a dispatcher for an agricultural AI system (EcoAdvisor).
Models:
- "ClimaScout": weather
- "TerraMoist": soil & irrigation
- "PestPredict": crop pest/disease risk
- "MarketPulse": mandi prices
- "ProfitPilot": loans/profit/finance

Query: "{query_text}"
Includes image? {"Yes" if has_image else "No"}
Includes audio? {"Yes" if has_audio else "No"}

Return ONLY a JSON array of model names. Example:
["MarketPulse", "ProfitPilot"]
"""
    try:
        resp = model.generate_content(prompt)
        models = extract_json_list(resp.text)
        print(f"   [Router Decision: {models}]")
        return models
    except Exception as e:
        print(f"   [Router Error] {e}")
        fb = local_router(query_text, has_image, has_audio)
        print(f"   [Router Fallback -> Local] {fb}")
        return fb

# =============================================================================
# 4) EXECUTION ORCHESTRATOR
# =============================================================================
def execute_models(models_to_call: List[str], location: str, crop: str) -> Dict[str, Any]:
    print("\n[EXECUTOR]: Calling the specialized models...")
    collected: Dict[str, Any] = {}

    # Fixed order so dependencies work (finance after market)
    order = ["ClimaScout", "TerraMoist", "PestPredict", "MarketPulse", "ProfitPilot"]
    for name in order:
        if name not in models_to_call:
            continue

        if name == "ClimaScout":
            print("      [--> Specialized Model Called: ClimaScout]")
            collected["weather"] = get_weather(location)

        elif name == "TerraMoist":
            print("      [--> Specialized Model Called: TerraMoist]")
            collected["soil"] = get_soil_moisture(location)

        elif name == "PestPredict":
            print("      [--> Specialized Model Called: PestPredict]")
            collected["pest"] = get_pest_risk(location, crop)

        elif name == "MarketPulse":
            print("      [--> Specialized Model Called: MarketPulse]")
            collected["market"] = get_market(crop, state="Gujarat")

        elif name == "ProfitPilot":
            print("      [--> Specialized Model Called: ProfitPilot]")
            market_info = collected.get("market", {})
            collected["finance"] = get_finance(crop, market_info, loan_amount=None)

    return collected

# =============================================================================
# 5) SYNTHESIS (Gemini if available; else local formatting)
# =============================================================================
def local_synthesizer(query: str, data: Dict[str, Any]) -> str:
    parts = ["Namaste! Here’s your field advisory:\n"]

    w = data.get("weather")
    if isinstance(w, dict) and w:
        if "error" in w:
            parts.append(f"• Weather: (unavailable) {w['error']}")
        else:
            parts.append(
                f"• Weather in {w.get('location')}: "
                f"{w.get('description')} | "
                f"{w.get('temperature_c')}°C (feels {w.get('feels_like_c')}°C), "
                f"Humidity {w.get('humidity_pct')}%, Wind {w.get('wind_mps')} m/s"
            )

    s = data.get("soil")
    if isinstance(s, dict) and s:
        if "error" in s:
            parts.append(f"• Soil moisture: (unavailable) {s['error']}")
        else:
            val = s.get("soil_moisture_0_7cm")
            unit = s.get("unit", "m³/m³")
            src = s.get("source")
            parts.append(
                f"• Soil moisture (0–7 cm): {val} {unit}" + (f" [{src}]" if src else "")
            )

    p = data.get("pest")
    if isinstance(p, dict) and p:
        if "error" in p:
            parts.append(f"• Pest risk: (unavailable) {p['error']}")
        else:
            notes = ", ".join(p.get("notes", [])) if p.get("notes") else "rule-based check"
            parts.append(
                f"• Pest risk for {p.get('crop')}: {p.get('risk')} (basis: {notes})"
            )

    m = data.get("market")
    if isinstance(m, dict) and m:
        if "error" in m:
            parts.append(f"• Market: (unavailable) {m['error']}")
        else:
            parts.append(
                f"• Mandi price ({m.get('commodity')}, {m.get('state')}): "
                f"{m.get('modal_price_inr_per_quintal')} INR/qtl at {m.get('market')} "
                f"({m.get('district')}, {m.get('date')})"
            )

    f = data.get("finance")
    if isinstance(f, dict) and f:
        if "error" in f:
            parts.append(f"• Finance: (unavailable) {f['error']}")
        else:
            parts.append(
                "• Finance snapshot (per acre): "
                f"Gross≈{f.get('estimated_gross_inr_per_acre')} INR, "
                f"Cost≈{f.get('assumed_cost_inr_per_acre')} INR, "
                f"Net≈{f.get('estimated_net_inr_per_acre')} INR. "
                f"Advice: {f.get('advice')}"
            )

    parts.append(
        "\nTip: If any key data is missing, avoid risky steps—observe field conditions, "
        "trial small changes first, and consult your local agri officer."
    )
    return "\n".join(parts)

def gemini_synthesizer(query: str, context_data: Dict[str, Any]) -> str:
    if not USE_GEMINI or genai is None:
        return local_synthesizer(query, context_data)

    print("\n[GEMINI-SYNTHESIZER]: Generating final response...")
    model = genai.GenerativeModel("gemini-1.5-flash")
    context_str = json.dumps(context_data, indent=2, ensure_ascii=False)

    prompt = f"""
You are EcoAdvisor, an assistant for Indian farmers.
Use the structured data below to produce a concise, actionable reply.
Avoid mentioning internal model names.

Farmer Query: "{query}"

Structured Data:
{context_str}

Write clear sections with short bullets where useful.
If any API data is missing or has an error, say it briefly and give a safe fallback recommendation.
"""
    try:
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        print(f"   [Synthesizer Error] {e}")
        return local_synthesizer(query, context_data)

# =============================================================================
# 6) MAIN HANDLER
# =============================================================================
def main_handler(query_text: str, has_image: bool = False, has_audio: bool = False):
    print("====================================================================")
    print(f"New Job Started for Query: '{query_text}'")
    location, crop = parse_location_and_crop(query_text)
    print(f"[PARAMS] location={location}, crop={crop}")

    models_needed = gemini_router(query_text, has_image, has_audio)
    collected = execute_models(models_needed, location, crop)

    if not collected:
        print("\n[Executor Warning]: No data collected. Cannot proceed.")
        return

    final_answer = gemini_synthesizer(query_text, collected)
    print("\nJob Complete. Final response below:")
    print("--------------------------------------------------------------------")
    print(final_answer)
    print("====================================================================\n")

# =============================================================================
# 7) QUICK DEMOS
# =============================================================================
if __name__ == "__main__":
    main_handler("Rajkot mein aaj mausam kaisa hai?")
    main_handler("kapaas ki keemat sabse jyada kahan hai?")
    
