import os
import json
import google.generativeai as genai

# ==============================================================================
# 1. CONFIGURE YOUR GEMINI API KEY
# ==============================================================================

# PASTE YOUR API KEY HERE
# WARNING: Do not share your code publicly with a key hardcoded like this.
# For your security, please regenerate this key immediately.
API_KEY = "AIzaSyDlIjBYYWQqb_pGtLqgPvwCTQXSJxKQg30"

try:
    genai.configure(api_key=API_KEY)
    print("Gemini API Key configured successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not configure API. Error: {e}")
    exit()


# ==============================================================================
# 2. THE SPECIALIZED EXPERT MODELS (Your Backend Services)
# ==============================================================================

def ClimaScout(location="Rajkot"):
    print("      [--> Specialized Model Called: ClimaScout]")
    return {"weather_forecast": "Sunny and hot, high of 39 C, no rain expected for the next 5 days."}

def TerraMoist(location="Rajkot"):
    print("      [--> Specialized Model Called: TerraMoist]")
    return {"soil_analysis": "Soil moisture is critically low (0.12 m3/m3), indicating a 'High Stress' level."}

def PestPredict(location="Rajkot", crop="Cotton"):
    print("      [--> Specialized Model Called: PestPredict]")
    return {"pest_forecast": "High risk of aphid and whitefly activity due to the current dry and hot conditions."}

def MarketPulse(crop="Cotton"):
    print("      [--> Specialized Model Called: MarketPulse]")
    return {"market_data": "Price for Cotton at Rajkot APMC is strong at 8350 INR/quintal with a stable upward trend."}

def ProfitPilot(crop="Cotton", loan_amount=50000):
    print("      [--> Specialized Model Called: ProfitPilot]")
    return {"financial_advice": f"The profit forecast for {crop} is positive. A loan of {loan_amount} INR for expansion is considered viable given the strong market."}

# This dictionary maps the model names Gemini will return to the actual Python functions
AVAILABLE_MODELS = {
    "ClimaScout": ClimaScout,
    "TerraMoist": TerraMoist,
    "PestPredict": PestPredict,
    "MarketPulse": MarketPulse,
    "ProfitPilot": ProfitPilot,
}

# ==============================================================================
# 3. THE TWO-STEP GEMINI LOGIC
# ==============================================================================

def gemini_as_router(query_text, has_image=False, has_audio=False):
    """
    *** GEMINI API CALL 1: THE ROUTER ***
    Analyzes the user's query to intelligently decide which models to call.
    """
    print("\n[GEMINI-ROUTER]: Analyzing query to determine required models...")
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    You are an expert dispatcher for an agricultural AI system called EcoAdvisor.
    Your job is to analyze a farmer's query and decide which specialized models are needed to provide a complete answer.

    The available models are:
    - "ClimaScout": For weather, rain, temperature, forecast.
    - "TerraMoist": For soil health, irrigation, water levels, moisture.
    - "PestPredict": For pests, diseases, insects, bugs, or if an image of a plant is provided.
    - "MarketPulse": For crop prices, market rates, selling, mandi bhav.
    - "ProfitPilot": For loans, profit, investment, financial planning.

    **Farmer's Query:** "{query_text}"
    **Input includes an image?** {"Yes" if has_image else "No"}
    **Input includes a voice note?** {"Yes" if has_audio else "No"}

    Based *only* on the query and input type, which models should be called?
    Return your answer as a clean JSON-formatted list of strings.
    Example: ["MarketPulse", "ProfitPilot"]
    """
    
    try:
        response = model.generate_content(prompt)
        # Clean the response to ensure it's valid JSON
        model_list_str = response.text.strip().replace("`", "").replace("json", "")
        models_to_call = json.loads(model_list_str)
        print(f"   [Router Decision: The following models are required: {models_to_call}]")
        return models_to_call
    except (json.JSONDecodeError, Exception) as e:
        print(f"   [Router Error]: Could not parse Gemini's response. Using default. Error: {e}")
        return ["ClimaScout", "TerraMoist"] # Fallback on error

def execute_specialized_models(models_to_call):
    """
    Calls the actual specialized models decided by the Gemini router.
    """
    print("\n[EXECUTOR]: Calling the specialized models...")
    collected_data = {}
    for model_name in models_to_call:
        if model_name in AVAILABLE_MODELS:
            model_function = AVAILABLE_MODELS[model_name]
            collected_data.update(model_function())
        else:
            print(f"   [Executor Warning]: Model '{model_name}' not found in available models.")
    return collected_data

def gemini_as_synthesizer(original_query, context_data):
    """
    *** GEMINI API CALL 2: THE SYNTHESIZER ***
    Takes the structured data and crafts the final, human-readable answer.
    """
    print("\n[GEMINI-SYNTHESIZER]: All data collected. Generating final response...")
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Create a string from the collected data for the prompt
    context_str = json.dumps(context_data, indent=2)

    prompt = f"""
    You are EcoAdvisor, a helpful and expert AI assistant for farmers in India.
    Your task is to create a single, clear, and actionable response to a farmer's query using the real-time data provided to you.
    Do not mention the names of the internal models (like TerraMoist, ClimaScout). Just give the advice directly.

    **Farmer's Original Query:** "{original_query}"

    **Real-time Data Collected:**
    {context_str}

    Based on the data above, provide a comprehensive, easy-to-understand answer.
    Structure the answer with headings if needed. Start with a friendly greeting.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Sorry, I encountered an error while generating the final response. Error: {e}"

# ==============================================================================
# 4. MAIN HANDLER
# ==============================================================================

def main_handler(query_text, has_image=False, has_audio=False):
    """
    Handles a user query from start to finish using the two-step Gemini process.
    """
    print(f"====================================================================\nNew Job Started for Query: '{query_text}'")
    
    # Step 1: Use Gemini as a router.
    models_needed = gemini_as_router(query_text, has_image, has_audio)
    
    # Step 2: Execute the specialized models.
    collected_data = execute_specialized_models(models_needed)
    
    # Step 3: Use Gemini as a synthesizer.
    if not collected_data:
        print("\n[Executor Warning]: No data was collected from specialized models. Cannot proceed.")
        return
        
    final_answer = gemini_as_synthesizer(query_text, collected_data)
    
    print("\nJob Complete. Final response below:")
    print("--------------------------------------------------------------------")
    print(final_answer)
    print("====================================================================\n")

# ==============================================================================
# 5. SIMULATIONS
# ==============================================================================

if __name__ == "__main__":
    # --- Simulation 1: Farmer sends an image and asks about pests ---
    main_handler(query_text="What is this disease on my cotton plant?", has_image=True)

    # --- Simulation 2: Farmer asks a complex financial and market question ---
    main_handler(query_text="Market rates seem good. Should I sell now, or should I take a loan to buy more fertilizer first?")
    
    # --- Simulation 3: Farmer records a voice note asking about irrigation ---
    main_handler(query_text="The ground looks very dry. Do I need to water my fields today?", has_audio=True)