import os
import mimetypes # Used to check file types

# ==============================================================================
# 1. SIMULATED SPECIALIZED MODELS (Your 5 Expert Agents)
# In a real app, these would be separate, deployed services or APIs.
# ==============================================================================

def terramoist_analyst(location="Rajkot"):
    print("      -> [Model Called: TerraMoist]")
    # ... logic to get soil data ...
    return {"status": "OK", "data": "Soil moisture is low (0.14 m³/m³). Stress level is moderate."}

def climascout_forecaster(location="Rajkot"):
    print("      -> [Model Called: ClimaScout]")
    # ... logic to get weather data ...
    return {"status": "OK", "data": "Forecast is hot and sunny for the next 4 days. No rain expected."}

def marketpulse_tracker(crop="Cotton", market="Rajkot APMC"):
    print("      -> [Model Called: MarketPulse]")
    # ... logic to get market data ...
    return {"status": "OK", "data": f"Price for {crop} at {market} is 8250 INR/quintal. Trend is stable."}

def crophealth_diagnoser(image_path):
    print("      -> [Model Called: CropHealth]")
    # ... logic to analyze image ...
    return {"status": "OK", "data": "Diagnosis: Aphid infestation detected with high confidence."}

def docu_analyzer(doc_path):
    print("      -> [Model Called: DocuAnalyzer]")
    # ... logic to analyze a document, e.g., a government subsidy PDF ...
    file_name = os.path.basename(doc_path)
    return {"status": "OK", "data": f"Summary of '{file_name}': The scheme offers a 50% subsidy on drip irrigation systems."}
    
def gemini_rag_synthesizer(context_data, query):
    print("      -> [Model Called: Gemini RAG Synthesizer]")
    # ... logic to call Gemini API with context ...
    # This simulates Gemini creating a comprehensive answer
    response = " 종합적인 조언:\n" # "Comprehensive Advice:" in Korean
    response += "1. **Irrigation**: Given the moderate soil stress and no rain forecasted, you should plan to irrigate within the next 2 days.\n"
    response += "2. **Pest Control**: The image confirms an Aphid infestation. You need to take action.\n"
    response += "3. **Market**: Prices are stable, so selling is a good option but not urgent."
    return {"status": "OK", "data": response}


# ==============================================================================
# 2. THE ECOADVISOR - INTELLIGENT ROUTER CLASS
# ==============================================================================

class EcoAdvisor:
    """
    An intelligent routing system that analyzes user queries and dispatches
    them to the appropriate specialized model.
    """
    def __init__(self):
        print("EcoAdvisor Initialized. Ready to assist.")
        # Keyword mapping to route simple text queries
        self.keyword_to_model_map = {
            "price": marketpulse_tracker,
            "market": marketpulse_tracker,
            "bhav": marketpulse_tracker, # "Price" in Gujarati
            "weather": climascout_forecaster,
            "rain": climascout_forecaster,
            "temperature": climascout_forecaster,
            "soil": terramoist_analyst,
            "moisture": terramoist_analyst,
            "irrigate": terramoist_analyst,
        }

    def _get_file_type(self, file_path):
        """Checks if a file is an image, document, or other."""
        if not file_path or not os.path.exists(file_path):
            return None
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            if mime_type.startswith('image/'):
                return 'image'
            if mime_type == 'application/pdf' or mime_type.startswith('text/'):
                return 'document'
        return 'other'

    def route_query(self, query_text="", file_path=None):
        """
        The main routing logic. It inspects the query and file to decide which model to call.
        """
        print(f"\n======================================================\n🚀 New Query Received: text='{query_text}', file='{file_path}'")
        
        # --- Priority 1: Handle files ---
        file_type = self._get_file_type(file_path)
        if file_type == 'image':
            print("   [Routing Decision: File is an image. Using CropHealth.]")
            return crophealth_diagnoser(file_path)
        
        if file_type == 'document':
            print("   [Routing Decision: File is a document. Using DocuAnalyzer.]")
            return docu_analyzer(file_path)

        # --- Priority 2: Handle simple text queries with keywords ---
        for keyword, model_func in self.keyword_to_model_map.items():
            if keyword in query_text.lower():
                print(f"   [Routing Decision: Keyword '{keyword}' found. Using {model_func.__name__}.]")
                return model_func()
        
        # --- Priority 3: Fallback to complex RAG for generic or multi-part questions ---
        print("   [Routing Decision: No specific keywords or files. Using full Gemini RAG.]")
        # In a real scenario, we would gather context from multiple models
        print("      -> [RAG Step: Gathering context...]")
        context = {
            "soil": terramoist_analyst()['data'],
            "weather": climascout_forecaster()['data'],
            "health_notes": "Image analysis suggests pest issues." if file_type else "No image provided."
        }
        return gemini_rag_synthesizer(context, query_text)

# ==============================================================================
# 3. HOW TO USE THE ECOADVISOR
# ==============================================================================

if __name__ == '__main__':
    # Create a single instance of the advisor
    advisor = EcoAdvisor()

    # --- SIMULATION 1: Simple text query about price ---
    # EXPECTATION: Only the MarketPulse model should be called.
    response1 = advisor.route_query(query_text="What is the cotton bhav today?")
    print("   >> Final Answer:", response1['data'])

    # --- SIMULATION 2: Image-based query ---
    # EXPECTATION: Only the CropHealth model should be called.
    # Create a dummy file to simulate the farmer's upload
    with open("dummy_leaf_image.jpg", "w") as f: f.write("fake image data")
    response2 = advisor.route_query(query_text="What is this disease?", file_path="dummy_leaf_image.jpg")
    print("   >> Final Answer:", response2['data'])
    os.remove("dummy_leaf_image.jpg") # Clean up dummy file

    # --- SIMULATION 3: Simple document query ---
    # EXPECTATION: Only the DocuAnalyzer model should be called.
    with open("subsidy_scheme.pdf", "w") as f: f.write("fake pdf data")
    response3 = advisor.route_query(query_text="Can you summarize this for me?", file_path="subsidy_scheme.pdf")
    print("   >> Final Answer:", response3['data'])
    os.remove("subsidy_scheme.pdf") # Clean up dummy file

    # --- SIMULATION 4: Complex, multi-part query ---
    # EXPECTATION: The system should fall back to the full RAG process.
    response4 = advisor.route_query(query_text="My plants look sick and the soil is dry, what should I do?")
    print("   >> Final Answer:", response4['data'])