import gradio as gr
import os
import sys
from typing import Optional, Tuple
import json

# Add the current directory to Python path to import EcoAdvisor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from EcoAdvisior.py import main_handler, parse_location_and_crop

# Custom CSS for better styling
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.header {
    text-align: center;
    background: linear-gradient(90deg, #4CAF50, #45a049);
    color: white;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}
.feature-box {
    background: #f8f9fa;
    border-left: 4px solid #4CAF50;
    padding: 15px;
    margin: 10px 0;
    border-radius: 5px;
}
"""

def process_query(query: str, location: str = "", crop: str = "") -> Tuple[str, str]:
    """
    Process the user query and return the response along with parsed parameters.
    """
    if not query.strip():
        return "Please enter a query about farming, weather, or market conditions.", ""
    
    try:
        # If location and crop are provided explicitly, use them
        if location.strip() or crop.strip():
            final_query = f"{query} in {location} for {crop}" if location and crop else query
        else:
            final_query = query
        
        # Parse location and crop from the query
        parsed_location, parsed_crop = parse_location_and_crop(final_query)
        
        # Capture the output from main_handler
        import io
        from contextlib import redirect_stdout
        
        output_buffer = io.StringIO()
        
        # Redirect stdout to capture the printed output
        with redirect_stdout(output_buffer):
            main_handler(final_query, has_image=False, has_audio=False)
        
        # Get the captured output
        full_output = output_buffer.getvalue()
        
        # Extract just the final response (after the last "----" line)
        lines = full_output.split('\n')
        response_started = False
        response_lines = []
        
        for line in lines:
            if "--------------------------------------------------------------------" in line:
                response_started = True
                continue
            elif "====================================================================" in line and response_started:
                break
            elif response_started:
                response_lines.append(line)
        
        final_response = '\n'.join(response_lines).strip()
        
        if not final_response:
            final_response = "I apologize, but I couldn't generate a proper response. Please try rephrasing your query."
        
        # Create parameter info
        param_info = f"📍 Location: {parsed_location} | 🌾 Crop: {parsed_crop}"
        
        return final_response, param_info
        
    except Exception as e:
        error_msg = f"An error occurred while processing your query: {str(e)}"
        return error_msg, ""

def create_interface():
    """
    Create and configure the Gradio interface.
    """
    
    with gr.Blocks(css=custom_css, title="EcoAdvisor - AI Agricultural Assistant") as interface:
        
        # Header
        gr.HTML("""
        <div class="header">
            <h1>🌾 EcoAdvisor</h1>
            <p>AI-Powered Agricultural Advisory System for Indian Farmers</p>
        </div>
        """)
        
        # Features description
        gr.HTML("""
        <div class="feature-box">
            <h3>🚀 Available Services:</h3>
            <ul>
                <li><strong>ClimaScout:</strong> Real-time weather conditions and forecasts</li>
                <li><strong>TerraMoist:</strong> Soil moisture analysis and irrigation advice</li>
                <li><strong>PestPredict:</strong> AI-powered pest and disease risk assessment</li>
                <li><strong>MarketPulse:</strong> Live mandi prices and market trends</li>
                <li><strong>ProfitPilot:</strong> Financial advisory and profit optimization</li>
            </ul>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Main query input
                query_input = gr.Textbox(
                    label="Ask your farming question",
                    placeholder="Example: What's the weather like in Rajkot? or Cotton prices in Gujarat today",
                    lines=3,
                    max_lines=5
                )
                
                with gr.Row():
                    location_input = gr.Textbox(
                        label="Location (Optional)",
                        placeholder="e.g., Rajkot, Jamnagar, Ahmedabad",
                        scale=1
                    )
                    crop_input = gr.Textbox(
                        label="Crop (Optional)",
                        placeholder="e.g., Cotton, Wheat, Groundnut",
                        scale=1
                    )
                
                submit_btn = gr.Button("Get Advisory", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="feature-box">
                    <h4>💡 Example Queries:</h4>
                    <ul>
                        <li>"Weather in Rajkot today"</li>
                        <li>"Cotton prices in Gujarat"</li>
                        <li>"Soil moisture for wheat farming"</li>
                        <li>"Pest risk for cotton in Jamnagar"</li>
                        <li>"Profit analysis for groundnut"</li>
                    </ul>
                </div>
                """)
        
        # Output section
        with gr.Row():
            with gr.Column():
                param_output = gr.Textbox(
                    label="Detected Parameters",
                    interactive=False,
                    show_label=True
                )
                
                response_output = gr.Textbox(
                    label="EcoAdvisor Response",
                    lines=15,
                    max_lines=25,
                    interactive=False,
                    show_label=True
                )
        
        # Event handlers
        submit_btn.click(
            fn=process_query,
            inputs=[query_input, location_input, crop_input],
            outputs=[response_output, param_output]
        )
        
        query_input.submit(
            fn=process_query,
            inputs=[query_input, location_input, crop_input],
            outputs=[response_output, param_output]
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 20px; padding: 10px; background: #f8f9fa; border-radius: 5px;">
            <p><strong>EcoAdvisor</strong> - Empowering farmers with AI-driven insights</p>
            <p>🌱 Supporting sustainable agriculture across India 🇮🇳</p>
        </div>
        """)
    
    return interface

# Create and launch the interface
if __name__ == "__main__":
    # Check if running on Hugging Face Spaces
    is_hf_space = os.getenv("SPACE_ID") is not None
    
    interface = create_interface()
    
    # Launch configuration
    interface.launch(
        server_name="0.0.0.0" if is_hf_space else "127.0.0.1",
        server_port=7860,
        share=False if is_hf_space else True,
        show_error=True,
        show_tips=True,
        enable_queue=True
    )