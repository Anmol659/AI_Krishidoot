# Add the current directory to Python path to import EcoAdvisor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from EcoAdvisior.EcoAdvisior import main_handler, parse_location_and_crop

# Launch configuration
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=not is_hf_space,
        show_error=True,
        show_tips=True,
        enable_queue=True
    )