# EcoAdvisor Backend

An AI-powered agricultural advisory system for Indian farmers, providing weather, soil, pest, market, and financial insights.

## Features

- **ClimaScout**: Hyperlocal weather data using OpenWeatherMap
- **TerraMoist**: Soil moisture analysis using Open-Meteo API
- **PestPredict**: ML-based pest risk prediction
- **MarketPulse**: Live mandi prices from AgMarkNet
- **ProfitPilot**: Financial advisory and profit analysis

## Quick Start

### Hugging Face Spaces Deployment

This project is ready for deployment on Hugging Face Spaces:

1. Create a new Space on Hugging Face
2. Upload all files from the backend directory
3. Set your environment variables in the Space settings:
   - `GOOGLE_API_KEY`
   - `OPENWEATHER_API_KEY`
   - `AGMARKNET_API_KEY`
   - `SOIL_API_URL` (optional)
   - `SOIL_API_KEY` (optional)
4. The Gradio interface will automatically launch

### Using Docker (Recommended)

1. Clone the repository and navigate to the backend directory:
```bash
cd backend
```

2. Copy the environment file and add your API keys:
```bash
cp .env.example .env
# Edit .env with your actual API keys
```

3. Build and run with Docker Compose:
```bash
docker-compose up --build
```

### Manual Installation

1. Install Python 3.11+ and create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. Run the application:
```bash
python EcoAdvisior/EcoAdvisior.py
```

## API Keys Required

- **Google Gemini API**: For intelligent query routing and response synthesis
- **OpenWeatherMap API**: For weather data
- **AgMarkNet API**: For mandi price data
- **Soil API** (Optional): Custom soil moisture API

## Docker Commands

```bash
# Build the image
docker build -t ecoadvisor-backend .

# Run the container
docker run -p 8000:8000 --env-file .env ecoadvisor-backend

# Using docker-compose
docker-compose up -d
docker-compose logs -f
docker-compose down
```

## Environment Variables

See `.env.example` for all available configuration options.

## Model Training

The system includes pre-trained models for pest prediction and price forecasting. To retrain:

```bash
# Pest prediction model
python models/PestPredict/pestpredict.py

# Price prediction model
python models/ProfitPilot/ProfitPilot.py
```

## Usage Examples

```python
from EcoAdvisior.EcoAdvisior import main_handler

# Weather query
main_handler("Rajkot mein aaj mausam kaisa hai?")

# Market price query
main_handler("kapaas ki keemat sabse jyada kahan hai?")

# Comprehensive advisory
main_handler("Cotton farming advice for Rajkot with current conditions")
```

## Architecture

The system uses a modular architecture with:
- **Router**: Determines which models to call based on query
- **Executor**: Orchestrates model execution
- **Synthesizer**: Combines results into coherent advice

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.