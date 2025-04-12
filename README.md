# Emotive Conversational Agent

A conversational agent with emotional intelligence using Google's Gemini API.

## Setup

1. Clone this repository
2. Create a `.api_key.json` file with your Gemini API key:
```json
{
  "api_key": "YOUR_API_KEY_HERE"
}
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run the initial setup:
```bash
python initial_setup.py
```

## Features

- AI-powered conversations using Google's Gemini models
- Emotion detection and realtime emotion analysis
- Natural language understanding

## Technologies

- Python
- Google Gemini API

## Running the Application

The `realtime_audio_gpt.py` script is more stable than `gemini_live_audio.py`. To run the more stable version, use the following command:

```bash
python gpt_realtime/realtime_audio_gpt.py
```

Ensure that your `.api_key.json` file is correctly configured with your OpenAI and Gemini API keys. 