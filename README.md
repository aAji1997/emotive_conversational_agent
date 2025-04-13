# Emotive Conversational Agent

A conversational agent with emotional intelligence using Google's Gemini API.

## Setup

1. Clone this repository
2. Create a `.api_key.json` file with your Gemini API key:
```json
{
  "gemini_api_key": "YOUR_GEMINI_API_KEY_HERE",
  "openai_api_key": "YOUR_OPENAI_API__KEY_HERE"
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

- AI-powered conversations using OpenAI's GPT and Google's Gemini models
- Emotion detection and realtime emotion analysis
- Natural language understanding

## Technologies

- Python
- OpenAI real-time API for real-time audio conversations
- Google Gemini API and Google ADK for sentiment analysis and report generation

## Running the Application

The `realtime_audio_gpt.py` script is more stable than `gemini_live_audio.py`. To run the more stable version, use the following command:

```bash
python gpt_realtime/realtime_audio_gpt.py
```

Ensure that your `.api_key.json` file is correctly configured with your OpenAI and Gemini API keys. 