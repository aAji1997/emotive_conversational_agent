import json
from google import genai

class GeminiClient:
    def __init__(self):
        with open('.api_key.json', 'r') as f:
            self.api_key = json.load(f)['api_key']
            self.client = genai.Client(api_key=self.api_key)
            self.model = "gemini-2.0-flash-lite"

    def test_gemini(self, query):
        response = self.client.models.generate_content(
            model=self.model,
            contents=query
        )
        return response.text
    
if __name__ == "__main__":
    gemini = GeminiClient()
    print(gemini.test_gemini("Explain the concept of AI"))

        

