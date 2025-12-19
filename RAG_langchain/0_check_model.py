import os
from google.genai import Client
from dotenv import load_dotenv

# Load your API key
load_dotenv()

client = Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("Fetching available models...")
try:
    # We will just print the name directly without filtering attributes
    for model in client.models.list():
        print(f"Model Name: {model.name}")
except Exception as e:
    print(f"Error fetching models: {e}")