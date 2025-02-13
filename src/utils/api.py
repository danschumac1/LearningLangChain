import os
import openai
from dotenv import load_dotenv
import datetime

# account for deprecation of LLM model
# Get the current date
CURRENT_DATE = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
TARGET_DATE = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if CURRENT_DATE > TARGET_DATE:
    LLM_MODEL = "gpt-3.5-turbo"
else:
    LLM_MODEL = "gpt-3.5-turbo-0301"

def load_env():
    load_dotenv("./resources/.env")
    return os.getenv("OPENAI_API_KEY")

def get_completion(prompt, model=LLM_MODEL):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, 
    )
    return response.choices[0].message["content"]