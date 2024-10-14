import google.generativeai as genai
import os
from dotenv import dotenv_values
config = dotenv_values(".env")

genai.configure(api_key=config.get("Gemini_API_KEY"))

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-latest",
)

# 輸入一個問題
user_input = "如何獲得幸福人生？"

response = model.generate_content(
    user_input,
)

print("Q: " + user_input)
print("A: " + response.text)
