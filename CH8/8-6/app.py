import google.generativeai as genai
import os
from dotenv import dotenv_values

config = dotenv_values(".env")
genai.configure(api_key=config.get("Gemini_API_KEY"))

# Upload the video file
video_file_name = "microsoft_ai_mvp_talk_2023.mp4"
print(f"Uploading file...")
video_file = genai.upload_file(path=video_file_name)
print(f"Completed upload: {video_file.uri}")

# Wait for the video to be processed
import time

while video_file.state.name == "PROCESSING":
    print("Waiting for video to be processed.")
    time.sleep(10)
    video_file = genai.get_file(video_file.name)

if video_file.state.name == "FAILED":
    raise ValueError(video_file.state.name)
print(f"Video processing complete: " + video_file.uri)

# Upload the image file
image_file_name = "ryan.jpg"
print(f"Uploading file...")
image_file = genai.upload_file(path=image_file_name)
print(f"Completed upload: {image_file.uri}")

# Create the prompt.
prompt = "請問你從影片中看到什麼？用繁體中文回答。"
prompt = "請詳細地條列出影片中每個人所說的話，用繁體中文回答。"
prompt = "請問影片中有沒有出現圖片裡的這個人，在第幾秒，他說了什麼，用繁體中文回答。"
# Set the model to Gemini 1.5 flash.
model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
# Make the LLM request.
print("Gemini思考中...")
response = model.generate_content(
    [prompt, image_file, video_file], request_options={"timeout": 600}
)
print(response.text)

# 沒有要再問問題時，再把影片從雲端刪除
# Delete the cloud video file
# genai.delete_file(video_file.name)
# print(f"Deleted file {video_file.uri}")
