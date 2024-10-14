from flask import Flask, render_template, request, redirect, url_for
# Azure OpenAI
from openai import AzureOpenAI
import json
import os

from dotenv import dotenv_values
config = dotenv_values(".env")

# Azure OpenAI Settings
client = AzureOpenAI(
    api_version=config.get("AzureOpenAI_VERSION"),
    azure_endpoint=config.get("AzureOpenAI_BASE"),
    api_key=config.get("AzureOpenAI_KEY"),
)


app = Flask(__name__)

all_messages = []

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if request.method == "POST":
        print("File uploaded!")
        file = request.files["audio"]
        file.save("static/" + file.filename)
        whisper_result = azure_whisper()
        openai_result = azure_openai(whisper_result)
        voice_result = azure_voice(openai_result)
        results = {
            "whisper": whisper_result,
            "openai": openai_result,
            "voice": voice_result,
        }
        return json.dumps(results)


@app.route("/clear_history", methods=["POST"])
def clear_history():
    if request.method == "POST":
        print("POST!")
        data = request.form
        print(data)
        global all_messages
        all_messages = []
        return "History cleared!"


@app.route("/call_llm", methods=["POST"])
def call_llm():
    if request.method == "POST":
        print("POST!")
        data = request.form
        print(data)
        to_llm = data["message"]
        try:
            result = azure_openai(to_llm)
        except Exception as e:
            print(e)
            return "Sorry, something wrong! You can talk to me later."
        print(result)
        return result


def azure_whisper():
    audio_file = open("static/audio.webm", "rb")
    transcript = client.audio.transcriptions.create(
        model=os.getenv("AzureOpenAI_WHISPER_DEPLOYMENT_NAME"),
        file=audio_file,
        language="en",
    )
    audio_file.close()
    print("Whisper:", transcript.text)
    return transcript.text


def azure_openai(user_input):

    role_play = """
    You are an excellent English teacher. 
    You are teaching a student who is learning English as a second language.
    You will correct the student's grammar and give suggestions.
    Your answer should be less than 100 words.
    """
    if len(all_messages) == 0:
        all_messages.append({"role": "assistant", "content": role_play})

    all_messages.append({"role": "user", "content": user_input})

    completion = client.chat.completions.create(
        model=os.getenv("AzureOpenAI_GPT4o_DEPLOYMENT_NAME"),
        messages=all_messages,
    )
    all_messages.append(
        {"role": "assistant", "content": completion.choices[0].message.content}
    )
    return completion.choices[0].message.content

def azure_voice(azure_openai_result):
    speech_file_path = "static/speech.mp3"
    response = client.audio.speech.create(
        model=os.getenv("AzureOpenAI_TTS_DEPLOYMENT_NAME"),
        voice="alloy",
        input=azure_openai_result,
    )
    response.stream_to_file(speech_file_path)
    return os.getenv("Deploy_URL") + "static/speech.mp3"


if __name__ == "__main__":
    app.run()
