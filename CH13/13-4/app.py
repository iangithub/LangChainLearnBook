import sys
import os, tempfile

from dotenv import dotenv_values

config = dotenv_values(".env")

# LangChain Azure OpenAI Settings
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# LangChain Azure OpenAI Client
langchain_gpt4o_client = AzureChatOpenAI(
    openai_api_version=config.get("AzureOpenAI_VERSION"),
    azure_deployment=config.get("AzureOpenAI_GPT4o_DEPLOYMENT_NAME"),
    api_key=config.get("AzureOpenAI_KEY"),
    azure_endpoint=config.get("AzureOpenAI_ENDPOINT"),
)

# LangChain Google Generative AI Client
langchain_gemini_client = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest", google_api_key=config.get("Gemini_API_KEY")
)

import base64

IMAGE_NAME = "output.jpg"

from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    ImageMessageContent,
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    MessagingApiBlob,
    ReplyMessageRequest,
    TextMessage,
)

UPLOAD_FOLDER = "static"

app = Flask(__name__)

channel_access_token = config.get("Line_Channel_Access_Token")
channel_secret = config.get("Line_Channel_Secret")
if channel_secret is None:
    print("Specify LINE_CHANNEL_SECRET as environment variable.")
    sys.exit(1)
if channel_access_token is None:
    print("Specify LINE_CHANNEL_ACCESS_TOKEN as environment variable.")
    sys.exit(1)

handler = WebhookHandler(channel_secret)

configuration = Configuration(access_token=channel_access_token)


@app.route("/callback", methods=["POST"])
def callback():
    # get X-Line-Signature header value
    signature = request.headers["X-Line-Signature"]
    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # parse webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"


@handler.add(MessageEvent, message=TextMessageContent)
def message_text(event):
    langchain_openai_result = langchain_llm(
        event.message.text, langchain_gpt4o_client)
    langchain_gemini_result = langchain_llm(
        event.message.text, langchain_gemini_client)
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[
                    TextMessage(
                        text="GPT-4o : " + langchain_openai_result),
                    TextMessage(
                        text="Gemini : " + langchain_gemini_result),
                ],
            )
        )


@handler.add(MessageEvent, message=ImageMessageContent)
def message_image(event):
    with ApiClient(configuration) as api_client:
        line_bot_blob_api = MessagingApiBlob(api_client)
        message_content = line_bot_blob_api.get_message_content(
            message_id=event.message.id
        )
        with tempfile.NamedTemporaryFile(
            dir=UPLOAD_FOLDER, prefix="", delete=False
        ) as tf:
            tf.write(message_content)
            tempfile_path = tf.name

    original_file_name = os.path.basename(tempfile_path)
    os.replace(
        UPLOAD_FOLDER + "/" + original_file_name,
        UPLOAD_FOLDER + "/" + "output.jpg",
    )

    finish_message = "上傳完成，請問你想要問關於這張圖片的什麼問題呢？"

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=finish_message)],
            )
        )


# Open the image file and encode it as a base64 string
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def langchain_llm(user_input, llm_client):
    base64_image = encode_image(UPLOAD_FOLDER + "/" + IMAGE_NAME)
    user_messages = []
    user_messages.append({"type": "text", "text": user_input + "。請用繁體中文回答。"})
    image_url = f"data:image/png;base64,{base64_image}"
    user_messages.append({"type": "image_url", "image_url": {"url": image_url}})
    human_messages = HumanMessage(content=user_messages)
    response = llm_client.invoke([human_messages])
    return response.content


if __name__ == "__main__":
    app.run()
