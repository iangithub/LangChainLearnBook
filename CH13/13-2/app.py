import sys
import os, tempfile

from dotenv import dotenv_values

config = dotenv_values(".env")

# Azure OpenAI
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint=config.get("AzureOpenAI_ENDPOINT"),
    api_version=config.get("AzureOpenAI_VERSION"),
    api_key=config.get("AzureOpenAI_KEY"),
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
    azure_openai_result = azure_openai(event.message.text)
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=azure_openai_result)],
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


def azure_openai(user_input):

    base64_image = encode_image(UPLOAD_FOLDER + "/" + IMAGE_NAME)

    message_text = [
        {
            "role": "system",
            "content": "",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{user_input}",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ],
        },
    ]

    message_text[0][
        "content"
    ] = """
    你是一個人工智慧助理,
    請一律用繁體中文回答。"
    """

    completion = client.chat.completions.create(
        model=config.get("AzureOpenAI_GPT4o_DEPLOYMENT_NAME"),
        messages=message_text,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
    )
    print(completion)
    return completion.choices[0].message.content


# Open the image file and encode it as a base64 string
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


if __name__ == "__main__":
    app.run()
