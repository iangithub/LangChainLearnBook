from dotenv import load_dotenv
import os

# 載入 .env 文件
# load_dotenv()
load_dotenv('./.env') # 指定 .env 文件路徑

# 取得環境變數
aoai_key = os.getenv("AOAI_API_KEY")

print(f"AOAI key: {aoai_key}")
