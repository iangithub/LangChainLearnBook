from langchain_core.runnables import RunnableSequence,RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser


import os
from dotenv import dotenv_values

config = dotenv_values(".env")

# 初始化語言模型
llm = AzureChatOpenAI(
    azure_endpoint=config.get("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=config.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=config.get("AZURE_OPENAI_API_VERSION"),
    api_key=config.get("AZURE_OPENAI_KEY"),
    temperature=0.8 
)


# 寫作風格的參考資料
style_examples = """
1. 一鄉二里，共三夫子不識四書五經六義，竟敢教七八九子，十分大膽
2. 十室九貧，湊得八兩七錢六分五毫四厘，尚且又三心二意，一等下流
3. 圖畫裡，龍不吟，虎不嘯，小小書童可笑可笑
4. 棋盤裡，車無輪，馬無韁，叫聲將軍提防提防
5. 鶯鶯燕燕翠翠紅紅處處融融洽洽
6. 雨雨風風花花葉葉年年暮暮朝朝
"""

# 定義提示樣板
writing_template = ChatPromptTemplate.from_template("""
你是一位精通對聯創作的文學大師。請根據以下提供的主題創作一組對聯。

主題: {topic}

請參考以下的寫作風格範例，創作時要體現類似的韻律感和文字技巧：

{style_examples}

要求：
1. 創作一組對仗工整、意境深遠的對聯
2. 對聯應與給定主題相關
3. 儘量融入範例中展現的數字遞進、重複疊字等修辭技巧
4. 確保對聯在音律和結構上和諧統一

請提供：
- 上聯
- 下聯
- 簡短解釋（說明對聯與主題的關聯，以及使用的技巧）
""")


# 定義分析函數
def analyze_couplet(couplet):
    lines = couplet.split('\n')
    if len(lines) < 2:
        return {"error": "無法識別完整對聯"}
    
    upper = lines[0].split('：')[-1].strip() # 取得上聯
    lower = lines[1].split('：')[-1].strip() # 取得下聯
    
    word_count = len(upper)
    char_set = set(upper + lower)
    repeated_chars = [char for char in char_set if (upper + lower).count(char) > 1]
    
    return {
        "word_count": word_count,
        "unique_chars": len(char_set),
        "repeated_chars": ', '.join(repeated_chars),
        "upper": upper,
        "lower": lower
    }


# 建立對聯生成系統
couplet_generation_system = RunnableSequence(
    {
        "topic": RunnablePassthrough(),
        "style_examples": lambda _: style_examples
    },
    writing_template,
    llm,
    lambda x: {"content": x.content}, # 將 LLM 輸出轉換為字典
    RunnablePassthrough.assign(
        analysis=lambda x: analyze_couplet(x["content"])
    ),
    lambda x: {
        "content": x["content"],
        "analysis": x["analysis"],
    }
)

# 使用對聯生成系統
topic = "生成式AI"
result = couplet_generation_system.invoke({"topic": topic})
print(result["content"])
print("\n分析:")
print(result["analysis"])