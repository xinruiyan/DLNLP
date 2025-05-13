
def call_qwen_model(prompt: str) -> str:

    print("正在使用通用大模型（Qwen3）生成网页...")
    # 以下为通过通义千问返回的 HTML 内容
    return """<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <title>自然语言处理（NLP）简介</title>
</head>
<body>
    <h1>自然语言处理（NLP）简介</h1>
    <p>自然语言处理是人工智能的一个重要分支，它使计算机能够理解、解释、生成人类语言。</p>

    <h2>典型应用</h2>
    <h3>1. 机器翻译</h3>
    <img src="images/machine_translation.png" alt="机器翻译" width="300">
    <p>将一种语言自动翻译为另一种语言，如谷歌翻译。</p>

    <h3>2. 情感分析</h3>
    <img src="images/sentiment_analysis.png" alt="情感分析" width="300">
    <p>分析文本中的情绪倾向，如正面、负面或中性。</p>

    <h3>3. 聊天机器人</h3>
    <img src="images/chatbot.png" alt="聊天机器人" width="300">
    <p>通过对话系统实现人机交互，例如智能客服。</p>
</body>
</html>"""

def save_html_file(html_content: str, filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f" 网页已保存为：{filename}")

if __name__ == "__main__":
    prompt = (
        "生成一个介绍自然语言处理（NLP）的网页，内容包括：标题、简介段落，"
        "三个应用（机器翻译、情感分析、聊天机器人）及简要说明，用HTML语言返回。"
    )
    html_code = call_qwen_model(prompt)
    save_html_file(html_code, "nlp_webpage_with_all_images.html")
