import os
from openai import OpenAI

# ====================== 仅需替换这1处 ======================
API_KEY = "sk-5293138d9b904409bbe36f43994f6703"  # 替换成你的真实通义千问 API Key
# ===========================================================

# 配置通义千问华北2（北京）环境
client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 你提供的地域 Base URL
    api_key=API_KEY,
)

# 核心：测试接口连通性（纯文本请求，最简单、最易成功）
def test_ai_connection():
    try:
        # 调用 qwen-plus 模型（你指定的模型名）
        response = client.chat.completions.create(
            model="qwen2-audio-7b-instruct",          # 你指定的模型名称
            messages=[
                {
                    "role": "user",
                    "content": "你好，你是谁"
                }
            ],
            stream=False,               # 关闭流式，减少复杂度
            max_tokens=100,             # 限制回复长度，加快响应
            temperature=0.7
        )

        # 打印成功结果（只输出一次）
        print("✅ AI 接口连接成功！")
        print("📝 模型回复：", response.choices[0].message.content)

    except Exception as e:
        # 精准提示错误原因，方便排查
        print("❌ 连接失败！错误详情：", e)
        if "AuthenticationError" in str(e):
            print("→ 原因：API Key 无效/过期！请检查 Key 是否正确")
        elif "Timeout" in str(e):
            print("→ 原因：网络超时！如需代理，取消下方代理配置的注释")
        elif "NotFoundError" in str(e):
            print("→ 原因：模型名错误！确认模型名是 qwen-plus")

# 可选：添加代理配置（如果你的网络需要）
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# 执行测试
if __name__ == "__main__":
    test_ai_connection()