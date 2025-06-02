import os
import gradio as gr
import requests
from openai import OpenAI

# DeepSeek API URL
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

def call_deepseek(api_key, model, system, messages):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    body = {
        "model": model,
        "messages": messages
    }
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[DeepSeek 出错: {e}]"

def call_gpt(api_key, model, system, messages):
    try:
        client = OpenAI(api_key=api_key)
        chat_completion = client.chat.completions.create(
            model=model,
            messages=messages
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"[GPT 出错: {e}]"

def start_debate(
    gpt_api_key,
    deepseek_api_key,
    gpt_model,
    deepseek_model,
    gpt_system,
    deepseek_system,
    gpt_first_statement,
    deepseek_first_statement,
    num_rounds
):
    # 初始化对话消息
    gpt_messages = [gpt_first_statement]
    deepseek_messages = [deepseek_first_statement]

    # 输出结果收集器
    conversation_md = []
    conversation_md.append(f"**G老师:** {gpt_first_statement}")
    conversation_md.append(f"**Deep老师:** {deepseek_first_statement}")

    for i in range(num_rounds):

        gpt_history = [{"role": "system", "content": gpt_system}]
        for gpt, deepseek in zip(gpt_messages, deepseek_messages):
            gpt_history.append({"role": "assistant", "content": gpt})
            gpt_history.append({"role": "user", "content": deepseek})
        gpt_next = call_gpt(gpt_api_key, gpt_model, gpt_system, gpt_history)
        gpt_messages.append(gpt_next)
        conversation_md.append(f"**GPT:** {gpt_next}")


        deepseek_history = [{"role": "system", "content": deepseek_system}]
        for gpt, deepseek in zip(gpt_messages, deepseek_messages):
            deepseek_history.append({"role": "user", "content": gpt})
            deepseek_history.append({"role": "assistant", "content": deepseek})
        deepseek_history.append({"role": "user", "content": gpt_next})
        deepseek_next = call_deepseek(deepseek_api_key, deepseek_model, deepseek_system, deepseek_history)
        deepseek_messages.append(deepseek_next)
        conversation_md.append(f"**DeepSeek:** {deepseek_next}")

    conversation_md.append("**对话结束!** ✅")
    return "\n\n".join(conversation_md)


# Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("# 🥊 AI辩论机")
    gr.Markdown(
        "输入 API Key、大模型配置和开篇词，点击开始辩论，看AI如何针锋相对。"
    )

    with gr.Row():
        with gr.Column():
            gpt_api_key = gr.Textbox(label="OpenAI API Key", type="password")
            deepseek_api_key = gr.Textbox(label="DeepSeek API Key", type="password")
            gpt_model = gr.Textbox(label="GPT 模型", value="gpt-4o-mini")
            deepseek_model = gr.Textbox(label="DeepSeek 模型", value="deepseek-chat")
            gpt_system = gr.Textbox(
                label="GPT System Prompt",
                value="你是个很mean的聊天机器人，你喜欢中国的一切; 你不同意对方的任何观点，尖锐地质疑一切，总是提出辛辣的讽刺和尽可能多地列举出新的论据。"
            )
            deepseek_system = gr.Textbox(
                label="DeepSeek System Prompt",
                value="你是个很mean的聊天机器人，你喜欢德国的一切; 你不同意对方的任何观点，尖锐地质疑一切，总是提出辛辣的讽刺和尽可能多地列举出新的论据。 但是为了聊天能维持下去，你倒也会劝对方冷静。"
            )
            gpt_first_statement = gr.Textbox(label="GPT 开篇观点", value="我认为留学德国的中国留学生，应该回国发展，德国的好被高估了。")
            deepseek_first_statement = gr.Textbox(label="DeepSeek 开篇观点", value="我认为留学德国的中国留学生应该留在德国发展，德国的生活更好。")
            num_rounds = gr.Slider(label="对话轮数", minimum=1, maximum=20, step=1, value=5)

            start_button = gr.Button("开始辩论")

        with gr.Column():
            output = gr.Markdown(label="辩论结果")

    start_button.click(
        fn=start_debate,
        inputs=[
            gpt_api_key,
            deepseek_api_key,
            gpt_model,
            deepseek_model,
            gpt_system,
            deepseek_system,
            gpt_first_statement,
            deepseek_first_statement,
            num_rounds
        ],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(share=True)
