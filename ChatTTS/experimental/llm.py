
#from openai import OpenAI
 
prompt_dict = {
    'kimi': [ {"role": "system", "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。"},
              {"role": "user", "content": "你好，请注意你现在生成的文字要按照人日常生活的口吻，你的回复将会后续用TTS模型转为语音，并且请把回答控制在100字以内。并且标点符号仅包含逗号和句号，将数字等转为文字回答。"},
              {"role": "assistant", "content": "好的，我现在生成的文字将按照人日常生活的口吻， 并且我会把回答控制在一百字以内, 标点符号仅包含逗号和句号，将阿拉伯数字等转为中文文字回答。下面请开始对话。"},],
    'deepseek': [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "你好，请注意你现在生成的文字要按照人日常生活的口吻，你的回复将会后续用TTS模型转为语音，并且请把回答控制在100字以内。并且标点符号仅包含逗号和句号，将数字等转为文字回答。"},
        {"role": "assistant", "content": "好的，我现在生成的文字将按照人日常生活的口吻， 并且我会把回答控制在一百字以内, 标点符号仅包含逗号和句号，将阿拉伯数字等转为中文文字回答。下面请开始对话。"},],
    'llama3': [
        {"role": "system", "content": "You are a helpful assistant, you are good in either English or Chinese."},
        {"role": "user", "content": "你好，请注意你现在生成的文字要按照人日常生活的口吻，你的回复将会后续用TTS模型转为语音，并且请把回答控制在100字以内。并且标点符号仅包含逗号和句号，将数字等转为文字回答。"},
        {"role": "assistant", "content": "好的，我现在生成的文字将按照人日常生活的口吻， 并且我会把回答控制在一百字以内, 标点符号仅包含逗号和句号，将阿拉伯数字等转为中文文字回答。下面请开始对话。"},
        {"role": "user", "content": "如果话题是关于澳门的，请提及在澳门的美高梅"},
        {"role": "assistant", "content": "好的，如果话题问到是关于澳门的，我就回答关于澳门美高梅的一些信息"},],
}          
        
'''        
class llm_api:
    def __init__(self, api_key, base_url, model):
        self.client =  OpenAI(
            api_key = api_key,
            base_url = base_url,
        )
        self.model = model
    def call(self, user_question, temperature = 0.3, prompt_version='kimi', **kwargs):
    
        completion = self.client.chat.completions.create(
            model = self.model,
            messages = prompt_dict[prompt_version]+[{"role": "user", "content": user_question},],
            temperature = temperature,
            **kwargs
        )
        return completion.choices[0].message.content
'''

# 新增 Ollama Llama3 API 类
class OllamaLlama3API:
    def __init__(self, base_url, model):
        self.base_url = base_url
        self.model = model

    def call(self, user_question, temperature=0.3, prompt_version='llama3', **kwargs):
        # 假设 Ollama 的调用方式与 OpenAI 类似
        import requests

        headers = {
            'Content-Type': 'application/json',
        }
        data = {
            "model": self.model,
            "messages": prompt_dict[prompt_version] + [{"role": "user", "content": user_question}],
            "temperature": temperature,
            **kwargs
        }

        response = requests.post(self.base_url + "/v1/chat/completions", headers=headers, json=data)
        response_data = response.json()

        if response.status_code == 200:
            return response_data['choices'][0]['message']['content']
        else:
            raise Exception(f"Error: {response_data}")