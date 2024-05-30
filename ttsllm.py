import torch
import ChatTTS
from IPython.display import Audio
import scipy

chat = ChatTTS.Chat()
chat.load_models()

from ChatTTS.experimental.llm import llm_api
from ChatTTS.experimental.llm import OllamaLlama3API
#API_KEY = 'sk-271415bd7d02454b80facfeae519e250'
#client = llm_api(api_key=API_KEY,
               # base_url="https://api.deepseek.com",
                # model="deepseek-chat")

user_question = '用中英文混杂的方式，简单介绍宁波'
#text = client.call(user_question, prompt_version='deepseek')
#text = client.call(text, prompt_version='deepseek_TN')
ollama_api = OllamaLlama3API(base_url="http://localhost:11434", model="llama3CN")

text = ollama_api.call(user_question, prompt_version='deepseek')
text = ollama_api.call(text, prompt_version='deepseek_TN')

spk_stat = torch.load('ChatTTS/asset/spk_stat.pt')
rand_spk = torch.randn(768) * spk_stat.chunk(2)[0] + spk_stat.chunk(2)[1]

params_infer_code = {'spk_emb': rand_spk, 'temperature': .3}
params_refine_text = {'prompt': '[oral_2][laugh_0][break_6]'}

# 将infer函数中的文本参数替换为从LLM获取的text
wav = chat.infer(text,
                 params_refine_text=params_refine_text, params_infer_code=params_infer_code)

Audio(wav[0], rate=24_000, autoplay=True)

# 导出音频
scipy.io.wavfile.write(filename = "./chattts_download.wav", rate = 24_000, data = wav[0].T)