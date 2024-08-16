import ChatTTS
import torch
import torchaudio

# 加载模型
chat = ChatTTS.Chat()
chat.load()

# from ChatTTS.experimental.llm import llm_api
from llm import OllamaLlama3API
# enable Ollama serve
ollama_api = OllamaLlama3API(base_url="http://127.0.0.1:11434", model="llama3.1:8b")

#API_KEY = 'sk-271415bd7d02454b80facfeae519e250'
#client = llm_api(api_key=API_KEY,
               # base_url="https://api.deepseek.com",
                # model="deepseek-chat")

# 提问题
user_question = '简单介绍美狮美高梅'
text = ollama_api.call(user_question, prompt_version='llama3')

#text = ollama_api.call(text, prompt_version='llama3')
#text = client.call(user_question, prompt_version='deepseek')
#text = client.call(text, prompt_version='deepseek_TN')

# Weird error with Triton to turn off error supression
import torch._dynamo
torch._dynamo.config.suppress_errors = True

spk_stat = torch.load('asset/spk_stat.pt')
rand_spk = torch.randn(768) * spk_stat.chunk(2)[0] + spk_stat.chunk(2)[1]

# torch.manual_seed(6615)
torch.manual_seed(2)
rand_spk = chat.sample_random_speaker()

params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_emb = rand_spk, # add sampled speaker 
    temperature = .3,   # using custom temperature
    top_P = .7,        # top P decode
    top_K = 20         # top K decode
)
# use oral_(0-9), laugh_(0-2), break_(0-7) 
# to generate special token in text to synthesize.
params_refine_text = ChatTTS.Chat.RefineTextParams(
    prompt='[oral_3][laugh_1][break_4]',
)

# 生成音频
# torch.manual_seed(6615)
wavs = chat.infer(
    text,
    params_refine_text=params_refine_text,
    params_infer_code=params_infer_code,
)

# Audio(wavs[0], rate=24_000, autoplay=True)

# 导出音频
#scipy.io.wavfile.write(filename = "./chattts_audio_result.wav", rate = 24_000, data = wavs[0].T)
torchaudio.save("chattts_audio_result.wav", torch.from_numpy(wavs[0]), 24000)
