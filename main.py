import numpy as np
import queue
import time
import threading
import lmstudio as lms
from kokoro import KPipeline
from IPython.display import display, Audio
import sounddevice as aud_device

MAX_WORDS_BUFFER_TTS = 20
MODEL_LATENCY = 0.05
SAMPLE_RATE = 24000

llm = lms.llm()
audio_pipeline = KPipeline(repo_id='hexgrad/Kokoro-82M', lang_code='a', device='cpu')
data_stream = queue.Queue()
aud_device.default.samplerate = SAMPLE_RATE

prediction = llm.respond_stream("Who are you?")

def substr_in_str(string, char_arr):
    for char in char_arr:
        if char in string:
            return True
    return False


def generate_prompt():
    text = ''
    text_len = 0
    
    for token in prediction:
        text += token.content
        text_len += 1
        print(token.content, end="", flush=True)
        
        if substr_in_str(text, ['.', '!', '?', ';']) or text_len > MAX_WORDS_BUFFER_TTS:
            data_stream.put(text.strip())
            text = ''
            text_len = 0
            
            time.sleep(MODEL_LATENCY)
        
    if text:
        data_stream.put(text)
    else:
        data_stream.put(None)
            

def speak_prompt():
    while True:
        text = data_stream.get()
        
        if text is None:
            break
        
        generator = audio_pipeline(text, voice='af_heart')
        
        for i, (gs, _, audio) in enumerate(generator):
            
            # print(gs)
            aud_device.play(audio.numpy(), blocking=True)


LLM_thread = threading.Thread(target=generate_prompt)
TTS_thread = threading.Thread(target=speak_prompt)

LLM_thread.start()
TTS_thread.start()
LLM_thread.join()
TTS_thread.join()