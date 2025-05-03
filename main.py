import numpy as np
import queue
import re
import threading
import lmstudio as lms
from kokoro import KPipeline
import sounddevice as aud_device

# Increased to 45 to allow for longer sentences before TTS.
# TODO: This is a hyperparameter, need to find a way to replace this with a more dynamic approach.
MAX_WORDS_BUFFER_TTS = 45
SAMPLE_RATE = 24000
PUNCTUATIONS = ['.', '!', '?', ':']

llm = lms.llm()
audio_pipeline = KPipeline(repo_id='hexgrad/Kokoro-82M', lang_code='a', device='cpu')
# These are the queues for text and audio data.
# The text queue will hold the text chunks to be converted to audio.
text_q = queue.Queue()
audio_q = queue.Queue()
aud_device.default.samplerate = SAMPLE_RATE

# Wanted a longer response, replaced with a more complex prompt.
prediction = llm.respond_stream("Explain the theory of relativity in detail.")

# TODO: Improve end_with_punc to handle more punctuation cases.
def end_with_punc(text, punctuation=PUNCTUATIONS):
    text_stripped = text.strip()
    if not text_stripped:
        return False
    last_char = text_stripped[-1]
    is_punctuation = last_char in punctuation
    # Also, I noticed, transcripts are often point-wise.
    # To that end, exclude cases like "1.", "2.", etc.
    # This is a bit of a hack, but it works for the current use case.
    if is_punctuation and last_char == '.' and len(text_stripped) > 1 and text_stripped[-2].isdigit():
        return False
    return is_punctuation

# Example: 1. **Time Dilation:**
# We use re.MULTILINE because a chunk might contain multiple lines.
# So basically, it'll pronounce it as:
# "1 {SMALL PAUSE} "TIME DILATION" {SMALL PAUSE}".
LIST_ITEM_PATTERN = re.compile(r"^\s*(\d+)\.\s*\*\*(.*?)\*\*:\s*", re.MULTILINE)

def format_tts_text(text_chunk):
    def replace_list_item(match):
        number = match.group(1)
        title = match.group(2)
        # Format: "1. Time Dilation, " - aiming for pauses after number and title.
        return f"{number}. {title}, "

    # Apply the replacement and remove any remaining '**' marks.
    formatted_text = LIST_ITEM_PATTERN.sub(replace_list_item, text_chunk)
    formatted_text = formatted_text.replace('**', '')
    return formatted_text.strip()

def generate_prompt():
    text = ''
    text_len = 0
    
    for token in prediction:
        text += token.content
        text_len += 1
        print(token.content, end="", flush=True)
        
        if end_with_punc(text) or text_len >= MAX_WORDS_BUFFER_TTS:
            formatted_chunk = format_tts_text(text)
            if formatted_chunk:
                 text_q.put(formatted_chunk)
            text = ''
            text_len = 0
            # Removed delay.
            # time.sleep(MODEL_LATENCY)
        
    if text:
        formatted_chunk = format_tts_text(text)
        if formatted_chunk:
            text_q.put(formatted_chunk)
    # This would signal the end of the text stream, could be any placeholder but None
    # seems just fine.
    text_q.put(None)
            

def get_audio():
    while True:
        text = text_q.get()
        
        if text is None:
            audio_q.put(None) # This would signal the end of the audio generation stream.
            break
        
        generator = audio_pipeline(text, voice='af_heart')
        audio_parts = []
        for i, (gs, _, audio) in enumerate(generator):
            audio_parts.append(audio.numpy())
        if audio_parts:
            full_audio = np.concatenate(audio_parts, axis=0)
            audio_q.put(full_audio)

def speak_prompt():
    while True:
        audio_data = audio_q.get()
        if audio_data is None:
            break
        # Nice idea here, kept blocking threads for sequential playback.
        aud_device.play(audio_data, blocking=True)


LLM_thread = threading.Thread(target=generate_prompt)
TTS_thread = threading.Thread(target=get_audio)
PLAYER_thread = threading.Thread(target=speak_prompt)

LLM_thread.start()
TTS_thread.start()
PLAYER_thread.start()
LLM_thread.join()
TTS_thread.join()
PLAYER_thread.join()