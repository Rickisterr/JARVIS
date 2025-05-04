import numpy as np
import queue
import re
import threading
import keyboard
import sys
import whisper
from faster_whisper import WhisperModel
import lmstudio as lms
from kokoro import KPipeline
import sounddevice as aud_device

# Increased to 45 to allow for longer sentences before TTS.
# TODO: This is a hyperparameter, need to find a way to replace this with a more dynamic approach.
MAX_WORDS_BUFFER_TTS = 45
SAMPLE_RATE = 24000
PUNCTUATIONS = ['.', '!', '?', ':']
MIN_TOKENS_PER_PROMPT = 4

# Example: 1. **Time Dilation:**
# We use re.MULTILINE because a chunk might contain multiple lines.
# So basically, it'll pronounce it as:
# "1 {SMALL PAUSE} "TIME DILATION" {SMALL PAUSE}".
LIST_ITEM_PATTERN = re.compile(r"^\s*(\d+)\.\s*\*\*(.*?)\*\*:\s*", re.MULTILINE)

llm = lms.llm()
audio_pipeline = KPipeline(repo_id='hexgrad/Kokoro-82M', lang_code='a', device='cpu')
# These queues are for the audio recorded and the text decoded from audio
record_q = queue.Queue()
question_q = queue.Queue()
# These are the queues for text and audio data.
# The text queue will hold the text chunks to be converted to audio.
text_q = queue.Queue()
audio_q = queue.Queue()
aud_device.default.samplerate = SAMPLE_RATE

# TODO: Add dynamic user inputs which take a whole question prompt before sending in place of duration
# Record 5 seconds of audio at 16000 Hz
sample_rate = 16000
duration = 5
audio = 1
prediction = ''
stop = 0

# model = whisper.load_model("base")
model = WhisperModel("base", device="cpu")  # Load Faster Whisper model

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

def replace_list_item(match):
        number = match.group(1)
        title = match.group(2)
        # Format: "1. Time Dilation, " - aiming for pauses after number and title.
        return f"{number}. {title}, "

def format_tts_text(text_chunk):
    # Apply the replacement and remove any remaining '**' marks.
    formatted_text = LIST_ITEM_PATTERN.sub(replace_list_item, text_chunk)
    formatted_text = formatted_text.replace('**', '')
    return formatted_text.strip()

def stop_program():
    global stop
    stop = 1

# TODO: Implement thread blocking/unblocking with event set and wait functions
#       to allow specific threads to run without being suddenly discontinued
#       or paraded over by another
# TODO: Fix issue of only first prompt query reaching LM Studio server
#       but all successive prompts not reaching the server (checked using Dev logs on LM Studio)
def record_user():
    global stop
    while True:
        if stop == 1:
            # sys.exit(0)
            break
        
        audio = aud_device.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        aud_device.wait()
        audio = audio.flatten()  # Ensure 1D array
        
        record_q.put(audio)
        
def get_text():
    global stop
    question = ''
    
    while True:
        audio = record_q.get()
        
        audio = whisper.pad_or_trim(audio)  # Trimming down audio to equivalent lengths
        
        if stop == 1:
            break

        # Transcribe audio with Faster Whisper
        segments, _ = model.transcribe(audio, beam_size=5, language="en")

        # Collect text from segments
        text = "".join(segment.text for segment in segments).strip()
        
        # If 2 or less words in text input, consider no more user input is being given
        if len(text.split(" ")) < MIN_TOKENS_PER_PROMPT:
            question_q.put(question)
            question = ''
        else:
            question += text

def predict():
    global stop
    global prediction
    text = ''
    
    while True:
        text = question_q.get()
        
        if stop == 1:
            break
        
        if len(text.split(" ")) >= MIN_TOKENS_PER_PROMPT:
            prediction = llm.respond_stream(text)
            print("User: ", text)

def generate_prompt():
    global stop
    text = ''
    text_len = 0
    
    while True:
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
            
        if text:
            formatted_chunk = format_tts_text(text)
            if formatted_chunk:
                text_q.put(formatted_chunk)
        # This would signal the end of the text stream, could be any placeholder but None
        # seems just fine.
        if stop == 1:
            break
            
def get_audio():
    while True:
        text = text_q.get()
        
        if stop == 1:
            break
        
        generator = audio_pipeline(text, voice='af_heart')
        audio_parts = []
        for i, (gs, _, audio) in enumerate(generator):
            audio_parts.append(audio.numpy())
        if audio_parts:
            full_audio = np.concatenate(audio_parts, axis=0)
            audio_q.put(full_audio)

# TODO: Fix issue of prompt being spoken being taken as user input by recording
def speak_prompt():
    while True:
        audio_data = audio_q.get()
        
        if stop == 1:
            break
        # Nice idea here, kept blocking threads for sequential playback.
        aud_device.play(audio_data, blocking=True)


print("Starting record program...")

keyboard.add_hotkey('esc', stop_program)

REC_thread = threading.Thread(target=record_user)
TEXT_thread = threading.Thread(target=get_text)
PREDICT_thread = threading.Thread(target=predict)
LLM_thread = threading.Thread(target=generate_prompt)
TTS_thread = threading.Thread(target=get_audio)
PLAYER_thread = threading.Thread(target=speak_prompt)

REC_thread.start()
TEXT_thread.start()
PREDICT_thread.start()
LLM_thread.start()
TTS_thread.start()
PLAYER_thread.start()

REC_thread.join()
TEXT_thread.join()
PREDICT_thread.join()
LLM_thread.join()
TTS_thread.join()
PLAYER_thread.join()
sys.exit(0)