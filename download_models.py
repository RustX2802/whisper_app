"""
This script allows you to download the models used by our speech to text app from their respective librairies. These models are saved in a models folder. This can save you some time when you initialize the app.
"""
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
import pickle

def load_models():
    # 1 - Whisper Speech to Text Model
    model_name = "openai/whisper-large-v2"
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name)

    pickle.dump(model, open("/workspace/models/STT_model_whisper-large-v2.sav", 'wb'))
    pickle.dump(processor, open("/workspace/models/STT_processor_whisper-large-v2.sav", 'wb'))

    # 2 - Summarization Model
    summarizer = pipeline("summarization", model="ainize/kobart-news")
    pickle.dump(summarizer, open("/workspace/models/summarizer.sav", 'wb'))

    # 3 - Other Whisper Speech to Text Model
    model_name = "openai/whisper-large"
    STT_tokenizer = WhisperProcessor.from_pretrained(model_name)
    STT_model = WhisperForConditionalGeneration.from_pretrained(model_name)

    pickle.dump(STT_model,
                open("/workspace/models/STT_model2_whisper-large.sav", 'wb'))
    pickle.dump(STT_tokenizer,
                open("/workspace/models/STT_processor2_whisper-large.sav", 'wb'))

    # 4 - Diarization model - Can't be saved anymore since pyannote.audio v2

if __name__ == '__main__':
    load_models()
    print("done")