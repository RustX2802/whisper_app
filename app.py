# Models
import torch
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
from pyannote.audio import Pipeline

# Audio Manipulation
import audioread
import librosa
import whisper
from pydub import AudioSegment, silence
import youtube_dl
from youtube_dl import DownloadError
from IPython.display import Audio

# Others
import pandas as pd
from datetime import timedelta
import os
import streamlit as st
import time