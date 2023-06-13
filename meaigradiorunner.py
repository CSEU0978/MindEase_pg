import os
import warnings
import requests

from modules.logging_colors import logger

os.environ['GRADIO_ANALYTICS_ENABLED'] = 'True'
os.environ['BITSANDBYTES_NOWELCOME'] = '1'
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

# This is a hack to prevent Gradio from phoning home when it gets imported
def my_get(url, **kwargs):
    logger.info('Gradio HTTP request redirected to localhost :)')
    kwargs.setdefault('allow_redirects', True)
    return requests.api.request('get', 'http://127.0.0.1/', **kwargs)

original_get = requests.get
requests.get = my_get
import gradio as gr
requests.get = original_get

import importlib
import io
import json
import math
import os
import re
import sys
import time
import traceback
import zipfile
from datetime import datetime
from functools import partial
from pathlib import Path
from threading import Lock

import psutil
import torch
import yaml
from PIL import Image

import gradioui_modules.extensions as extensions_module
from gradioui_modules import chat, shared, training, ui, utils
from gradioui_modules.extensions import apply_extensions
from gradioui_modules.html_generator import chat_html_wrapper
#removed LoRA import
from gradioui_modules.models import load_model, load_soft_prompt, unload_model
from gradioui_modules.text_generation import (generate_reply_wrapper, get_encoded_length, stop_everything_event)

#function related to model loading, loading error if no model, updating settings for manual load
def load_model_wrapper(selected_model, autoload=False):
    if not autoload:
        yield f"The settings for {selected_model} have been updated.\nClick on \"Load the model\" to load it."
        return

    if selected_model == 'None':
        yield "No model selected"
    else:
        try:
            yield f"Loading {selected_model}..."
            shared.model_name = selected_model
            unload_model()
            if selected_model != '':
                shared.model, shared.tokenizer = load_model(shared.model_name)

            yield f"Successfully loaded {selected_model}"
        except:
            yield traceback.format_exc()



# removed Function -def load_lora_wrapper(selected-loras): 



# function to load parameters manually
def load_preset_values(preset_menu, state, return_dict=False):
   
    generate_params = {
        'do_sample': True, 'temperature': 1, 'top_p': 1, 'typical_p': 1, 'epsilon_cutoff': 0,
        'eta_cutoff': 0, 'tfs': 1, 'top_a': 0, 'repetition_penalty': 1, 'encoder_repetition_penalty': 1,
        'top_k': 0, 'num_beams': 1, 'penalty_alpha': 0, 'min_length': 0, 'length_penalty': 1,
        'no_repeat_ngram_size': 0, 'early_stopping': False, 'mirostat_mode': 0,
        'mirostat_tau': 5.0, 'mirostat_eta': 0.1,
    } # removed 3 functions to set params from presets menu

# nudges the model towards direction of specific responses. inject additional context and steer the model's language generation
def upload_soft_prompt(file):
    with zipfile.ZipFile(io.BytesIO(file)) as zf:
        zf.extract('meta.json')
        j = json.loads(open('meta.json', 'r').read())
        name = j['name']
        Path('meta.json').unlink()

    with open(Path(f'softprompts/{name}.zip'), 'wb') as f:
        f.write(file)

    return name



# 3 functions open, save and load prompts from prompts folder

def open_save_prompt():
    fname = f"{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
    return gr.update(value=fname, visible=True), gr.update(visible=False), gr.update(visible=True)


def save_prompt(text, fname):
    if fname != "":
        with open(Path(f'prompts/{fname}.txt'), 'w', encoding='utf-8') as f:
            f.write(text)

        message = f"Saved to prompts/{fname}.txt"
    else:
        message = "Error: No prompt name given."

    return message, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)


def load_prompt(fname):
    if fname in ['None', '']:
        return ''
    elif fname.startswith('Instruct-'):
        fname = re.sub('^Instruct-', '', fname)
        with open(Path(f'characters/instruction-following/{fname}.yaml'), 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            output = ''
            if 'context' in data:
                output += data['context']

            replacements = {
                '<|user|>': data['user'],
                '<|bot|>': data['bot'],
                '<|user-message|>': 'Input',
            }

            output += utils.replace_all(data['turn_template'].split('<|bot-message|>')[0], replacements)
            return output.rstrip(' ')
    else:
        with open(Path(f'prompts/{fname}.txt'), 'r', encoding='utf-8') as f:
            text = f.read()
            if text[-1] == '\n':
                text = text[:-1]

            return text


# need to modify count input token function to fetch user response instead
def count_tokens(text):
    tokens = get_encoded_length(text)
    return f'{tokens} tokens in the input.'


# removed function def download_model_wrapper(repo_id): to download additional models

# removed function def update_model_parameters(state, initial=False): which updated 