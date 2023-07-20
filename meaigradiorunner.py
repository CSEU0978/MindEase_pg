import os
import warnings

import requests

from modules.logging_colors import logger

os.environ['GRADIO_ANALYTICS_ENABLED'] = 'True'
os.environ['BITSANDBYTES_NOWELCOME'] = '1'
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


# This is a hack to prevent Gradio from phoning home when it gets imported
def my_get(url, **kwargs):
    logger.info('Gradio HTTP request redirected to localhost ;P')
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

import modules.extensions as extensions_module
from modules import chat, shared, ui  # removed training, utils
from modules.extensions import apply_extensions
from modules.html_generator import chat_html_wrapper
from modules.LoRA import add_lora_to_model
from modules.models import load_model  # load_soft_prompt, unload_model
from modules.text_generation import (  # generate_reply_wrapper,
    get_encoded_length,
    stop_everything_event)

# Torch Cuda Sanity checks 
if torch.cuda.is_available:
    print("\n Cuda is enabled")
    print("\n Cuda Device Count is :", torch.cuda.device_count())
    print("\n Cuda Device Name :", torch.cuda.get_device_name())
    print("\n Cuda Device Compute Capability :", torch.cuda.get_device_capability())
    print("\n Cuda is initialized: ", torch.cuda.is_initialized())
    print("\n")
elif not torch.cuda.is_initialized:
    print("Cuda is not initialised")
else:
    print("Cuda is DISABLED")


# updated def load_model_wrapper function related to model loading, loading error if no model, updating settings for manual load
def load_model_wrapper(selected_model):
    try:
        yield f"Loading {selected_model}..."
        shared.model_name = selected_model
        unload_model()
        if selected_model != '':
            shared.model, shared.tokenizer = load_model(shared.model_name)

        yield f"Successfully loaded {selected_model}"
    except:
        yield traceback.format_exc()


def load_lora_wrapper(selected_loras):
    yield ("Applying the following LoRAs to {}:\n\n{}".format(shared.model_name, '\n'.join(selected_loras)))
    add_lora_to_model(selected_loras)
    yield ("Successfuly applied the LoRAs")


def load_preset_values(state, return_dict=False):
    print("here 1 ")
    generate_params = {}
    print("here 2")
    if not shared.args['flexgen']:
        for k in shared.generate_params:
            print(k)
            generate_params[k] = shared.generate_params[k]
    else: 'Naive'
    print(shared.gradio)
    #state.update(generate_params)
    return generate_params #, state


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


# removed 3 functions open, save and load prompts from prompts folder

def load_prompt(fname):
    if fname in ['None', '']:
        return ''
    # removed elif fname.startswith('Instruct-'):
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
# removed function def update_model_parameters(state, initial=False): which updated command line arguments based on interface values
# gets model settings from shared file for modelspecific or userspecific settings

def get_model_specific_settings(model):
    settings = shared.model_config
    model_settings = {}

    for pat in settings:
        if re.match(pat.lower(), model.lower()):
            for k in settings[pat]:
                model_settings[k] = settings[pat][k]

    return model_settings


# loads model/user specific settings from function above
def load_model_specific_settings(model, state, return_dict=False):
    model_settings = get_model_specific_settings(model)
    for k in model_settings:
        if k in state:
            state[k] = model_settings[k]

    return state


# removed def save_model_settings(model,state): as users will not be saving model settings
# removed def create_model_menus(): gradio page where you select the model, define gpu,cpu mem, loading settings autogptq,llama,transformers,etc
# removed def create_settings_menu(default_preset): parameter tuning menu with the top_k, top_p, temp, repetition values
'''
def settings_menu(state, default_preset):
    generate_params = load_preset_values(state, return_dict=True)
 # Main parameters
    shared.gradio['seed'] = gr.State({'seed': shared.settings['seed']})
    shared.gradio['temperature'] = gr.State({'temperature': shared.generate_params['temperature']})
    shared.gradio['top_p'] = gr.State({'top_p': shared.generate_params['top_p']})
    shared.gradio['top_k'] = gr.State({'top_k': shared.generate_params['top_k']})
    shared.gradio['typical_p'] = gr.State({'typical_p': shared.generate_params['typical_p']})
    shared.gradio['epsilon_cutoff'] = gr.State({'epsilon_cutoff': shared.generate_params['epsilon_cutoff']})
    shared.gradio['eta_cutoff'] = gr.State({'eta_cutoff': shared.generate_params['eta_cutoff']})
    shared.gradio['tfs'] = gr.State({'tfs': shared.generate_params['tfs']})
    shared.gradio['top_a'] = gr.State({'top_a': shared.generate_params['top_a']})

    shared.gradio['repetition_penalty'] = gr.State({'repetition_penalty': shared.generate_params['repetition_penalty']})
    shared.gradio['encoder_repetition_penalty'] = gr.State({'encoder_repetition_penalty': shared.generate_params['encoder_repetition_penalty']})
    shared.gradio['no_repeat_ngram_size'] = gr.State({'no_repeat_ngram_size': shared.generate_params['no_repeat_ngram_size']})
    shared.gradio['min_length'] = gr.State({'min_length': shared.generate_params['min_length']})
    shared.gradio['do_sample'] = gr.State({'do_sample': shared.generate_params['do_sample']})

    # Contrastive Search
    shared.gradio['penalty_alpha'] = gr.State({'penalty_alpha': shared.generate_params['penalty_alpha']})

    # Beam Search
    shared.gradio['num_beams'] = gr.State({'num_beams': shared.generate_params['num_beams']})
    shared.gradio['length_penalty'] = gr.State({'length_penalty': shared.generate_params['length_penalty']})
    shared.gradio['early_stopping'] = gr.State({'early_stopping': shared.generate_params['early_stopping']})

    # Mirostat for llama.cpp
    shared.gradio['mirostat_mode'] = gr.State({'mirostat_mode': shared.generate_params['mirostat_mode']})
    shared.gradio['mirostat_tau'] = gr.State({'mirostat_tau': shared.generate_params['mirostat_tau']})
    shared.gradio['mirostat_eta'] = gr.State({'mirostat_eta': shared.generate_params['mirostat_eta']})

    # Other
    shared.gradio['truncation_length'] = gr.State({'truncation_length': shared.settings['truncation_length']})
    shared.gradio['custom_stopping_strings'] = gr.State({'custom_stopping_strings': shared.settings['custom_stopping_strings']})

    shared.gradio['ban_eos_token'] = gr.State({'ban_eos_token': shared.settings['ban_eos_token']})
    shared.gradio['add_bos_token'] = gr.State({'add_bos_token': shared.settings['add_bos_token']})
    shared.gradio['skip_special_tokens'] = gr.State({'skip_special_tokens': shared.settings['skip_special_tokens']})
    
    shared.gradio['stream'] = gr.State({'stream': True})
'''

# function to set interface mode
# removed def set_interface_arguments(interface_mode, extensions, bool_active):

# extensive modification done to def create_interface(): refer to source code
def create_interface():
    # Defining some variables
    gen_events = []
    # default_preset = shared.settings | shared.generate_params
    # default_text = load_prompt(shared.settings['prompt'])
    title = 'MindEaseAi : meAI'

    # Authentication variables
    auth = None
    gradio_auth_creds = []
    if shared.args['gradio_auth']:
        gradio_auth_creds += [x.strip() for x in shared.args['gradio_auth'].strip('"').replace('\n', '').split(',') if
                              x.strip()]
    if shared.args['gradio_auth_path'] is not None:
        with open(shared.args['gradio_auth_path'], 'r', encoding="utf8") as file:
            for line in file.readlines():
                gradio_auth_creds += [x.strip() for x in line.split(',') if x.strip()]
    if gradio_auth_creds:
        auth = [tuple(cred.split(':')) for cred in gradio_auth_creds]

    # Importing the extension files and executing their setup() functions
    if shared.args['extensions'] is not None and len(shared.args['extensions']) > 0:
        extensions_module.load_extensions()

    # css/js strings
    # IF condition applied to show maincss or mainjs if chat mode is deselected. If function can be removed later as needed.
    css = ui.css if not shared.is_chat() else ui.css + ui.chat_css
    js = ui.main_js if not shared.is_chat() else ui.main_js + ui.chat_js
    css += apply_extensions('css')
    js += apply_extensions('js')

    # Gradio Array to prevent KeyErrors
    for k in shared.settings:
        shared.gradio[k] = shared.settings[k]

    # removed generate_params = load_preset_values(default_preset if not shared.args['flexgen'] else 'Naive' , return_dict=True)

    with gr.Blocks(css=css, analytics_enabled=False, title=title, theme=ui.theme) as shared.gradio['interface']:
        if Path("notification.mp3").exists():
            shared.gradio['audio_notification'] = gr.Audio(interactive=False, value="notification.mp3",
                                                           elem_id="audio_notification", visible=False)
            audio_notification_js = "document.querySelector('#audio_notification audio')?.play();"
        else:
            audio_notification_js = ""

        # Create chat mode interface
        if shared.is_chat():

            #shared.input_elements = ui.list_interface_input_elements(chat=True)

            '''shared.input_elements = shared.args | shared.settings
            removals = {'notebook', 'chat', 'chat_style_dir', 'character', 'model_dir', 'lora_dir', 'lora',
                        'model_menu', 'no_stream', 'settings_file', 'gpu_memory_secondary', 'no_cache', 'cache_capacity',
                        'gptq_loader_dir', 'quant_attn', 'warmup_attn', 'fused_mlp', 'flexgen', 'percent', 'compress_weight', 'pin_weight', 'listen',
                        'listen_port', 'share', 'auto_launch', 'gradio_auth', 'gradio_auth_path', 'api', 'api_blocking_port',
                        'api_streaming_port', 'public_api', 'dark_theme', 'autoload_model', 'max_new_tokens_min', 'max_new_tokens_max',
                        'truncation_length_max', 'chat_prompt_size_min', 'chat_prompt_size_max', 'chat_generation_attempts_max',
                        'chat_generation_attempts_min', 'default_extensions', 'chat_default_extensions', 'preset', 'prompt', 'initial'} #'cache', 'starts_with', 'extensions'
            for k in removals:
                shared.input_elements.pop(k)
            '''
            shared.gradio['interface_state'] = gr.State({k: None for k in shared.input_elements})
            shared.gradio['Chat input'] = gr.State()
            # shared.gradio['dummy'] = gr.State()

            with gr.Tab('Text generation', elem_id='main'):
                shared.gradio['display'] = gr.HTML(
                    value=chat_html_wrapper(shared.history['visible'], shared.gradio['name1'], shared.gradio['name2'],
                                            shared.gradio['mode'], shared.gradio['chat_style']))

                shared.gradio['textbox'] = gr.Textbox(label='Input', elem_id='textbox')

                with gr.Row():
                    shared.gradio['Stop'] = gr.Button('Stop', elem_id='stop')
                    shared.gradio['Generate'] = gr.Button('Generate', elem_id='Generate', variant='primary')
                    shared.gradio['Continue'] = gr.Button('Continue')

                # removed 2 gr.row() for impersonate, regenerate, remove last,
                # removed copy last reply, replace last reply, dummy message, dummy reply, 

                with gr.Row():
                    shared.gradio['Clear history'] = gr.Button('Clear history')
                    shared.gradio['Clear history-confirm'] = gr.Button('Confirm', variant='stop', visible=False)
                    shared.gradio['Clear history-cancel'] = gr.Button('Cancel', visible=False)

                # removed gr.row() for "start reply with: 'sure thing!' "
                # removed gr.Row() for mode(chat,chat-instruct,instruct) and chat_Style
                # removed gr.Tab() for chat settings(character, context, greeting, instruction template, etc)

             # removed gr.Tab() for parameters tab
             #state = shared.persistent_interface_state
             # settings_menu(default_preset)

                # removed notebook mode interface
                # removed default mode interface
                # removed Model tab
                # removed Training tab
                # removed Interface mode tab
        with gr.Tab("Interface mode", elem_id="interface-mode"):
            modes = ["chat"]
            current_mode = "chat"
            for mode in modes[1:]:
                if getattr(shared.args, mode):
                    current_mode = mode
                    break

            cmd_list = shared.args | shared.generate_params | shared.settings
            bool_list = sorted(
                [k for k in cmd_list if type(cmd_list[k]) is bool and k not in modes + ui.list_model_elements()])
            bool_active = [k for k in bool_list if vars(shared.args)[k]]

            with gr.Row():
                shared.gradio['interface_modes_menu'] = gr.Dropdown(choices=modes, value=current_mode, label="Mode")
                shared.gradio['toggle_dark_mode'] = gr.Button('Toggle dark/light mode', elem_classes="small-button")

            shared.gradio['extensions_menu'] = gr.CheckboxGroup(choices=utils.get_available_extensions(),
                                                                value=shared.args.extensions,
                                                                label="Available extensions",
                                                                info='Note that some of these extensions may require manually installing Python requirements through the command: pip install -r extensions/extension_name/requirements.txt')
            shared.gradio['bool_menu'] = gr.CheckboxGroup(choices=bool_list, value=bool_active,
                                                          label="Boolean command-line flags")
            shared.gradio['reset_interface'] = gr.Button("Apply and restart the interface")

        # chat mode event handlers
        if shared.is_chat():
            shared.input_params = [shared.gradio[k] for k in ['Chat input', 'start_with', 'interface_state']]
            clear_arr = [shared.gradio[k] for k in ['Clear history-confirm', 'Clear history', 'Clear history-cancel']]
            shared.reload_inputs = [shared.gradio[k] for k in ['name1', 'name2', 'mode', 'chat_style']]

            gen_events.append(shared.gradio['Generate'].click(
                print("shared.gradio from gen_events_append generate", shared.gradio),
                shared.gradio['interface_state']).then(
                lambda x: (x, ''), shared.gradio['textbox'], [shared.gradio['Chat input'], shared.gradio['textbox']], show_progress=False).then(
                chat.generate_chat_reply_wrapper, shared.input_params, shared.gradio['display'], show_progress=False).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False).then(
                lambda: None, None, None, _js=f"() => {{{audio_notification_js}}}")
            )

            gen_events.append(shared.gradio['textbox'].submit(
                shared.gradio['interface_state']).then(
                lambda x: (x, ''), shared.gradio['textbox'], [shared.gradio['Chat input'], shared.gradio['textbox']],
                show_progress=False).then(
                chat.generate_chat_reply_wrapper, shared.input_params, shared.gradio['display'],
                show_progress=False).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False).then(
                lambda: None, None, None, _js=f"() => {{{audio_notification_js}}}")
            )

            # removed gen_events.append(shared.gradio['Regenerate'].click()

            gen_events.append(shared.gradio['Continue'].click(
                shared.gradio['interface_state']).then(
                partial(chat.generate_chat_reply_wrapper, _continue=True), shared.input_params,
                shared.gradio['display'], show_progress=False).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False).then(
                lambda: None, None, None, _js=f"() => {{{audio_notification_js}}}")
            )

            # removed gen_events.append(shared.gradio['Impersonate'])
            # removed shared.gradio['Replace last reply'].click()
            # removed shared.gradio['Send dummy message'].click()
            # removed shared.gradio['Send dummy reply'].click()

            shared.gradio['Clear history-confirm'].click(
                lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None,
                clear_arr).then(
                chat.clear_chat_log, [shared.gradio[k] for k in ['greeting', 'mode']], None).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False).then(
                chat.redraw_html, shared.reload_inputs, shared.gradio['display'])

            shared.gradio['Stop'].click(
                stop_everything_event, None, None, queue=False,
                cancels=gen_events if shared.args['no_stream'] else None).then(
                chat.redraw_html, shared.reload_inputs, shared.gradio['display'])

            # removed shared.gradio['mode'].change()

            # shared.gradio['chat_style'].change(chat.redraw_html, shared.reload_inputs, shared.gradio['display'])
            # shared.gradio['instruction_template'].change(
            #    partial(chat.load_character, instruct=True), [shared.gradio[k] for k in ['instruction_template', 'name1_instruct', 'name2_instruct']], [shared.gradio[k] for k in ['name1_instruct', 'name2_instruct', 'dummy', 'dummy', 'context_instruct', 'turn_template']])

            # removed shared.gradio['upload_chat_history'].upload()

            # shared.gradio['Copy last reply'].click(chat.send_last_reply_to_input, None, shared.gradio['textbox'], show_progress=False)
            shared.gradio['Clear history'].click(
                lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)], None, clear_arr)
            shared.gradio['Clear history-cancel'].click(
                lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, clear_arr)
            # removed shared.gradio['Remove last'].click(

            # removed Save/delete a character - 11 functions

        # removed notebook/default modes event handlers

        shared.gradio['interface'].load(lambda: None, None, None, _js=f"() => {{{js}}}")
        if shared.settings['dark_theme']:
            shared.gradio['interface'].load(lambda: None, None, None,
                                            _js="() => document.getElementsByTagName('body')[0].classList.add('dark')")

        shared.gradio['interface'].load(partial(ui.apply_interface_values, {}, use_persistent=True), None, show_progress=False)

        # removed Extensions tabs
        # removed Extensions block

    # Launch the interface
    shared.gradio['interface'].queue()
    if shared.args['listen']:
        shared.gradio['interface'].launch(prevent_thread_lock=True, share=shared.args['share'],
                                          server_name=shared.args['listen_host'] or '0.0.0.0',
                                          server_port=shared.args['listen_port'], inbrowser=shared.args['auto_launch'],
                                          auth=auth)
    else:
        shared.gradio['interface'].launch(prevent_thread_lock=True, share=shared.args['share'],
                                          server_port=shared.args['listen_port'], inbrowser=shared.args['auto_launch'],
                                          auth=auth)


if __name__ == "__main__":
    # Loading custom settings
    # if shared.args['settings_file'] is not None and Path(shared.args['settings']).exists():
    #    settings_file = Path(shared.args['settings'])
    # elif Path('settings.yaml').exists():
    #  settings_file = Path('settings.yaml')
    # elif Path('settings.json').exists():
    #  settings_file = Path('settings.json')

    if shared.args['settings_file'] is not None and Path(
            shared.args['settings_file'] + os.sep + 'settings.yaml').exists():
        logger.info(f"Loading settings from {shared.args['settings_file']}...")
        file_contents = open(shared.args['settings_file'], 'r', encoding='utf-8').read()
        new_settings = json.loads(file_contents) if shared.args['settings_file'].suffix == "json" else yaml.safe_load(
            file_contents)
        for item in new_settings:
            shared.settings[item] = new_settings[item]


    # shared.model_config.move_to_end('.*', last=False)  # Move to the beginning

    # Default extensions modified - for no exts usage. 
    # imported get_available_extensions(), def natural_keys(text) and def atoi(text): from utils.py


    def atoi(text):
        return int(text) if text.isdigit() else text.lower()

    def natural_keys(text):
        return [atoi(c) for c in re.split(r'(\d+)', text)]

    def get_available_extensions():
        return sorted(set(map(lambda x: x.parts[1], Path('extensions').glob('*/script.py'))), key=natural_keys)

    extensions_module.available_extensions = get_available_extensions()
    if shared.is_chat():
        for extension in shared.settings['chat_default_extensions']:
            shared.args['extensions'] = shared.args['extensions'] or []
            if extension not in shared.args['extensions']:
                shared.args['extensions'].append(extension)
    else:
        for extension in shared.settings['default_extensions']:
            shared.args['extensions'] = shared.args['extensions'] or []
            if extension not in shared.args['extensions']:
                shared.args['extensions'].append(extension)

    # available_models = shared.args['model']

    # Model defined through --model
    # if shared.args['model'] is not None:
    #   shared.model_name = shared.args['model']

    # removed elif len(available_models) == 1:

    # removed elif shared.args['model_menu']: Select the model from a command-line menu

    # If any model has been selected, load it
    if shared.model_name != 'None':
        model_settings = get_model_specific_settings(shared.model_name)
        # shared.settings.update(model_settings)  # hijacking the interface defaults
        # update_model_parameters(model_settings, initial=True)  # hijacking the command-line arguments
        shared.settings.update(model_settings, initial=True)  # hijacking the interface defaults

        # Load the model
        print("model name is: " + shared.model_name)
        print(shared.model_config)
        shared.model, shared.tokenizer = load_model(shared.model_name)
        if shared.args['lora'] is not None:
            add_lora_to_model(shared.args['lora'])
        else:
            print("No LoRAs selected for loading.")

    # removed if shared.is_chat(): Force a character to be loaded
    if shared.is_chat():
        shared.persistent_interface_state.update({'mode':shared.settings['mode']})

    shared.generation_lock = Lock()
    # Launch the web UI
    create_interface()
''' # removed restart interface
    while True:
        time.sleep(0.5)
        if shared.need_restart:
            shared.need_restart = False
            time.sleep(0.5)
            shared.gradio['interface'].close()
            time.sleep(0.5)
            create_interface()
'''
