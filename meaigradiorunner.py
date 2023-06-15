import os
import warnings
import requests

from gradioui_modules.logging_colors import logger

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
from gradioui_modules import chat, shared, ui, utils # removed training
from gradioui_modules.extensions import apply_extensions
from gradioui_modules.html_generator import chat_html_wrapper
#removed LoRA import
from gradioui_modules.models import load_model, load_soft_prompt, unload_model
from gradioui_modules.text_generation import (generate_reply_wrapper, get_encoded_length, stop_everything_event)

# removed def load_model_wrapper function related to model loading, loading error if no model, updating settings for manual load

# removed Function -def load_lora_wrapper(selected-loras): 

# removed def load_preset_values function to load parameters manually
    # removed 3 functions to set params from presets menu

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

# function to set interface mode
def set_interface_arguments(interface_mode, extensions, bool_active):
    modes = ["default", "notebook", "chat", "cai_chat"]
    cmd_list = vars(shared.args)
    bool_list = [k for k in cmd_list if type(cmd_list[k]) is bool and k not in modes]

    shared.args.extensions = extensions
    for k in modes[1:]:
        setattr(shared.args, k, False)
    if interface_mode != "default":
        setattr(shared.args, interface_mode, True)

    for k in bool_list:
        setattr(shared.args, k, False)
    for k in bool_active:
        setattr(shared.args, k, True)

    shared.need_restart = True

# extensive modification done to def create_interface(): refer to source code 
def create_interface():

    # Defining some variables
    gen_events = []
    default_preset = shared.settings['preset']
    default_text = load_prompt(shared.settings['prompt'])
    title = 'Text generation web UI'

    # Authentication variables
    auth = None
    gradio_auth_creds = []
    if shared.args.gradio_auth:
        gradio_auth_creds += [x.strip() for x in shared.args.gradio_auth.strip('"').replace('\n', '').split(',') if x.strip()]
    if shared.args.gradio_auth_path is not None:
        with open(shared.args.gradio_auth_path, 'r', encoding="utf8") as file:
            for line in file.readlines():
                gradio_auth_creds += [x.strip() for x in line.split(',') if x.strip()]
    if gradio_auth_creds:
        auth = [tuple(cred.split(':')) for cred in gradio_auth_creds]

    # Importing the extension files and executing their setup() functions
    if shared.args.extensions is not None and len(shared.args.extensions) > 0:
        extensions_module.load_extensions()

    # css/js strings
    css = ui.css if not shared.is_chat() else ui.css + ui.chat_css
    js = ui.main_js if not shared.is_chat() else ui.main_js + ui.chat_js
    css += apply_extensions('css')
    js += apply_extensions('js')

    with gr.Blocks(css=css, analytics_enabled=False, title=title, theme=ui.theme) as shared.gradio['interface']:
        if Path("notification.mp3").exists():
            shared.gradio['audio_notification'] = gr.Audio(interactive=False, value="notification.mp3", elem_id="audio_notification", visible=False)
            audio_notification_js = "document.querySelector('#audio_notification audio')?.play();"
        else:
            audio_notification_js = ""

        # Create chat mode interface
        if shared.is_chat():
            shared.input_elements = ui.list_interface_input_elements(chat=True)
            shared.gradio['interface_state'] = gr.State({k: None for k in shared.input_elements})
            shared.gradio['Chat input'] = gr.State()
            #shared.gradio['dummy'] = gr.State()

            with gr.Tab('Text generation', elem_id='main'):
                shared.gradio['display'] = gr.HTML(value=chat_html_wrapper(shared.history['visible'], shared.settings['name1'], shared.settings['name2'], 'chat', 'cai-chat'))
                shared.gradio['textbox'] = gr.Textbox(label='Input')
                with gr.Row():
                    shared.gradio['Stop'] = gr.Button('Stop', elem_id='stop')
                    shared.gradio['Generate'] = gr.Button('Generate', elem_id='Generate', variant='primary')
                    shared.gradio['Continue'] = gr.Button('Continue')

           # removed 2 gr.row() for impersonate, regenerate, remove last, copy and replace last reply, dummy message, dummy reply, 

                with gr.Row():
                    shared.gradio['Clear history'] = gr.Button('Clear history')
                    shared.gradio['Clear history-confirm'] = gr.Button('Confirm', variant='stop', visible=False)
                    shared.gradio['Clear history-cancel'] = gr.Button('Cancel', visible=False)

           # removed gr.row() for "start reply with: 'sure thing!' "
            
           # removed gr.Row() for chat mode(chat,chat-instruct,instruct) and chat_Style 
            
           # removed gr.Tab() for chat settings(character, context, greeting, instruction template, etc)

           # removed gr.Tab() for parameters tab
            
           # create_settings_menus(default_preset)


        # removed notebook mode interface
        # removed default mode interface


        # removed Model tab
        
        # removed Training tab
        
        # removed Interface mode tab
        
        # chat mode event handlers
        if shared.is_chat():
            shared.input_params = [shared.gradio[k] for k in ['Chat input', 'start_with', 'interface_state']]
            clear_arr = [shared.gradio[k] for k in ['Clear history-confirm', 'Clear history', 'Clear history-cancel']]
            shared.reload_inputs = [shared.gradio[k] for k in ['name1', 'name2', 'mode', 'chat_style']]

            gen_events.append(shared.gradio['Generate'].click(
                ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
                lambda x: (x, ''), shared.gradio['textbox'], [shared.gradio['Chat input'], shared.gradio['textbox']], show_progress=False).then(
                chat.generate_chat_reply_wrapper, shared.input_params, shared.gradio['display'], show_progress=False).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False).then(
                lambda: None, None, None, _js=f"() => {{{audio_notification_js}}}")
            )

            gen_events.append(shared.gradio['textbox'].submit(
                ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
                lambda x: (x, ''), shared.gradio['textbox'], [shared.gradio['Chat input'], shared.gradio['textbox']], show_progress=False).then(
                chat.generate_chat_reply_wrapper, shared.input_params, shared.gradio['display'], show_progress=False).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False).then(
                lambda: None, None, None, _js=f"() => {{{audio_notification_js}}}")
            )

            gen_events.append(shared.gradio['Regenerate'].click(
                ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
                partial(chat.generate_chat_reply_wrapper, regenerate=True), shared.input_params, shared.gradio['display'], show_progress=False).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False).then(
                lambda: None, None, None, _js=f"() => {{{audio_notification_js}}}")
            )

            gen_events.append(shared.gradio['Continue'].click(
                ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
                partial(chat.generate_chat_reply_wrapper, _continue=True), shared.input_params, shared.gradio['display'], show_progress=False).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False).then(
                lambda: None, None, None, _js=f"() => {{{audio_notification_js}}}")
            )

            gen_events.append(shared.gradio['Impersonate'].click(
                ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
                lambda x: x, shared.gradio['textbox'], shared.gradio['Chat input'], show_progress=False).then(
                chat.impersonate_wrapper, shared.input_params, shared.gradio['textbox'], show_progress=False).then(
                lambda: None, None, None, _js=f"() => {{{audio_notification_js}}}")
            )

            shared.gradio['Replace last reply'].click(
                chat.replace_last_reply, shared.gradio['textbox'], None).then(
                lambda: '', None, shared.gradio['textbox'], show_progress=False).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False).then(
                chat.redraw_html, shared.reload_inputs, shared.gradio['display'])

            shared.gradio['Send dummy message'].click(
                chat.send_dummy_message, shared.gradio['textbox'], None).then(
                lambda: '', None, shared.gradio['textbox'], show_progress=False).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False).then(
                chat.redraw_html, shared.reload_inputs, shared.gradio['display'])

            shared.gradio['Send dummy reply'].click(
                chat.send_dummy_reply, shared.gradio['textbox'], None).then(
                lambda: '', None, shared.gradio['textbox'], show_progress=False).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False).then(
                chat.redraw_html, shared.reload_inputs, shared.gradio['display'])

            shared.gradio['Clear history-confirm'].click(
                lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, clear_arr).then(
                chat.clear_chat_log, [shared.gradio[k] for k in ['greeting', 'mode']], None).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False).then(
                chat.redraw_html, shared.reload_inputs, shared.gradio['display'])

            shared.gradio['Stop'].click(
                stop_everything_event, None, None, queue=False, cancels=gen_events if shared.args.no_stream else None).then(
                chat.redraw_html, shared.reload_inputs, shared.gradio['display'])

            shared.gradio['mode'].change(
                lambda x: gr.update(visible=x != 'instruct'), shared.gradio['mode'], shared.gradio['chat_style'], show_progress=False).then(
                chat.redraw_html, shared.reload_inputs, shared.gradio['display'])

            shared.gradio['chat_style'].change(chat.redraw_html, shared.reload_inputs, shared.gradio['display'])
            shared.gradio['instruction_template'].change(
                partial(chat.load_character, instruct=True), [shared.gradio[k] for k in ['instruction_template', 'name1_instruct', 'name2_instruct']], [shared.gradio[k] for k in ['name1_instruct', 'name2_instruct', 'dummy', 'dummy', 'context_instruct', 'turn_template']])

            shared.gradio['upload_chat_history'].upload(
                chat.load_history, [shared.gradio[k] for k in ['upload_chat_history', 'name1', 'name2']], None).then(
                chat.redraw_html, shared.reload_inputs, shared.gradio['display'])

            shared.gradio['Copy last reply'].click(chat.send_last_reply_to_input, None, shared.gradio['textbox'], show_progress=False)
            shared.gradio['Clear history'].click(lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)], None, clear_arr)
            shared.gradio['Clear history-cancel'].click(lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, clear_arr)
            shared.gradio['Remove last'].click(
                chat.remove_last_message, None, shared.gradio['textbox'], show_progress=False).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False).then(
                chat.redraw_html, shared.reload_inputs, shared.gradio['display'])

            # Save/delete a character
            shared.gradio['save_character'].click(
                lambda x: x, shared.gradio['name2'], shared.gradio['save_character-filename'], show_progress=True).then(
                lambda: [gr.update(visible=True)] * 3, None, [shared.gradio[k] for k in ['save_character-filename', 'save_character-confirm', 'save_character-cancel']], show_progress=False)

            shared.gradio['save_character-cancel'].click(
                lambda: [gr.update(visible=False)] * 3, None, [shared.gradio[k] for k in ['save_character-filename', 'save_character-confirm', 'save_character-cancel']], show_progress=False)

            shared.gradio['save_character-confirm'].click(
                partial(chat.save_character, instruct=False), [shared.gradio[k] for k in ['name2', 'greeting', 'context', 'character_picture', 'save_character-filename']], None).then(
                lambda: [gr.update(visible=False)] * 3, None, [shared.gradio[k] for k in ['save_character-filename', 'save_character-confirm', 'save_character-cancel']], show_progress=False).then(
                lambda x: x, shared.gradio['save_character-filename'], shared.gradio['character_menu'])

            shared.gradio['delete_character'].click(
                lambda: [gr.update(visible=True)] * 2, None, [shared.gradio[k] for k in ['delete_character-confirm', 'delete_character-cancel']], show_progress=False)

            shared.gradio['delete_character-cancel'].click(
                lambda: [gr.update(visible=False)] * 2, None, [shared.gradio[k] for k in ['delete_character-confirm', 'delete_character-cancel']], show_progress=False)

            shared.gradio['delete_character-confirm'].click(
                partial(chat.delete_character, instruct=False), shared.gradio['character_menu'], None).then(
                lambda: gr.update(choices=utils.get_available_characters()), outputs=shared.gradio['character_menu']).then(
                lambda: 'None', None, shared.gradio['character_menu']).then(
                lambda: [gr.update(visible=False)] * 2, None, [shared.gradio[k] for k in ['delete_character-confirm', 'delete_character-cancel']], show_progress=False)

            shared.gradio['download_button'].click(lambda x: chat.save_history(x, timestamp=True), shared.gradio['mode'], shared.gradio['download'])
            shared.gradio['Upload character'].click(chat.upload_character, [shared.gradio['upload_json'], shared.gradio['upload_img_bot']], [shared.gradio['character_menu']])
            shared.gradio['character_menu'].change(
                partial(chat.load_character, instruct=False), [shared.gradio[k] for k in ['character_menu', 'name1', 'name2']], [shared.gradio[k] for k in ['name1', 'name2', 'character_picture', 'greeting', 'context', 'dummy']]).then(
                chat.redraw_html, shared.reload_inputs, shared.gradio['display'])

            shared.gradio['upload_img_tavern'].upload(chat.upload_tavern_character, [shared.gradio['upload_img_tavern'], shared.gradio['name1'], shared.gradio['name2']], [shared.gradio['character_menu']])
            shared.gradio['your_picture'].change(
                chat.upload_your_profile_picture, shared.gradio['your_picture'], None).then(
                partial(chat.redraw_html, reset_cache=True), shared.reload_inputs, shared.gradio['display'])

        # removed notebook/default modes event handlers
        
        shared.gradio['interface'].load(lambda: None, None, None, _js=f"() => {{{js}}}")
        if shared.settings['dark_theme']:
            shared.gradio['interface'].load(lambda: None, None, None, _js="() => document.getElementsByTagName('body')[0].classList.add('dark')")

        shared.gradio['interface'].load(partial(ui.apply_interface_values, {}, use_persistent=True), None, [shared.gradio[k] for k in ui.list_interface_input_elements(chat=shared.is_chat())], show_progress=False)

        # removed Extensions tabs

        # removed Extensions block
        
    # Launch the interface
    shared.gradio['interface'].queue()
    if shared.args.listen:
        shared.gradio['interface'].launch(prevent_thread_lock=True, share=shared.args.share, server_name=shared.args.listen_host or '0.0.0.0', server_port=shared.args.listen_port, inbrowser=shared.args.auto_launch, auth=auth)
    else:
        shared.gradio['interface'].launch(prevent_thread_lock=True, share=shared.args.share, server_port=shared.args.listen_port, inbrowser=shared.args.auto_launch, auth=auth)


if __name__ == "__main__":
    # Loading custom settings
    settings_file = None
    if shared.args.settings is not None and Path(shared.args.settings).exists():
        settings_file = Path(shared.args.settings)
    # elif Path('settings.yaml').exists():
      #  settings_file = Path('settings.yaml')
    # elif Path('settings.json').exists():
      #  settings_file = Path('settings.json')

    if settings_file is not None:
        logger.info(f"Loading settings from {settings_file}...")
        file_contents = open(settings_file, 'r', encoding='utf-8').read()
        new_settings = json.loads(file_contents) if settings_file.suffix == "json" else yaml.safe_load(file_contents)
        for item in new_settings:
            shared.settings[item] = new_settings[item]

    # Set model settings based on settings file
    shared.model_config['.*'] = {
        'wbits': 'None',
        'model_type': 'None',
        'groupsize': 'None',
        'pre_layer': 0,
        'mode': shared.settings['mode'],
        'skip_special_tokens': shared.settings['skip_special_tokens'],
        'custom_stopping_strings': shared.settings['custom_stopping_strings'],
    }

    shared.model_config.move_to_end('.*', last=False)  # Move to the beginning

    # Default extensions
    extensions_module.available_extensions = utils.get_available_extensions()
    if shared.is_chat():
        for extension in shared.settings['chat_default_extensions']:
            shared.args.extensions = shared.args.extensions or []
            if extension not in shared.args.extensions:
                shared.args.extensions.append(extension)
    else:
        for extension in shared.settings['default_extensions']:
            shared.args.extensions = shared.args.extensions or []
            if extension not in shared.args.extensions:
                shared.args.extensions.append(extension)

    available_models = utils.get_available_models()

    # Model defined through --model
    if shared.args.model is not None:
        shared.model_name = shared.args.model

    # Only one model is available
    elif len(available_models) == 1:
        shared.model_name = available_models[0]

    # Select the model from a command-line menu
    elif shared.args.model_menu:
        if len(available_models) == 0:
            logger.error('No models are available! Please download at least one.')
            sys.exit(0)
        else:
            print('The following models are available:\n')
            for i, model in enumerate(available_models):
                print(f'{i+1}. {model}')

            print(f'\nWhich one do you want to load? 1-{len(available_models)}\n')
            i = int(input()) - 1
            print()

        shared.model_name = available_models[i]

    # If any model has been selected, load it
    if shared.model_name != 'None':
        model_settings = get_model_specific_settings(shared.model_name)
        #shared.settings.update(model_settings)  # hijacking the interface defaults
        #update_model_parameters(model_settings, initial=True)  # hijacking the command-line arguments
        shared.settings.update(model_settings,initial=True)  # hijacking the interface defaults
        

        # Load the model
        shared.model, shared.tokenizer = load_model(shared.model_name)
        #if shared.args.lora:
        #    add_lora_to_model(shared.args.lora)

    # Force a character to be loaded
    if shared.is_chat():
        shared.persistent_interface_state.update({
            'mode': shared.settings['mode'],
            'character_menu': shared.args.character or shared.settings['character'],
            'instruction_template': shared.settings['instruction_template']
        })

    shared.generation_lock = Lock()
    # Launch the web UI
    create_interface()
    while True:
        time.sleep(0.5)
        if shared.need_restart:
            shared.need_restart = False
            time.sleep(0.5)
            shared.gradio['interface'].close()
            time.sleep(0.5)
            create_interface()
