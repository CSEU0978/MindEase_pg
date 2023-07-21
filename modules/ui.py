from pathlib import Path

import gradio as gr
import torch

from modules import shared

with open(Path(__file__).resolve().parent / '../frontend/ChatBotcss/main.css', 'r') as f:
    css = f.read()
with open(Path(__file__).resolve().parent / '../frontend/ChatBotcss/chat_style-messenger.css', 'r') as f:
    chat_css = f.read()
with open(Path(__file__).resolve().parent / '../chatBotjs/main.js', 'r') as f:
    main_js = f.read()
with open(Path(__file__).resolve().parent / '../chatBotjs/chat.js', 'r') as f:
    chat_js = f.read()

# refresh_symbol = '\U0001f504'  # 🔄
# delete_symbol = '🗑️'
# save_symbol = '💾'

theme = gr.themes.Default(
    font=['Helvetica', 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
).set(
    border_color_primary='#c5c5d2',
    button_large_padding='6px 12px',
    body_text_color_subdued='#484848',
    background_fill_secondary='#eaeaea'
)


def list_model_elements():
    elements = ['cpu_memory', 'auto_devices', 'disk', 'cpu', 'bf16', 'load_in_8bit', 'trust_remote_code',
                 'load_in_4bit', 'compute_dtype', 'quant_type', 'use_double_quant', 'wbits', 'groupsize', 'model_type', 
                 'pre_layer', 'autogptq', 'triton', 'desc_act', 'threads', 'n_batch', 'no_mmap', 'mlock', 'n_gpu_layers',
                   'n_ctx', 'llama_cpp_seed']
    for i in range(torch.cuda.device_count()):
        elements.append(f'gpu_memory_{i}')

    return elements


def list_model_setting_elements(chat=True):
    elements = [ 'max_new_tokens', 'seed', 'temperature', 'top_p', 'top_k', 'typical_p', 'epsilon_cutoff', 
                'eta_cutoff', 'repetition_penalty', 'encoder_repetition_penalty', 'no_repeat_ngram_size',
                  'min_length', 'do_sample', 'penalty_alpha', 'num_beams', 'length_penalty', 'early_stopping',
                    'mirostat_mode', 'mirostat_tau', 'mirostat_eta', 'add_bos_token', 'ban_eos_token', 
                    'truncation_length', 'custom_stopping_strings', 'skip_special_tokens', 'stream', 'tfs', 'top_a'] 
    if chat:
        elements += ['name1', 'name2', 'greeting', 'context', 'chat_prompt_size', 'chat_generation_attempts',
                      'stop_at_newline', 'mode', 'instruction_template', 'turn_template', 'chat_style']
                    # removed 'character_menu', 'name1_instruct', 'name2_instruct', 'context_instruct', 'chat-instruct_command'
    elements += list_model_elements()
    return elements


def list_interface_input_elements(chat=True):
    if chat:
        elements = ['max_new_tokens', 'seed', 'temperature', 'top_p', 'top_k', 'typical_p', 'epsilon_cutoff', 'eta_cutoff',
                    'repetition_penalty', 'encoder_repetition_penalty', 'no_repeat_ngram_size', 'min_length', 'do_sample',
                    'penalty_alpha', 'num_beams', 'length_penalty', 'early_stopping', 'mirostat_mode', 'mirostat_tau', 'mirostat_eta',
                    'add_bos_token', 'ban_eos_token', 'truncation_length', 'custom_stopping_strings', 'skip_special_tokens',
                    'preset_menu', 'stream', 'tfs', 'top_a', 'name1', 'name2', 'greeting', 'context', 'chat_prompt_size',
                    'chat_generation_attempts', 'stop_at_newline', 'mode', 'instruction_template', 'character_menu', 'name1_instruct',
                    'name2_instruct', 'context_instruct', 'turn_template', 'chat_style', 'chat-instruct_command']

    return elements


def gather_interface_values(*args: object) -> object:
    output = {}
    for i, element in enumerate(shared.input_elements):
        output[element] = args[i]

    shared.persistent_interface_state = output
    return output


def apply_interface_values(state, use_persistent=True):
    if use_persistent:
        state = shared.persistent_interface_state

    elements = {'max_new_tokens', 'seed', 'temperature', 'top_p', 'top_k', 'typical_p', 'epsilon_cutoff',
                'eta_cutoff', 'repetition_penalty', 'encoder_repetition_penalty', 'no_repeat_ngram_size',
                'min_length', 'do_sample', 'penalty_alpha', 'num_beams', 'length_penalty', 'early_stopping',
                'mirostat_mode', 'mirostat_tau', 'mirostat_eta', 'add_bos_token', 'ban_eos_token',
                'truncation_length', 'custom_stopping_strings', 'skip_special_tokens', 'stream', 'tfs', 'top_a',
                'name1', 'name2', 'greeting', 'context', 'chat_prompt_size', 'chat_generation_attempts',
                'stop_at_newline', 'mode', 'instruction_template', 'turn_template', 'chat_style',
                'character_menu', 'name1_instruct', 'name2_instruct', 'context_instruct', 'chat-instruct_command'
                }

    for key, value in shared.generate_params:
        if key == elements[key]:
            elements[key] = shared.generate_params[key]

    for key, value in shared.args:
        if key == elements[key]:
            elements[key] = shared.args[key]

    for key, value in shared.settings:
        if key == elements[key]:
            elements[key] = shared.settings[key]

    print("elements from apply interface values", elements)
    if len(state) == 0:
        return [gr.update() for k in elements]  # Dummy, small brain, do nothing
    else:
        return [state[k] if k in state else gr.update() for k in elements]


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"


# removed def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id):
# removed def create_delete_button(**kwargs):
# removed def create_save_button(**kwargs):