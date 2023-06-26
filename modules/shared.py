import argparse
from collections import OrderedDict
from pathlib import Path
import os

import yaml

from modules.logging_colors import logger

generation_lock = None
model = None
tokenizer = None
#model_name = "mayaeary_pygmalion-6b_dev-4bit-128g\config.json"
model_name = "mayaeary_pygmalion-6b_dev-4bit-128g\pygmalion-6b_dev-4bit-128g.safetensors"
model_type = None
lora_names = []
soft_prompt_tensor = None
soft_prompt = False

# Chat variables
history = {'internal': [], 'visible': []}
character = 'None'
stop_everything = False
processing_message = '*Is typing...*'

# UI elements (buttons, sliders, HTML, etc)
gradio = {}

# For keeping the values of UI elements on page reload
persistent_interface_state = {}

input_params = []  # Generation input parameters
reload_inputs = []  # Parameters for reloading the chat interface

# For restarting the interface
need_restart = False

settings = {
    'dark_theme': False,
    'autoload_model': True,
    'max_new_tokens': 400,
    'max_new_tokens_min': 2,
    'max_new_tokens_max': 2000,
    'seed': -1, #random
    'character': 'None',
    'name1': 'You',
    'name2': 'Joi',
    'context': 'This is a conversation with your Psychologist named Joi. Joi is a patient, kind and helpful individual who advises you how to effectively deal with your mental health issues. Joi will ask relevant questions about the users life and influence them to be more positive without explicitly saying so. Joi also does not take any physical actions. Joi is a psychologist and talks to the user professionally using their name. ',
    'greeting': "Welcome to mindEase ! My name is Joi and I'll be your guide on this journey.",
    'turn_template': '<|name1|><|name1-message|>\n<|name2|><|name2-message|>\n',
    'custom_stopping_strings': 'end conv',
    'stop_at_newline': False,
    'add_bos_token': True,
    'ban_eos_token': False,
    'skip_special_tokens': True,
    'truncation_length': 2048,
    'truncation_length_min': 0,
    'truncation_length_max': 8192,
    'mode': 'chat',
    'start_with': '',
    'chat_style': 'messenger',
    'instruction_template': 'None',
    #'chat-instruct_command': 'Continue the chat dialogue below. Write a single reply for the character "<|character|>".\n\n<|prompt|>',
    'chat_prompt_size': 2048,
    'chat_prompt_size_min': 0,
    'chat_prompt_size_max': 2048,
    'chat_generation_attempts': 2,
    'chat_generation_attempts_min': 1,
    'chat_generation_attempts_max': 10,
    'default_extensions': [],
    'chat_default_extensions': [], # ['gallery'],
    'preset': 'LLaMA-Precise',
    'prompt': 'QA',
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


#parser = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=54))
parser = argparse.ArgumentParser()

'''
SHARED FILE IS RESPONSIBLE FOR ALL CHANGES IN SETTINGS REGARDING MODEL, DIRECTORIES, ARGS, ETC.
ARGS LISTED BELOW ARE HARDCODED. CAN BE REVERTED BACK TO COMMAND LINE CODE VIA:
parser.add_argument('--ARG', TYPE, NARGS(IF REQUIRED - for joining multiple), HELP='HELP MESSAGE')

'''

# Basic settings -x- removed 11 add args
notebook = False         
chat = True
character = False
model_dir = ''              # type=string
lora_dir = None             # [], type=string nargs "+"
model_menu = None           # [], not required
no_stream = True            # check that it only applies to stopping generation 
settings_file = None        # directory of file that leads to settings.yaml or settings.json
extensions = []             # type=str, nargs="+"
verbose = True              # prints prompts to terminal


# Accelerate/transformers -x- removed 11 add args
cpu = False                 # use CPU ONLY to generate text
auto_devices = True         # automatically split model between available GPUs and CPUs
gpu_memory = 4              # type=string , nargs="+" in GiB, alternatively can set in MiB like 3500MiB
cpu_memory = ''             # type=string , same as above in GiB or MiB
disk = True                 # offloads remaining model layers to disk if model is too big -> creates disk cache on env creation and wipes it on termination
disk_cache_dir = "cache"    # type=string , defaults to cache directory
load_in_8bit = False        # 8-bit precision using BitsandBytes 
bf16 = False                # bloat16 precision, NVIDIA Ampere GPU required
no_cache = False            # no caching, reduces VRAM usage
#xformers = False           # memory efficient attention, miniscule increase in tokens/s -> needs explicit imports and installation of xformers
#sdp_attention = False      # torch 2.0 sdp attention for llama_attn_hijack -> use with xformers to marginally increase tokens/s
trust_remote_code = False   # for loading ChatGLM and Falcon models


# Accelerate 4-bit -x- removed 4 add args
load_in_4bit = True         # 4-bit precision with BitsAndBytes
compute_dtype= 'bfloat16'   # type=string , Valid options: bfloat16, float16, float32
quant_type = 'fp4'          # type=string , Valid options: nf4, fp4
use_double_quant = False    


# llama.cpp -x- removed 7 add args
threads = 0                 # type=int , default=0 , threads to use
n_batch = 512               # type=int , default=512 , Max prompt tokens in a batch when calling llama_eval
no_mmap = False             # setting to True prevents mmap from being used
mlock = False               # Memory-lock makes system to keep model in RAM
cache_capacity =''          # type=string , max capacity in ####MiB or #GiB , defaults to bytes without units
n_gpu_layers = 0            # type=int , default=0 , layer offload to GPU
n_ctx = 2048                # type=int , default=2048 , size of prompt context
llama_cpp_seed = 0          # type=int , default=0 (random) 


# GPTQ -x- removed 9 add args
# set according to mayaeary_pygmalion-6b_dev-4bit-128g
wbits = 4                   # type=int , default=0, Valid options = 2, 3, 4 and 8 bits supported for loading a pre-quantized model
model_type = 'gptj'         # type=string , Valid options = llama , opt , gptj
groupsize = 128             # type=int , default= -1  
pre_layer = []              # type=int , nargs="+" , sets the number of layers to allocate to GPU(s*), enables CPU offloading for 4-bit models
checkpoint = ''             # type=string , path to checkpoint file, automatically detects file if not set
monkey_patch = False        # monkey patch for LoRA applications on quantized models -> functionality not added yet
# TRITON GPU INFERENCE SUPPORT #
quant_attn = False          # for CUDA Triton support. -> enable for Triton GPU inference 
warmup_attn = False         # Triton warmup autotune
fused_mlp = False           # Triton fused mlp (multilayer perceptron)


# AutoGPTQ
parser.add_argument('--autogptq', action='store_true', help='Use AutoGPTQ for loading quantized models instead of the internal GPTQ loader.')
parser.add_argument('--triton', action='store_true', help='Use triton.')
parser.add_argument('--desc_act', action='store_true', help='For models that don\'t have a quantize_config.json, this parameter is used to define whether to set desc_act or not in BaseQuantizeConfig.')

# FlexGen
parser.add_argument('--flexgen', action='store_true', help='Enable the use of FlexGen offloading.')
parser.add_argument('--percent', type=int, nargs="+", default=[0, 100, 100, 0, 100, 0], help='FlexGen: allocation percentages. Must be 6 numbers separated by spaces (default: 0, 100, 100, 0, 100, 0).')
parser.add_argument("--compress-weight", action="store_true", help="FlexGen: activate weight compression.")
parser.add_argument("--pin-weight", type=str2bool, nargs="?", const=True, default=True, help="FlexGen: whether to pin weights (setting this to False reduces CPU memory by 20%%).")

# DeepSpeed
parser.add_argument('--deepspeed', action='store_true', help='Enable the use of DeepSpeed ZeRO-3 for inference via the Transformers integration.')
parser.add_argument('--nvme-offload-dir', type=str, help='DeepSpeed: Directory to use for ZeRO-3 NVME offloading.')
parser.add_argument('--local_rank', type=int, default=0, help='DeepSpeed: Optional argument for distributed setups.')

# RWKV
parser.add_argument('--rwkv-strategy', type=str, default=None, help='RWKV: The strategy to use while loading the model. Examples: "cpu fp32", "cuda fp16", "cuda fp16i8".')
parser.add_argument('--rwkv-cuda-on', action='store_true', help='RWKV: Compile the CUDA kernel for better performance.')

# Gradio
parser.add_argument('--listen', action='store_true', help='Make the web UI reachable from your local network.')
parser.add_argument('--listen-host', type=str, help='The hostname that the server will use.')
parser.add_argument('--listen-port', type=int, help='The listening port that the server will use.')
parser.add_argument('--share', action='store_true', help='Create a public URL. This is useful for running the web UI on Google Colab or similar.')
parser.add_argument('--auto-launch', action='store_true', default=False, help='Open the web UI in the default browser upon launch.')
parser.add_argument("--gradio-auth", type=str, help='set gradio authentication like "username:password"; or comma-delimit multiple like "u1:p1,u2:p2,u3:p3"', default=None)
parser.add_argument("--gradio-auth-path", type=str, help='Set the gradio authentication file path. The file should contain one or more user:password pairs in this format: "u1:p1,u2:p2,u3:p3"', default=None)

# API
parser.add_argument('--api', action='store_true', help='Enable the API extension.')
parser.add_argument('--api-blocking-port', type=int, default=5000, help='The listening port for the blocking API.')
parser.add_argument('--api-streaming-port', type=int,  default=5005, help='The listening port for the streaming API.')
parser.add_argument('--public-api', action='store_true', help='Create a public URL for the API using Cloudfare.')

# Multimodal
parser.add_argument('--multimodal-pipeline', type=str, default=None, help='The multimodal pipeline to use. Examples: llava-7b, llava-13b.')

args = parser.parse_args()
args_defaults = parser.parse_args([])

# Deprecation warnings for parameters that have been renamed
deprecated_dict = {}
for k in deprecated_dict:
    if getattr(args, k) != deprecated_dict[k][1]:
        logger.warning(f"--{k} is deprecated and will be removed. Use --{deprecated_dict[k][0]} instead.")
        setattr(args, deprecated_dict[k][0], getattr(args, k))

# Security warnings
if args.trust_remote_code:
    logger.warning("trust_remote_code is enabled. This is dangerous.")
if args.share:
    logger.warning("The gradio \"share link\" feature uses a proprietary executable to create a reverse tunnel. Use it with care.")


def add_extension(name):
    if args.extensions is None:
        args.extensions = [name]
    elif 'api' not in args.extensions:
        args.extensions.append(name)


# Activating the API extension
if args.api or args.public_api:
    add_extension('api')

# Activating the multimodal extension
if args.multimodal_pipeline is not None:
    add_extension('multimodal')


def is_chat():
    return args.chat


# removed Loading model-specific settings

# Applying user-defined model settings
# hardcoded path
model_config = {}
p = Path("C:"+ os.sep +"Users"+ os.sep +"suran"+ os.sep +"Desktop"+ os.sep +"School"+ os.sep +"MINDEASE_pg"
         + os.sep +"MindEase_pg"+ os.sep +"config-user.yaml" 
        )
#with Path(f'{args.model_dir}/config-user.yaml') as p:
if p.exists():
    user_config = yaml.safe_load(open(p, 'r').read())
    for k in user_config:
        if k in model_config:
            model_config[k].update(user_config[k])
        else:
            model_config[k] = user_config[k]

model_config = OrderedDict(model_config)

