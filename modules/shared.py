from collections import OrderedDict
from pathlib import Path
import os

import yaml

from modules.logging_colors import logger

generation_lock = None
model = None
tokenizer = None
model_name = "pygmalion-6b_dev-4bit-128g.safetensors"
model_type = 'gptj'
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

'''
------------------------------------------------------------------------------------------------
SHARED FILE IS RESPONSIBLE FOR ALL CHANGES IN SETTINGS REGARDING MODEL, DIRECTORIES, ARGS, ETC.
ARGS LISTED BELOW ARE HARDCODED. CAN BE REVERTED BACK TO COMMAND LINE CODE VIA:
parser.add_argument('--ARG', TYPE, NARGS(IF REQUIRED - for joining multiple), HELP='HELP MESSAGE')
------------------------------------------------------------------------------------------------

SETTINGS :
'''
args =  {
        # Basic settings -x- removed 11 add args
        'notebook': False,         
        'chat': True,
        'character': False,
        'model_dir': Path ("C:"+ os.sep +"Users"+ os.sep +"suran"+ os.sep +"text-generation-webui"+ os.sep +
                           "models"+ os.sep + "mayaeary_pygmalion-6b_dev-4bit-128g"
                           ),        # type=string
        'lora_dir': None,            # [], type=string nargs "+"
        'model_menu': None,          # [], not required
        'no_stream': True,           # default=True , check that it only applies to stopping generation 
        'settings_file': None,       # directory of file that leads to settings.yaml or settings.json
        'extensions': [],            # type=str, nargs="+"
        'verbose': True,             # toggle to print prompts to terminal

        # Accelerate/transformers -x- removed 11 add args
        'cpu':False,                 # toggle to use CPU ONLY to generate text
        'auto_devices': True,        # toggle to automatically split model between available GPUs and CPUs
        'gpu_memory': 4,             # type=string , nargs="+" in GiB, alternatively can set in MiB like 3500MiB
        'cpu_memory':'',             # type=string , same as above in GiB or MiB
        'disk': True,                # toggle for offloading remaining model layers to disk if model is too big -> creates disk cache on env creation and wipes it on termination
        'disk_cache_dir': 'cache',   # type=string , defaults to cache directory
        'load_in_8bit': False,       # toggle for 8-bit precision using BitsandBytes 
        'bf16': False,               # toggle for bloat16 precision, NVIDIA Ampere GPU required
        'no_cache': False,           # toggle for no caching, reduces VRAM usage
      # 'xformers': False,           # memory efficient attention, miniscule increase in tokens/s -> needs explicit imports and installation of xformers
      # 'sdp_attention': False,      # torch 2.0 sdp attention for llama_attn_hijack -> use with xformers to marginally increase tokens/s
        'trust_remote_code': False,  # toggle for loading ChatGLM and Falcon models

        # Accelerate 4-bit -x- removed 4 add args
        
        'load_in_4bit': True,        # 4-bit precision with BitsAndBytes
        'compute_dtype': 'bfloat16', # type=string , Valid options: bfloat16, float16, float32
        'quant_type': 'fp4',         # type=string , Valid options: nf4, fp4
        'use_double_quant': False,  

        # llama.cpp -x- removed 7 add args
        'threads': 0,                # type=int , default=0 , threads to use
        'n_batch': 512,              # type=int , default=512 , Max prompt tokens in a batch when calling llama_eval
        'no_mmap': False,            # toggle to prevents mmap from being used
        'mlock': False,              # toggle for Memory-lock which forces system to keep model in RAM
        'cache_capacity': '',        # type=string , max capacity in ####MiB or #GiB , defaults to bytes without units
        'n_gpu_layers': 0,           # type=int , default=0 , layer offload to GPU
        'n_ctx': 2048,               # type=int , default=2048 , size of prompt context
        'llama_cpp_seed': 0,         # type=int , default=0 (random) 

        # GPTQ -x- removed 9 add args
        # set according to mayaeary_pygmalion-6b_dev-4bit-128g
        'wbits': 4,                  # type=int , default=0, Valid options = 2, 3, 4 and 8 bits supported for loading a pre-quantized model
        'model_type': 'gptj',        # type=string , Valid options = llama , opt , gptj
        'groupsize': 128,            # type=int , default= -1  
        'pre_layer': [],             # type=int , nargs="+" , sets the number of layers to allocate to GPU(s*), enables CPU offloading for 4-bit models
        'checkpoint': '',            # type=string , path to checkpoint file, automatically detects file if not set
      # 'monkey_patch': False,       # toggle for monkey patch for LoRA applications on quantized models -> functionality not added yet
        'gptq_loader_dir': Path("C:"+ os.sep +"Users"+ os.sep +"suran"+ os.sep +"Desktop"+ os.sep 
                                +"School"+ os.sep +"MINDEASE_pg" + os.sep +"MindEase_pg"+ os.sep + "repositories" + os.sep + "GPTQ-for-LLaMa-cuda"
                            ),       # hardcoded path for gptq repo directory on local system
        # TRITON GPU INFERENCE SUPPORT with GPTQ
        'quant_attn': False,         # toggle for CUDA Triton support. -> enable for Triton GPU inference 
        'warmup_attn': False,        # toggle for Triton warmup autotune
        'fused_mlp': False,          # toggle for Triton fused mlp (multilayer perceptron)

        # FlexGen -x- removed 4  add args
        'flexgen': False,            # toggle to use Flexgen offloading
        'percent': [],               # type=int , nargs="+" , default=[0, 100, 100, 0, 100, 0] ,  allocation percentages -> must be 6 numbers separated by spaces
        'compress_weight': False,    # toggle to activate weight compression
        'pin_weight': False,         # toggle to pin weights , const=True , default=True , False setting reduces CPU memory by 20% 

        # Gradio -x- removed 7 add args
        'listen': True,              # toggle to webui reachable from own local network
        'listen_host': '',           # type=string , hostname the server will use
        'listen_port': '',           # type=int , listening port that the server will use
        'share': False,               # default=True , creates public URL , for running webui on Colab or similar
        'auto_launch': True,         # default=False , opens webui in default browser upon launch
        'gradio_auth': None,         # type=string , default=None , set gradio authentication like "username:password"; or comma-delimit multiple like "u1:p1,u2:p2,u3:p3"
        'gradio_auth_path': None,    # type=string , default=None , gradio authentication file path. The file should contain one or more user:password pairs in this format: "u1:p1,u2:p2,u3:p3"

        # add api folder under extensions
        # API -x- removed 4 add args
        'api': False,                # toggle to enable api extension 
        'api_blocking_port': 5000,   # type=int , default=5000 , listening port for the blocking api
        'api_streaming_port': 5005,  # type=int , default=5005 , listening port for streaming api
        'public_api': False          # default=True , create public URL for api using Cloudfare
    }



# Deprecation warnings for parameters that have been renamed
deprecated_dict = {}
for k in deprecated_dict:
    if getattr(args, k) != deprecated_dict[k][1]:
        logger.warning(f"--{k} is deprecated and will be removed. Use --{deprecated_dict[k][0]} instead.")
        setattr(args, deprecated_dict[k][0], getattr(args, k))

# Security warnings
if args['trust_remote_code'] is True:
    logger.warning("trust_remote_code is enabled. This is dangerous.")
if args['share']:
    logger.warning("The gradio \"share link\" feature uses a proprietary executable to create a reverse tunnel. Use it with care.")

def add_extension(name):
    if args['extensions'] is None:
        args['extensions'] = [name]
    elif 'api' not in args['extensions']:
        args['extensions'].append(name)


# Activating the API extension
if args['api'] or args['public_api']:
    add_extension('api')


def is_chat():
    return args['chat']

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

'''CODE FOR FURTHER DEVELOPMENT 
-> NEEDS DEPENDENCIES AND ADDITIONAL FILES FOR EXTENSIONS 
#import argparse

# removed def str2bool(v) for parsing CLI into proper values
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

## apply before declaring add args
#parser = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=54))
## apply after declaring add args
#args = parser.parse_args()
#args_defaults = parser.parse_args([])
        

# removed Activating the multimodal extension
if args.multimodal_pipeline is not None:
    add_extension('multimodal')

FROM ARGS: 
 
functionality not added in yet -> modify code and load file : AutoGPTQ_loader.py under modules for use 
# AutoGPTQ -x- removed 3 add args
autogptq = False            # use AutoGPTQ for loading quantized models instead of using internal GPTQ loader
triton = False              
desc_act = False            # used to define whether to set desc_act or not in BaseQuantizeConfig for models without quantize_config.json file

-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-

removed all instances for linux specific deepspeed -> modify models.py and text_generation.py code to reintegrate
# DeepSpeed -x- removed 3 add args
deepspeed = True            # toggle for inference via Transformers integration using DeepSpeed ZeRO-3
nvme_offload_dir = ''       # type=string , directory for ZeRO-3 NVME offloading
local_rank = 0              # type=int , default=0 , optional argument for distributed setups

-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-

removed all instances of rwkv -> modify text-genertaion.py, models.py and load file RWKV.py under modules to add support for this model type
# RWKV
rwkv_strategy = ''          # type=string , default=None , strategy to use while loading the model. Examples: "cpu fp32", "cuda fp16", "cuda fp16i8
rwkv_cuda_on = True         # toggle to compile CUDA for better performance

-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-

removing support for multimodality (text+images) to text-generation-webui. -> modify shared.py and add extension for use
# Multimodal -x- removed 1 add arg
multimodal_pipeline = None   # type=string , default=None , multimodal pipeline to use. Examples: llava-7b, llava-13b.

'''