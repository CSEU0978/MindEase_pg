import gc
import json
import os
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import transformers
from accelerate import infer_auto_device_map, init_empty_weights
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM, AutoTokenizer,
                          BitsAndBytesConfig, LlamaTokenizer)

from modules import shared
from modules import sampler_hijack #DOUBLE CHECK THESE FILES 
from modules.logging_colors import logger # to differentiate input and output/ bot and user

transformers.logging.set_verbosity_error()

local_rank = None

# removed if shared.args.deepspeed as its only for linux

#tail-free and top-a sampling patch
sampler_hijack.hijack_samplers()

# Some models require special treatment in various parts of the code.
# This function detects model name for def load_model 
def find_model_type(model_name):
    path_to_model = Path(str(shared.args['model_dir']) + os.sep + shared.model_name)
    print(" path to model:  " , path_to_model)
    if not path_to_model.exists():
        return 'None'
    # convert model name to lowercase and resume search for model for def load_model
    model_name_lower = model_name.lower()
    # removed if re.match('.*rwkv.*\.pth', model_name_lower):
    #    return 'rwkv'
    if len(list(path_to_model.glob('*ggml*.bin'))) > 0:
        return 'llamacpp'
    elif re.match('.*ggml.*\.bin', model_name_lower):
        return 'llamacpp'
    elif 'chatglm' in model_name_lower:
        return 'chatglm'
    elif 'galactica' in model_name_lower:
        return 'galactica'
    elif 'llava' in model_name_lower:
        return 'llava'
    elif 'oasst' in model_name_lower:
        return 'oasst'
    elif any((k in model_name_lower for k in ['gpt4chan', 'gpt-4chan'])):
        return 'gpt4chan'
    else:
        config = AutoConfig.from_pretrained(path_to_model, trust_remote_code=shared.args['trust_remote_code'])
        # Not a "catch all", but fairly accurate
        if config.to_dict().get("is_encoder_decoder", False):
            return 'HF_seq2seq'
        else:
            return 'HF_generic'


def load_model(model_name):
    logger.info(f"Loading {model_name}...")
    t0 = time.time()

    shared.model_type = find_model_type(model_name)
    #if shared.model_type == 'None':
    #    logger.error('The path to the model does not exist. Exiting.')
    #    return None, None

    # removed if shared.args.autogptq:
    # makeshift modification to load gptj style model with wbits specified settings 
    if shared.args['wbits'] > 0 or shared.model_type == 'gptj':
        load_func = GPTQ_loader
    elif shared.model_type == 'llamacpp':
        load_func = llamacpp_loader
    # removed elif shared.model_type == 'rwkv':
    elif shared.args['flexgen']:
        load_func = flexgen_loader
    else:
        load_func = huggingface_loader

    output = load_func(model_name)
    if type(output) is tuple:
        model, tokenizer = output
    else:
        model = output
        if model is None:
            return None, None
        else:
            tokenizer = load_tokenizer(model_name, model)

    # removed Hijack attention with xformers or sdp_attention
   
    logger.info(f"Loaded the model in {(time.time()-t0):.2f} seconds.\n")
    return model, tokenizer

def load_tokenizer(model_name, model):
    tokenizer = None
    if shared.model_type == 'gpt4chan' and Path(f"{shared.args['model_dir']}"+ os.sep +"gpt-j-6B"+os.sep).exists():
        tokenizer = AutoTokenizer.from_pretrained(Path(f"{shared.args['model_dir']}"+ os.sep +"gpt-j-6B" + os.sep))
    elif type(model) is transformers.LlamaForCausalLM or "LlamaGPTQForCausalLM" in str(type(model)):
        # Try to load an universal LLaMA tokenizer
        if shared.model_type not in ['llava', 'oasst']:
            for p in [Path(f"{shared.args['model_dir']}"+ os.sep +"llama-tokenizer"+ os.sep), Path(f"{shared.args['model_dir']}"+ os.sep +"oobabooga_llama-tokenizer"+ os.sep)]:
                if p.exists():
                    logger.info(f"Loading the universal LLaMA tokenizer from {p}...")
                    tokenizer = LlamaTokenizer.from_pretrained(p, clean_up_tokenization_spaces=True)
                    return tokenizer

        # Otherwise, load it from the model folder and hope that these
        # are not outdated tokenizer files.
        tokenizer = LlamaTokenizer.from_pretrained(Path(f"{shared.args['model_dir']}"+ os.sep + "{model_name}" + os.sep), clean_up_tokenization_spaces=True)
        try:
            tokenizer.eos_token_id = 2
            tokenizer.bos_token_id = 1
            tokenizer.pad_token_id = 0
        except:
            pass
    else:
        path_to_model = Path(f"{shared.args['model_dir']}"+ os.sep +"{model_name}"+ os.sep)
        if path_to_model.exists():
            tokenizer = AutoTokenizer.from_pretrained(path_to_model, trust_remote_code=shared.args['trust_remote_code'])

    return tokenizer


def huggingface_loader(model_name):
    if shared.model_type == 'chatglm':
        LoaderClass = AutoModel
    elif shared.model_type == 'HF_seq2seq':
        LoaderClass = AutoModelForSeq2SeqLM
    else:
        LoaderClass = AutoModelForCausalLM

    if (torch.cuda.is_available):
        params = {}
    # Load the model in simple 16-bit mode by default
    if not any([shared.args['cpu'], shared.args['load_in_8bit'], shared.args['load_in_4bit'], shared.args['auto_devices'], shared.args['disk'], shared.args['deepspeed'], shared.args['gpu_memory'] is not None, shared.args['cpu_memory'] is not None]):
        model = LoaderClass.from_pretrained(Path(f"{shared.args['model_dir']}"+ os.sep +"{model_name}"), low_cpu_mem_usage=True, torch_dtype=torch.bfloat16 if shared.args['bf16'] else torch.int8, trust_remote_code=shared.args['trust_remote_code'])
        # removed if torch.has_mps: as mps is support related to mac and apple architectures - support is unrelated for now
        model = model.cuda()
 
    # removed elif shared.args. deepspeed for DeepSpeed ZeRO-3

    # Custom
    else:
        params = {
            "low_cpu_mem_usage": True,
            "trust_remote_code": shared.args['trust_remote_code']
        }

        if not any((shared.args['cpu'], torch.cuda.is_available(), torch.has_mps)):
            logger.warning("torch.cuda.is_available() returned False. This means that no GPU has been detected. Falling back to CPU mode.")
            shared.args['cpu'] = True

        if shared.args['cpu']:
            params["torch_dtype"] = torch.float32
        else:
            params["device_map"] = 'auto'
            if shared.args['load_in_4bit']:

                # See https://github.com/huggingface/transformers/pull/23479/files
                # and https://huggingface.co/blog/4bit-transformers-bitsandbytes
                quantization_config_params = {
                    'load_in_4bit': True,
                    'bnb_4bit_compute_dtype': eval("torch.{}".format(shared.args['compute_dtype'])) if shared.args['compute_dtype'] in ["bfloat16", "float16", "float32"] else None,
                    'bnb_4bit_quant_type': shared.args['quant_type'],
                    'bnb_4bit_use_double_quant': shared.args['use_double_quant'],
                }

                logger.warning("Using the following 4-bit params: " + str(quantization_config_params))
                params['quantization_config'] = BitsAndBytesConfig(**quantization_config_params)

            elif shared.args['load_in_8bit'] and any((shared.args['auto_devices'], shared.args['gpu_memory'])):
                params['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
            elif shared.args['load_in_8bit']:
                params['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
            elif shared.args['bf16']:
                params["torch_dtype"] = torch.bfloat16
            else:
                params["torch_dtype"] = torch.float16

            params['max_memory'] = get_max_memory_dict()
            if shared.args['disk']:
                params["offload_folder"] = shared.args['disk_cache_dir']

        checkpoint = Path(f"{shared.args['model_dir']}"+ os.sep +'{model_name}')
        if shared.args['load_in_8bit'] and params.get('max_memory', None) is not None and params['device_map'] == 'auto':
            config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=shared.args['trust_remote_code'])
            with init_empty_weights():
                model = LoaderClass.from_config(config, trust_remote_code=shared.args['trust_remote_code'])

            model.tie_weights()
            params['device_map'] = infer_auto_device_map(
                model,
                dtype=torch.int8,
                max_memory = params['max_memory'],
                no_split_module_classes=model._no_split_modules
            )

        model = LoaderClass.from_pretrained(checkpoint, **params)

    return model


def flexgen_loader(model_name):
    from flexgen.flex_opt import CompressionConfig, ExecutionEnv, OptLM, Policy

    # Initialize environment
    env = ExecutionEnv.create(shared.args['disk_cache_dir'])

    # Offloading policy
    policy = Policy(1, 1,
                    shared.args['percent'[0]], shared.args['percent'[1]],
                    shared.args['percent'[2]], shared.args['percent'[3]],
                    shared.args['percent'[4]], shared.args['percent'[5]],
                    overlap=True, sep_layer=True, pin_weight=shared.args['pin_weight'],
                    cpu_cache_compute=False, attn_sparsity=1.0,
                    compress_weight=shared.args['compress_weight'],
                    comp_weight_config=CompressionConfig(
                        num_bits=4, group_size=64,
                        group_dim=0, symmetric=False),
                    compress_cache=False,
                    comp_cache_config=CompressionConfig(
                        num_bits=4, group_size=64,
                        group_dim=2, symmetric=False))

    model = OptLM(f"facebook/{model_name}", env, shared.args['model_dir'], policy)
    return model

# removed RWKV_loader(model_name):

def llamacpp_loader(model_name):
    from modules.llamacpp_model import LlamaCppModel

    path = Path(f"{shared.args['model_dir']}"+ os.sep +'{model_name}')
    if path.is_file():
        model_file = path
    else:
        model_file = list(Path(f"{shared.args['model_dir']}"+ os.sep +'{model_name}').glob('*ggml*.bin'))[0]

    logger.info(f"llama.cpp weights detected: {model_file}\n")
    model, tokenizer = LlamaCppModel.from_pretrained(model_file)
    return model, tokenizer


def GPTQ_loader(model_name):

    # removed the Monkey patch for loading LoRAs for GPTQ - uses load_model_llama
    import modules.GPTQ_loader
    model = modules.GPTQ_loader.load_quantized(model_name)
    model.tie_weights()
    return model

# removed AutoGPTQ_loader(model_name):

def get_max_memory_dict():
    max_memory = {}
    if shared.args['gpu_memory']:
        memory_map = list(map(lambda x: x.strip(), shared.args['gpu_memory']))
        for i in range(len(memory_map)):
            max_memory[i] = f'{memory_map[i]}GiB' if not re.match('.*ib$', memory_map[i].lower()) else memory_map[i]

        max_cpu_memory = shared.args['cpu_memory'].strip() if shared.args['cpu_memory'] is not None else '99GiB'
        max_memory['cpu'] = f'{max_cpu_memory}GiB' if not re.match('.*ib$', max_cpu_memory.lower()) else max_cpu_memory

    # If --auto-devices is provided standalone, try to get a reasonable value
    # for the maximum memory of device :0
    elif shared.args['auto_devices']:
        total_mem = (torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
        suggestion = round((total_mem - 1000) / 1000) * 1000
        if total_mem - suggestion < 800:
            suggestion -= 1000

        suggestion = int(round(suggestion / 1000))
        logger.warning(f"Auto-assiging --gpu-memory {suggestion} for your GPU to try to prevent out-of-memory errors. You can manually set other values.")
        max_memory = {0: f'{suggestion}GiB', 'cpu': f"{shared.args['cpu_memory'] or 99}GiB"}

    return max_memory if len(max_memory) > 0 else None


def clear_torch_cache():
    gc.collect()
    if not shared.args['cpu']:
        torch.cuda.empty_cache()


def unload_model():
    shared.model = shared.tokenizer = None
    clear_torch_cache()


def reload_model():
    unload_model()
    shared.model, shared.tokenizer = load_model(shared.model_name)


def load_soft_prompt(name):
    if name == 'None':
        shared.soft_prompt = False
        shared.soft_prompt_tensor = None
    else:
        with zipfile.ZipFile(Path(f'softprompts'+ os.sep +'{name}.zip')) as zf:
            zf.extract('tensor.npy')
            zf.extract('meta.json')
            j = json.loads(open('meta.json', 'r').read())
            logger.info(f"\nLoading the softprompt \"{name}\".")
            for field in j:
                if field != 'name':
                    if type(j[field]) is list:
                        logger.info(f"{field}: {', '.join(j[field])}")
                    else:
                        logger.info(f"{field}: {j[field]}")

            tensor = np.load('tensor.npy')
            Path('tensor.npy').unlink()
            Path('meta.json').unlink()

        tensor = torch.Tensor(tensor).to(device=shared.model.device, dtype=shared.model.dtype)
        tensor = torch.reshape(tensor, (1, tensor.shape[0], tensor.shape[1]))
        shared.soft_prompt = True
        shared.soft_prompt_tensor = tensor

    return name
