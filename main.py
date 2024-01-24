from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    AutoModelForCausalLM
)
import argparse
from loguru import logger
import os
from os.path import join
import torch
import bitsandbytes as bnb
from collections import defaultdict

from component.collator import SFTDataCollator
from component.dataset import SFTDataset
from component.argument import QLoRAArguments
from component.trainer import LoRATrainer,ModifiedTrainer
from component.loss import TargetLMLoss


def verify_model_dtype(model):
    """
    查看模型种各种类型的参数的情况
    """
    dtype2param_num = defaultdict(int)  
    dtype2param_name = defaultdict(list)  
    dtype2trainable_param_num = defaultdict(int) 
    dtype2trainable_param_name = defaultdict(list) 
    for name, p in model.named_parameters():
        dtype = p.dtype
        dtype2param_num[dtype] += p.numel()
        dtype2param_name[dtype].append(name)
        if p.requires_grad:
            dtype2trainable_param_num[dtype] += p.numel()
            dtype2trainable_param_name[dtype].append(name)

    total = 0
    print('verify all params of the model')
    for k, v in dtype2param_num.items():
        total += v
    for k, v in dtype2param_num.items():
        print(k, v, v / total)
    for k, v in dtype2trainable_param_name.items():
        print(k, v)

    print()

    print('verify trainable params the model')
    total_trainable = 0
    for k, v in dtype2trainable_param_num.items():
        total_trainable += v
    for k, v in dtype2trainable_param_num.items():
        print(k, v, v / total_trainable)
    for k, v in dtype2trainable_param_num.items():
        print(k, v)


def setup_everything():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_file", type=str, default='train_args/baichuan-sft-qlora.json', help="")
    args = parser.parse_args()
    train_args_file = args.train_args_file
    
    parser = HfArgumentParser((QLoRAArguments, TrainingArguments))

    args, training_args = parser.parse_json_file(json_file=train_args_file)

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    set_seed(training_args.seed)
    return args, training_args


def init_components(args, training_args):
    """
    初始化各个组件
    """
    logger.info('Initializing components...')

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    training_args.ddp_find_unused_parameters = False
    device_map = "auto"
    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}


    print("*** model loading ***")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        load_in_8bit=False, 
        trust_remote_code=True,
        device_map="auto"
    )
    print("*** model finish ***")

    model.gradient_checkpointing_enable() 
    # note: use gradient checkpointing to save memory at the expense of slower backward pass.
    model.enable_input_require_grads()


    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )



    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id

    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise Exception('pad_token_id should not be equal to eos_token_id')



    config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["W_pack"],
        lora_dropout=args.lora_dropout
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    model.config.torch_dtype = torch.float32



    loss_func = TargetLMLoss(ignore_index=tokenizer.pad_token_id)


    train_dataset = SFTDataset(args.train_file, tokenizer, args.max_seq_length)
    data_collator = SFTDataCollator(tokenizer, args.max_seq_length)


    print("*** trainer ***")
    trainer = ModifiedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
       # compute_loss=loss_func
    )

    print("*** starting training ***")
    train_result = trainer.train()
    model.save_pretrained(training_args.output_dir)

    return trainer


def main():
    args, training_args = setup_everything()

    trainer = init_components(args, training_args)



if __name__ == "__main__":
    main()

