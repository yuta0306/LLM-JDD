import argparse
from pathlib import Path
import os

import torch
import numpy as np
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from src.models import get_lora_model
from src.data import load_data, ConversationalDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--adapter_path", required=True, type=str)
    return parser.parse_args()

    
def main():
    args = parse_args()
    
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = PeftModel.from_pretrained(model, args.adapter_path, torch_dtype="auto")
    streamer = TextStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    
    history = ""
    while True:
        try:
            # print("HISTORY:", history)
            utterance = input("Input:")
            history += f"{utterance}{tokenizer.eos_token}"
            inputs = tokenizer(history, return_tensors="pt").to(model.device)
            if args.model_path in ("matsuo-lab/weblab-10b"):
                inputs.pop("token_type_ids")
            
            with torch.no_grad():
                tokens = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=True,
                    temperature=0.9,
                    streamer=streamer,
                )
            outputs = tokenizer.decode(tokens[0], skip_special_tokens=True)
            outputs = outputs.split("\n")[0]
            outputs = outputs.replace(history.replace(tokenizer.eos_token, ""), "")
            history += f"{outputs}{tokenizer.eos_token}"
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()