import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, PeftModelForCausalLM


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)

    
def preprocess_model(model):
    for param in model.parameters():
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    if hasattr(model.config, "pretraining_tp"):
        model.config.pretraining_tp = 1
    
    # model.embed_out = CastOutputToFloat(model.embed_out)
    return model

def get_lora_model(
    model_path: str,
    r: int = 16,
    alpha: int = 16,
    target_modules=["gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    fan_in_fan_out=False,
    task_type=TaskType.CAUSAL_LM,
) -> PeftModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto", trust_remote_code=True)
    model = preprocess_model(model)
    
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        bias=bias,
        fan_in_fan_out=fan_in_fan_out,
        task_type=task_type,
    )
    
    try:
        model = get_peft_model(model, config)
    except ValueError as e:
        print(model)
        raise ValueError(e)
    model.print_trainable_parameters()
    return model
