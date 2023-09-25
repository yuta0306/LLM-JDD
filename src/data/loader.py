from typing import Literal
from collections import defaultdict

import torch.utils.data as data
import torch
from transformers import AutoTokenizer, LlamaTokenizer
from tqdm import tqdm


def format_data(data: list[dict], task: str, sep: str):
    if task == "JDD":
        processed = [sep.join([row["utterance"] for row in session["utterances"]]) for session in data]
        return processed

class ConversationalDataset(data.Dataset):
    def __init__(self, model_path: str, data: list[dict], task: Literal["JDD"]) -> None:
        if model_path in ("stabilityai/japanese-stablelm-base-alpha-7b"):
            self.tokenizer = LlamaTokenizer.from_pretrained("novelai/nerdstash-tokenizer-v1", additional_special_tokens=['▁▁'], trust_remote_code=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        processed = []
        if task == "JDD":
            processed = format_data(data, task="JDD", sep=self.tokenizer.eos_token)
        
        self.data = [
            self.tokenizer(
                session,
                padding=False,
                truncation=True,
                max_length=300,
            ).pop("token_type_ids")
            if model_path in ("matsuo-lab/weblab-10b")
            else self.tokenizer(
                session,
                padding=False,
                truncation=True,
                max_length=300,
            )
            for session in tqdm(processed, leave=False)
        ]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int):
        return self.data[idx]

class ConversationalCollator(object):
    def __init__(self) -> None:
        pass
    
    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        res = defaultdict(list)
        for row in batch:
            for k, v in row.items():
                res[k].append(v)
        return {k: torch.stack(v) for k, v in res.items()}
