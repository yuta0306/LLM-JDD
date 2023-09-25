import argparse
from pathlib import Path
import os

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import wandb
import numpy as np

from src.models import get_lora_model
from src.data import load_data, ConversationalDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, type=str)
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--experiment", required=True, type=str)
    parser.add_argument("--rank", required=False, type=int, default=16)
    parser.add_argument("--alpha", required=False, type=int, default=16)
    parser.add_argument("--target_modules", required=False, type=str, nargs="+", default=["gate_proj", "up_proj", "down_proj"])
    parser.add_argument("--dropout", required=False, type=float, default=0.05)
    parser.add_argument("--batch_size", required=False, type=int, default=8)
    parser.add_argument("--eval_batch_size", required=False, type=int, default=64)
    parser.add_argument("--learning_rate", required=False, type=float, default=4e-4)
    return parser.parse_args()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": np.mean(predictions == labels)}

    
def main():
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    train_data, eval_data = load_data(input_path=args.input_path, task="JDD")
    
    model = get_lora_model(
        model_path=args.model_path,
        r=args.rank,
        alpha=args.alpha,
        target_modules=args.target_modules,
        lora_dropout=args.dropout,
    )
    train_dataset = ConversationalDataset(
        model_path=args.model_path,
        data=train_data,
        task="JDD"
    )
    eval_dataset = ConversationalDataset(
        model_path=args.model_path,
        data=eval_data,
        task="JDD"
    )
    
    config = {
        "model": args.model_path,
        "epochs": 1,
        "lr": args.learning_rate,
        "warmup_steps": 100,
        "lr_scheduler": "cosine",
        "dataset": "JDD",
    }
    config.update(model.peft_config)
    
    wandb.init(project="LLM-JDD", name=args.experiment, config=config)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        save_total_limit=1,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=1,
        remove_unused_columns=False,
        bf16=True,
        dataloader_num_workers=16,
        report_to="wandb",
        logging_strategy="steps",
        logging_steps=20,
        evaluation_strategy="steps",
        per_device_eval_batch_size=args.eval_batch_size,
        eval_accumulation_steps=1,
        eval_steps=100,
        learning_rate=args.learning_rate,
        warmup_steps=50,
        lr_scheduler_type="cosine",
    )
    
    trainer = Trainer(
        model=model,
        data_collator=DataCollatorForLanguageModeling(train_dataset.tokenizer, mlm=False),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()