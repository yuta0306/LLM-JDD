#! /bin/bash
pip3 install transformers accelerate datasets peft wandb sentencepiece
mkdir -p ~/datasets
git clone https://github.com/jqk09a/japanese-daily-dialogue ~/datasets/japanese-daily-dialogue
