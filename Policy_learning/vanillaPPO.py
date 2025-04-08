import grp
from venv import logger
import torch
import trl
import torch.nn as nn
import torch.optim as optim
import random
import os
import logging
import sys
import re

# setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
if torch.cuda.is_available():
    device = 'cuda'
    logging.info("CUDA is available. Running on Nvidia GPU")
elif torch.mps.is_available():
    device = 'mps'
    logging.info("MPS is available. Running on Apple M4 GPU")
else:
    device = 'cpu'
    logging.info("CUDA and MPS are not available. Running on CPU")
# Set the random seed for reproducibility
random.seed(42)
torch.manual_seed(42)
if device == 'cuda':
    torch.cuda.manual_seed_all(42)
elif device == 'mps':
    torch.mps.manual_seed(42)
else:
    torch.manual_seed(42)

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset

# Load the dataset
ds = load_dataset("open-r1/codeforces")
ds_code = Dataset.load_from_disk("/home/sureshm/DeepRL/dataset/babeltower")
#ds_code = ds_code.select(range(100))

logger.info(f"Dataset: {ds_code}")

class PPOAgent():
    def __init__(self):
        self.actor = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        self.optimizer = optim.AdamW(self.actor.parameters(), lr=1e-5, eps=1e-8, weight_decay=0.01, betas=(0.9, 0.999))
        self.trainergrpo = None
        self.actor.train()
        self.template = ("Convert C/C++ Code to CUDA. Provide the answer in the following format <think> {think} </think> <analysis> {analysis} </analysis> <code> {code} </code>\n"
                         "Input code  \n" 
                         "<input_code>"
                         )
        for param in self.actor.parameters():
            param.requires_grad = True

    @staticmethod
    def reward_structured_format(completions, **kwargs):
        rewards = []

        for completion in completions:
            reward = 0

            # 1. Check if all tags are present
            has_think = bool(re.search(r"<think>.*?</think>", completion, re.DOTALL))
            has_answer = bool(re.search(r"<analysis>.*?</analysis>", completion, re.DOTALL))
            has_code = bool(re.search(r"<code>.*?</code>", completion, re.DOTALL))

            if has_think: reward += 1
            if has_answer: reward += 1
            if has_code: reward += 1

            # 2. Check tag order (think → analysis → code)
            tag_order = ["<think>", "<analysis>", "<code>"]
            tag_positions = [completion.find(tag) for tag in tag_order]

            if all(pos != -1 for pos in tag_positions) and tag_positions == sorted(tag_positions):
                reward += 1  # bonus for correct ordering

            # 3. Penalize malformed tags
            total_open_tags = sum(completion.count(f"<{tag}>") for tag in ["think", "analysis", "code"])
            total_close_tags = sum(completion.count(f"</{tag}>") for tag in ["think", "analysis", "code"])
            if total_open_tags != 3 or total_close_tags != 3:
                reward -= 1  # structural issue

            rewards.append(reward)

        return rewards
        
    def create_grpo_trainer(self, ds_code):
        training_args = trl.GRPOConfig(
            output_dir="grpo_output",
            logging_steps=100,
            save_steps=1000,
            num_train_epochs=1,
            do_train = True,
        )
        self.trainergrpo = trl.GRPOTrainer(
            self.actor,
            reward_funcs=self.reward_structured_format,
            train_dataset=ds_code,
            args =training_args,
            optimizers=(self.optimizer, None)
        )
        return self.trainergrpo
    
    def apply_template(self, ds_code):
        def replace_code(example):
            prompt = self.template.replace("<input_code>", example.get("code", ""))
            return {"prompt": prompt}
        
        new_ds_code = ds_code.map(replace_code)

        return new_ds_code
        
    def learn(self):
        pass
 
agent = PPOAgent()
ds_code = agent.apply_template(ds_code)
ds_code = ds_code.remove_columns(["identifier", "code"])
grpotrainer = agent.create_grpo_trainer(ds_code)
logger.info(f"ActorCritic model: {grpotrainer}")
grpotrainer.train(resume_from_checkpoint=True)
#grpotrainer.save_model("ppo_output")
