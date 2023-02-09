#!/usr/bin/env python3

# the output from another model and human feedback can be used as a reward for the response. 
# The reward can be any measure of quality or relevance of the response, 
# and using the output from another model or human feedback are both common ways of defining this reward.
# The reward is used to train the dialogue model, encouraging it to generate responses that are more likely to receive high rewards.

import torch

from transformers import pipeline, AutoTokenizer
from datasets import load_dataset

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, set_seed
from trl.core import LengthSampler

########################################################################
# This is a fully working simple example to use trl with accelerate.
#
# This example fine-tunes a GPT2 model on the IMDB dataset using PPO
# (proximal policy optimization).
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - multi GPUS (using DeepSpeed ZeRO-Offload stages 1 & 2)
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, first initialize the accelerate
# configuration with `accelerate config`
#
########################################################################

# We first define the configuration of the experiment, defining the model, the dataset,
# the training parameters, and the PPO parameters.
# Check the default arguments in the `PPOConfig` class for more details.
# If you want to log with tensorboard, add the kwarg
# `accelerator_kwargs={"logging_dir": PATH_TO_LOGS}` to the PPOConfig.

config = PPOConfig(
	model_name="lvwerra/gpt2-imdb",
	learning_rate=1.41e-5,
)

# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": config.forward_batch_size}


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(config, dataset_name="imdb", input_min_text_length=2, input_max_text_length=8):
	"""
	Build dataset for training. This builds the dataset from `load_dataset`, one should
	customize this function to train the model on its own dataset.
	Args:
		dataset_name (`str`):
			The name of the dataset to be loaded.
	Returns:
		dataloader (`torch.utils.data.DataLoader`):
			The dataloader for the dataset.
	"""
	tokenizer = AutoTokenizer.from_pretrained(config.model_name)
	tokenizer.pad_token = tokenizer.eos_token
	# load imdb with datasets
	ds = load_dataset(dataset_name, split="train")
	ds = ds.rename_columns({"text": "review"})
	ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)
	
	input_size = LengthSampler(input_min_text_length, input_max_text_length)
	
	def tokenize(sample):
		sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
		sample["query"] = tokenizer.decode(sample["input_ids"])
		return sample
	
	ds = ds.map(tokenize, batched=False)
	ds.set_format(type="torch")
	return ds


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(config)