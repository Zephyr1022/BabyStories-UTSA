#!/usr/bin/env python3

import torch
import glob, os, sys
from pathlib import Path

from datasets import load_dataset, DatasetDict
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.dataloader import DataLoader

from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig, AutoModelForSeq2SeqLM

from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets

# Gathering the data
# Define the directory containing the train files
train_files_path = './babylm_data/babylm_10M/'
dev_files_path = './babylm_data/babylm_dev/'

train_files = glob.glob('./babylm_data/babylm_10M/*.train') 
dev_files = glob.glob('./babylm_data/babylm_dev/*.dev')

ds_train = load_dataset('text', data_files='./babylm_data/babylm_10M/*.train')
print("ds_train", ds_train)

ds_valid = load_dataset('text', data_files='./babylm_data/babylm_dev/*.dev')
ds_valid["val"] = ds_valid.pop("train") # rename the test split to valid
print("ds_valid", ds_valid)

raw_datasets = DatasetDict({**ds_train, **ds_valid})
print("raw_datasets", raw_datasets)

# this results in:
# DatasetDict({
#	train: Dataset({
#		features: ['text'],
#		num_rows: 1026747
#	})
#	valid: Dataset({
#		features: ['text'],
#		num_rows: 1026747
#	})
#})

# Initializing Tokenizer...
device = torch.device("cuda") #if torch.cuda.is_available() else torch.device("cpu")
print("device:", device)
tokenizer = AutoTokenizer.from_pretrained("./babytoken")

# A more efficient way to prepare the data is to join all the tokenized samples in a batch with an eos_token_id token in between, 
# and then perform the chunking on the concatenated sequences. 
# As an exercise, modify the tokenize() function to make use of that approach. 
# Note that you’ll want to set truncation=False and remove the other arguments from the tokenizer to get the full sequence of token IDs.

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8
context_length = 1024

# Encode the text using the tokenizer
def process(example):
	ids = tokenizer.encode(
		example["text"],
		add_special_tokens=False, # encode_ordinary ignores any special tokens
		max_length=context_length, # max_length parameter is used to set the maximum length of the encoded sequence 
		truncation=True # truncation parameter is set to True to truncate the sequence if it exceeds the maximum length
		#return_overflowing_tokens=True,
		#return_length=True
	)
	
	# Get the end of text token id
	# note: I think eot should be prepended not appended... hmm. it's called "eot" though...
	eos_token_id=tokenizer.eos_token_id # add the end of text token, e.g. 50256 for gpt2 bpe
	
	# Append the end of text token id to the encoded ids
	ids.append(eos_token_id) 
	out = {'ids': ids, 'len': len(ids)}
	return out

# tokenize the dataset
tokenized = raw_datasets.map(
	process,
	remove_columns=['text'],
	desc="tokenizing the splits",
	num_proc=num_proc
)

# concatenate all the ids in each dataset into one large file we can use for training
for split, dset in tokenized.items():
	arr_len = np.sum(dset['len'])
	filename = f'{split}.bin'
	filepath = "./data/babylm/" + filename
	
	dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16) and Uint stands for unsigned integer
	arr = np.memmap(filepath, dtype=dtype, mode='w+', shape=(arr_len,))
	
	print(f"writing {filepath}...")
	idx = 0
	for example in tqdm(dset):
		arr[idx : idx + example['len']] = example['ids']
		idx += example['len']
	arr.flush()
	
# return train.bin and val.bin
# Step1, prepare the data is to join all the tokenized samples in a batch with an eos_token_id token in between: train.bin, no pad
# Step2, perform the chunking on the concatenated sequences, pad or not?
	
###################################################################################################
###################################################################################################
#context_length = 128
#
#def tokenize(element):
#	outputs = tokenizer(
#		element["text"],
#		truncation=True,
#		max_length=context_length,
#		return_overflowing_tokens=True,
#		return_length=True,
#	)
#	
#	input_batch = []
#	for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
#		if length == context_length:
#			input_batch.append(input_ids)
#	return {"input_ids": input_batch}
#
## DataLoader to load the data in batches.
#tokenized_datasets = raw_datasets.map(
#	tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
#)
###################################################################################################
###################################################################################################