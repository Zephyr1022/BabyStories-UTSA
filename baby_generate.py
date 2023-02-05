#!/usr/bin/env python3

# Code generation with a pipeline using a pretrained GPT2 model
# Generate text sequences based on seed texts
# Convert text sequences into numerical representations
# start training the full model, you will probably want to watch the losses carefully to make sure it is converging.
# If you see something weird, you can modify learning rates without waiting for it to finish and finding it did not learn anything.

from transformers import AutoTokenizer, GPT2LMHeadModel, set_seed
import numpy as np
import torch


# Text Generation Using GPT2
set_seed(42)

def main():
	
	# Load pre-trained model tokenizer (vocabulary)
	tokenizer = AutoTokenizer.from_pretrained("./babytoken")
#   tokenizer.add_special_tokens({'pad_token': '[PAD]'})
	print('Words in vocabulary: ', tokenizer.vocab_size)
	
	# Load pre-trained model (weights)
	base_model = GPT2LMHeadModel.from_pretrained('babylm-ds-accelerate')
	
	# set pad_token_id to eos_token_id because GPT2 does not have a PAD token
	tokenizer.pad_token_id = tokenizer.eos_token_id
	base_model.config.pad_token_id = tokenizer.eos_token_id
	
	# Set the model in evaluation mode to deactivate the DropOut modules
	base_model.eval() 
	
	# prompt
	prompt = "# What are the names of their children?"
	input_text = "Pruitt married Marlyn Pruitt in 1992. They have two children."
	
	context = """\
	# Answer the question: Pruitt married Marlyn Pruitt in 1992. They have two children. What boy's name of their children?
	Boy's name is John.
	
	# What's the girl's name?
	"""
	
#   context = input_text + " " + prompt
	
	# encode context the generation is conditioned on
	# generate text until the output length (which includes the context length) reaches 50
	base_model.to('cuda')
	for exp in range(10):
		## If you have a GPU, put everything on cuda
		input_ids = tokenizer.encode(context, return_tensors = 'pt')
		
		# create attention mask
		attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
		attention_mask = attention_mask.to('cuda')
		
		input_ids = input_ids.to('cuda')
		
		# generate until max_len or the end of sequence token is predicted
		greedy_output = base_model.generate(input_ids, 
			do_sample = True,
			min_length = 30,
			max_length = 128,
			top_k = 100, # sampling generation method to get different generations.
			attention_mask = attention_mask
		)
		print("Output:\n" + 100 * '-')
		print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
		
main()


#  it will generate until max_len or the end of sequence token is predicted
# Generation algorithms are important and can have a big impact on results. 
# I definitely recommend reviewing the different algorithms in the future.
# greedy_output = base_model.generate(input_ids, do_sample=True, max_length=128, top_k=100, min_length=50)
# would generate a sequence using topk sampling with k=100. 
# It will generate a sequence of at least 50 tokens and will go up to 128 or until the EOS token is generated