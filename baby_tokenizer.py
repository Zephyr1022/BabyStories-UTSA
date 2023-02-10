import os, sys
import glob
import json
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

#folder_path = "./babylm_data/babylm_dev/"
##paths = [str(x) for x in Path(folder_path).glob("*.dev")]
#print(Path(folder_path))
#paths = [str(x) for x in Path(".").glob("**/*.txt")] [str(x) for x in Path(folder_path).glob("**/*.txt")]

paths = glob.glob(os.path.join('./babylm_data/babylm_100M/', '*.train'))

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training, args.vocab_size
tokenizer.train(files=paths, vocab_size= 50257, min_frequency=2, special_tokens=[
  "<s>",
  "<pad>",
  "</s>",
  "<unk>",
  "<mask>",
])

#if not os.path.exists(token_dir):
#os.makedirs(token_dir)

# Save the tokenizer's model
token_dir = './babytoken'
tokenizer.save_model(directory=token_dir)


# Define the configuration of the Model
# Create a dictionary with the tokenizer's configuration
tokenizer_config = {
  "max_len": 1024,
  "vocab_size": 50257,
  "max_position_embeddings": 514,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "type_vocab_size": 1
}

# Save the tokenizer's configuration to a json file
with open(os.path.join('./babytoken/', "tokenizer_config.json"), 'w') as fp:
  json.dump(tokenizer_config, fp)
  
# The tokenizer is responsible for preprocessing text data by tokenizing it into words or subwords, 
# while the model is responsible for generating text based on the input tokens. 

# Create a dictionary with the model's configuration
# model_config = {
#   "model_type": "gpt2",
#   "architecture": "transformer",
#   "hidden_size": 512,
#   "num_layers": 12,
#   "num_attention_heads": 12,
#   "intermediate_size": 2048,
#   "hidden_act": "gelu",
#   "hidden_dropout_prob": 0.1,
#   "attention_probs_dropout_prob": 0.1,
#   "max_position_embeddings": 514,
#   "type_vocab_size": 1,
#   "vocab_size": 50257
# }
  
## Create a dictionary with the model's configuration
#model_config = {
# "model_type": "gpt2",
# "config": {
#     "n_embd": 1024,
#     "n_layer": 12,
#     "n_head": 12,
#     "afn": "gelu",
#     "resid_pdrop": 0.1,
#     "attn_pdrop": 0.1,
#     "embed_pdrop": 0.1,
#     "initializer_range": 0.02,
#     "vocab_size": 50257
#   }
#}
  
#if the tokenizer is associated with a GPT-2 model, the model_config would contain information about 
# the number of layers, the number of attention heads, the hidden size, etc. 
# This information can be useful for understanding how the tokenizer was trained and how it may perform on different types of text.
# Additionally, having access to the model_config can be useful for fine-tuning the tokenizer on your own data. 
# For example, if you are fine-tuning a GPT-2 tokenizer on a dataset of long documents, you may want to increase the max_len parameter in the model_config so that the tokenizer can handle longer input sequences.

# Create a dictionary with the model's configuration
model_config = {
  "model_type": "gpt2",
  "config": {
      "n_embd": 768,
      "n_layer": 12,
      "n_head": 12,
      "afn": "gelu",
      "resid_pdrop": 0.1,
      "attn_pdrop": 0.1,
      "embed_pdrop": 0.1,
      "initializer_range": 0.02,
      "vocab_size": 50257
    }
}
  

# Save the model's configuration to a json file
with open(os.path.join('./babytoken/', "config.json"), 'w') as fp:
    json.dump(model_config, fp)
  
  
# https://wandb.ai/bkkaggle/lm-finetuning/reports/Pretraining-a-124-M-Parameter-GPT-2-Language-Model--VmlldzoyMjg4NzA