# reference baby_pipeline3.py

import torch
import time
import pickle
import glob, os, sys
import numpy as np
from pathlib import Path
import datetime

from tqdm.notebook import tqdm
from transformers import pipeline
from accelerate import Accelerator

from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForSeq2SeqLM
from transformers import GPT2LMHeadModel
from transformers import get_scheduler

from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

from datasets import load_dataset, DatasetDict
from torch.utils.data import Dataset, DataLoader, ConcatDataset, TensorDataset
from torch.utils.data.dataloader import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

from transformers import PretrainedConfig
from typing import List

from transformers import GPT2LMHeadModel, GPT2Config

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M)
# I/O
# global variable
# Model and Training Parameters
SEED = 123
context_length = 1024 # 128, 256, 1024
block_size = 1024 # 128, 256, 1024

# data
dataset = 'babylm'
master_process = True
# batch_size = 8 # test
gradient_accumulation_steps = 2 # used to simulate larger batch sizes
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE =  8
VALID_BATCH_SIZE = 8

# model
EPOCHS = 30
#n_layer = 12
#n_head = 12
#n_embd = 768
#dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
#bias = False # do we use bias inside LayerNorm and Linear layers?

# adamw optimizer
LEARNING_RATE = 1e-3 # 1e-4, 2e-5, 5e-4, e-4, 6e-4 
weight_decay = 0.1

# system
# device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks

# save best model for use later
output_dir="babylm-ds-accelerate" # MODEL_PATH = best_model
# -----------------------------------------------------------------------------

##################################################################################################################
# Customize config
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
##################################################################################################################

# Writing a custom configuration
class BabyConfig(PretrainedConfig):
    model_type = "babylm"
    
    def __init__(
        self,
        block_type="bottleneck",
        layers: List[int] = [3, 4, 6, 3],
        num_classes: int = 1000,
        input_channels: int = 3,
        cardinality: int = 1,
        base_width: int = 64,
        stem_width: int = 64,
        stem_type: str = "",
        avg_down: bool = False,
        **kwargs,
    ):
        if block_type not in ["basic", "bottleneck"]:
            raise ValueError(f"`block_type` must be 'basic' or bottleneck', got {block_type}.")
        if stem_type not in ["", "deep", "deep-tiered"]:
            raise ValueError(f"`stem_type` must be '', 'deep' or 'deep-tiered', got {stem_type}.")

        self.block_type = block_type
        self.layers = layers
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.cardinality = cardinality
        self.base_width = base_width
        self.stem_width = stem_width
        self.stem_type = stem_type
        self.avg_down = avg_down
        super().__init__(**kwargs)

# baby_config = BabyConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
# baby_config.save_pretrained("custom-babylm") $ save a file named config.json inside the folder custom-resnet. 
# You can then reload your config with the from_pretrained method:
# baby_config = BabyConfig.from_pretrained("custom-babylm")

# Writing a custom model(CustomModel)
# Extracting the Body and adding our own layers
# how to use a pre-trained Body and add a Custom Head. 
class BabyModel(GPT2LMHeadModel): # nn.Module
    def __init__(self, num_labels):
        
        # config = AutoConfig.from_pretrained(checkpoint)
        # config = GPT2Config.from_pretrained("gpt2") 
        config = BabyConfig.from_pretrained("custom-babylm")
        config.output_attentions = True
        config.output_hidden_states = True

        super(BabyModel, self).__init__(config) #GPT2LMHeadModel(config)
        self.num_labels = num_labels
        
        # Load Model with given checkpoint and extract its body
        self.model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels) # initialize weights

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask) 
        
        # add custom layers
        sequence_output = self.dropout(outputs[0]) # outputs[0]=last hidden state

        logits = self.classifier(sequence_output[:, 0, :].view(-1, 768)) # calculate logits
        loss = None
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
    
    

# custom loss function that takes the input sequence, the logits - probability
def keytoken_weighted_loss(inputs, logits, alpha=1.0):
    
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    
    # Calculate per-token loss
#   loss_fct = CrossEntropyLoss(reduce=False)
    loss_fct = CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # Resize and average loss per sample
    loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)
    
    # Calculate and scale weighting: filter
    # weights = torch.stack([(inputs == kt).float() for kt in keytoken_ids]).sum(
    # axis=[0, 2])
    # weights = alpha * (1.0 + weights)
    
    # Calculate weighted average
    weighted_loss = loss_per_sample.mean()
    return weighted_loss

# a weight decay (a value used to penalize larger values in the model's parameters to prevent overfitting
# and a list of parameters that should not be penalized by weight decay (no_decay).
# returns a list of dictionaries representing the different parameter groups, 
# with each dictionary specifying the parameters (params) and the weight decay (weight_decay) for that group. 
def get_grouped_params(model, weight_decay, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
        # {"params": params_without_wd, "weight_decay": weight_decay},
    ]

# In the training loop we iterate over the dataloader and pass the batches (random sample) to the model.
# With the logits, we can then evaluate our custom loss function. 
# We scale the loss by the number of gradient accumulation steps 
# so as not to create larger losses when aggregating more steps.
# every few steps we evaluate the model on the evaluation set with our new evaluate() function
# and save the best model
def train(model, accelerator, train_dataloader, optimizer, lr_scheduler, device, epoch, epochs):
    
    # capture time
    total_t0 = time.time()
    
    # scale the loss by the number of gradient accumulation steps 
    # so as not to create larger losses when aggregating more steps.
    # gradient_accumulation_steps = 1 # global variable
    
    # reset total loss for epoch
    current_time_train = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")
    train_total_loss = 0
    total_train_perplexity = 0
    completed_steps = 0
    max_norm = 1.0
    losses = []
    
    # Perform one full pass over the training set.
#   print("")
#   print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
#   print('Training...')
    print(f"{current_time_train} - {'-' * 89}") # training start
    
    model.train() # put model into traning mode

    # for each batch of training data...
    for step, batch in enumerate(train_dataloader, start=1):
        
        # progress update every 40 batches.
        # the number of updates
        # practical experience: total number of updates for each epoch between 500 to 4k
        # If the number of updates is greater than that because of limited minibatch sizes, maybe try with accumulation
        # if step % 1 == 0:# and not step == 0:
        #     print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader))) # Report progress.
        # clear previously calculated gradients
        # optimizer.zero_grad()
                    
        # 1. input output
        # forward propagation (evaluate model on training batch)
        logits = model(batch.to('cuda')).logits # pred, Forward pass
        loss = keytoken_weighted_loss(batch.to('cuda'), logits) # loss functions, for the current batch
        
        # sum the training loss over all batches for average loss at end
        # loss is a tensor containing a single value
        train_total_loss += loss.item()

        # progress update every 100 batches.
        if step % 100 == 0:
            print(f"{current_time_train} - epoch {epoch+1:1d} - iter {step:1d}/{len(train_dataloader):1d} - steps {completed_steps} - train/loss {loss.item() * gradient_accumulation_steps:.8f} - lr: {lr_scheduler.get_lr()[0]:.6f}")
            sys.stdout.flush()

#           accelerator.print(
#               {
#                   "lr": lr_scheduler.get_lr(),
#                   #"samples": step * samples_per_step,
#                   #"steps": completed_steps, # time of update
#                   "loss/train": loss.item() * gradient_accumulation_steps, 
                    # the total training loss for a set of gradient accumulation steps
                    # This total loss is used for computing gradients and updating the model parameters.
#               } # Report progress.
#           )
        
        # 2.1 loss regularization
        # Scales loss. Calls backward() on scaled loss to create scaled gradients.
        loss = loss / gradient_accumulation_steps # default 1, normalizing the loss based on the number of accumulation steps 
        
        # 2.2 back propagation # 反向传播，计算梯度
        # 多次循环步骤 1-2，不清空梯度，使梯度累加在已有梯度上
        # backpropagation-> gradient
        # generating the gradient of the parameters for the loss and storing it until a step (update) is made.
        accelerator.backward(loss)

        # 3. update parameters of net
        if step % gradient_accumulation_steps == 0: # Wait for several backward steps
                # clip gradients at this value, or disable if == 0.0
                accelerator.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step() # taking the gradients stored from accelerator.backward(loss) to make an update parameters of net
                #  lr_scheduler.step() # update the learning rate
                optimizer.zero_grad() # reset gradient, clear previously calculated gradients
                completed_steps += 1
        
        # 简单的说就是进来一个 batch 的数据，计算一次梯度，更新一次网络
        # calculate preds
#       with torch.no_grad():
#           outputs = model(batch.to('cuda'), labels=batch.to('cuda'))
#           total_train_perplexity += torch.exp(outputs.loss).mean()
#           #print("outputs:", outputs) # no decode
#       losses.append(accelerator.gather(outputs.loss))


    # calculate the average loss over all of the batches
    # avg_train_loss = torch.mean(torch.stack(losses)) # cat -> stack # train_total_loss / len(train_dataloader)
    avg_train_loss = train_total_loss / len(train_dataloader)
    
    # calculate the average perplexity over all of the batches
    avg_train_perplexity = torch.exp(torch.tensor(avg_train_loss)) # total_train_f1 / len(train_dataloader)
    
    # training time end
    training_time = time.time() - total_t0
    
    # learning rate: print("lr:", lr_scheduler.get_lr()[0])
    lr_train = lr_scheduler.get_lr()[0]
    
    # print result summaries
#   print("-" * 50)
#   print("training summary results")
#   print("-" * 50)
#   print("epoch | trn loss | trn perplexity | trn time | lr ")
#   print(f" epoch {epoch+1:5d} - loss {avg_train_loss:.5f} - perplexity {avg_train_perplexity:.5f} -time (sec) {training_time:.2f} - lr {lr_train:.5f}" )
    print(f"{current_time_train} - {'-' * 89}")
    print(f"{current_time_train} - EPOCH {epoch+1:1d} TRAIN done: - loss {avg_train_loss:.5f} - perplexity {avg_train_perplexity:.5f} - time (sec) {training_time:.2f} - lr {lr_train:.5f}")


    
# We want to evaluate the model regularly on the validation set during training 
# With the evaluate() function we can report loss and perplexity at regular intervals. 
# evaluate the model regularly on the validation set during training
# runs through the evaluation dataloader and gathers all the losses across processes
def evaluate(model, accelerator, eval_dataloader, epoch, epoches):
    # capture validation time
    total_t0 = time.time()
    current_time_val = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")
#   print("")
#   print("Running Validation...")
    
    model.eval() 
    # track variables
    total_valid_perplexity = 0
    total_valid_loss = 0
    losses = []
    for step, batch in enumerate(eval_dataloader):
        
        # tell pytorch not to bother calculating gradients
        # as its only necessary for training
        with torch.no_grad(): # calculate preds
            outputs = model(batch.to('cuda'), labels=batch.to('cuda')) # batch = batch["input_ids"]
        
        losses.append(accelerator.gather(outputs.loss))
        
    valid_loss = torch.mean(torch.stack(losses)) # cat -> stack
    
    try:
        perplexity = torch.exp(valid_loss)
    except OverflowError:
        perplexity = float("inf")
    
    # capture end validation time
    training_time = time.time() - total_t0
        
    # print result summaries
#   print("")
#   print("validation summary results")
#   print("epoch | val loss | val perplexity | val time")
#   print(f"{epoch+1:5d} | {loss} | {perplexity} | {training_time:}")
    print(f"{current_time_val} - EPOCH {epoch+1:1d} DEV done: - loss {valid_loss:.5f} - perplexity {perplexity:.5f}")
    
    return valid_loss.item(), perplexity.item()




def main():

#   log_file = open(MODEL_PATH + "/output.log","w")
#   sys.stdout = log_file
    
    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(SEED) # pytorch random seed
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED) # numpy random seed
    
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    # torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    # device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    
    # effective batch size of 256
    # samples_per_step = per_device_train_batch_size * gradient_accumulation_steps # 128
    # setting the accumulation_steps attribute, update the gradients every 4 iterations.
    # optimizer.accumulation_steps = 4
    
    MODEL_PATH = os.path.join('experiments', output_dir)
    if master_process:
        os.makedirs(MODEL_PATH, exist_ok=True)

    # DataLoader
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r') # embeddings
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
#   ##########################################TEST#######################################
    # Specify the size of the subset
#   subset_size = 10000
#   # Generate a random index for subset
#   subset_indices_tr = np.random.randint(0, train_data.shape[0], subset_size)
#   subset_indices_vl = np.random.randint(0, val_data.shape[0], subset_size)
#   # Create the subset by indexing the full dataset
#   subset_data_train = train_data[subset_indices_tr]
#   subset_data_valid = val_data[subset_indices_vl]
#   print("subset", subset_data_train,subset_data_valid)
#   
#   train_data = subset_data_train 
#   val_data = subset_data_valid 
    
    train_data = val_data
    val_data = val_data
#   ##########################################TEST#######################################
    
    train_block = torch.arange(0, len(train_data)-block_size, block_size)
    val_block = torch.arange(0, len(val_data)-block_size, block_size)
    
    train_dataset = torch.stack([torch.from_numpy((train_data[i:i+block_size]).astype(np.int64)) for i in train_block])
    val_dataset = torch.stack([torch.from_numpy((val_data[i:i+block_size]).astype(np.int64)) for i in val_block])
    
    # We need dataloaders to load the data in batches.
    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers = 0, shuffle=True, drop_last=True)
    eval_dataloader = DataLoader(val_dataset, batch_size=VALID_BATCH_SIZE)


    # Initializing Tokenizer...
#   print(f"Initializing Tokenizer...")
    device = torch.device("cuda") #if torch.cuda.is_available() else torch.device("cpu")    
    tokenizer = AutoTokenizer.from_pretrained("./babytoken") # pretrained checkpoint = "./babytoken"
    # tokenizer.pad_token = tokenizer.eos_token


    # Initializing a New Model - 124M parameters need to tune
    print(f"Initializing Model...")
    gptconfig = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size= len(tokenizer), # or using GPT-2 default of 50257
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # pad_token_id=tokenizer.pad_token_id,
    ) # With that configuration, we can load a new model.
    # DataCollatorForLanguageModeling supports both masked language modeling (MLM) and causal language modeling (CLM).
    
    print(f" Config: {gptconfig}")
    
    current_time_main = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")
    print(f"{current_time_main} - {'-' * 89}") 
    print(f"{current_time_main} - Model training base path:: {MODEL_PATH}")
    print(f"{current_time_main} - {'-' * 89}") 
    print(f"{current_time_main} - Device: {device}")
    print(f"{current_time_main} - {'-' * 89}") 
    
#   print("config", config)
    print(f"{current_time_main} Parameters:")
    for cf_key, cf_value in config.items():
        print(f"{current_time_main} - {cf_key}: {cf_value}")
    print(f"{current_time_main} - {'-' * 89}")

    # init a new model from scratch
    model = GPT2LMHeadModel(gptconfig).to('cuda')
    model_size = sum(t.numel() for t in model.parameters())
    print(f"{current_time_main} - GPT-2 size: {model_size/1000**2:.1f}M parameters")
    sys.stdout.flush()
    
    
    # Next, we group the parameters so that the optimizer knows which ones will get an additional weight decay.
    # set up weight decay parameters.    
    optimizer = AdamW(get_grouped_params(model, weight_decay),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps= 1e-08,
        weight_decay=0.,
        amsgrad=False
    )
#   print("test: get_grouped_params", get_grouped_params(model, weight_decay))

    # prepare the model, optimizer, and dataloaders so we can start training:
    accelerator = Accelerator(fp16=False)
    
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    
    # check if the evaluation function works properly:
    max_loss, best_perplexity = evaluate(model, accelerator, eval_dataloader, 1, 1)
#   print("first evaluation metric:", max_loss, best_perplexity) # record as max 
    sys.stdout.flush()
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=20,
        eta_min=0
    )
    
#   num_update_steps_per_epoch = len(train_dataloader)
#   num_training_steps = EPOCHS * num_update_steps_per_epoch
#   
#   # using a classic linear schedule from the learning rate to 0
#   # "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
#   lr_scheduler = get_scheduler(
#       name="linear",
#       optimizer=optimizer,
#       num_warmup_steps=1_000,
#       num_training_steps=num_training_steps,
#   )

    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    
    # Training 
    epochs = EPOCHS
    for epoch in range(epochs):
        
        # training
        model.to(device)
        
        train(model, accelerator, train_dataloader, optimizer, lr_scheduler, device, epoch, epochs)
        sys.stdout.flush()

        # validate each epoch
        val_loss, perplexity = evaluate(model,accelerator, eval_dataloader,epoch,epochs)
        # accelerator.print({"loss/eval": val_loss, "perplexity": perplexity})
        sys.stdout.flush()
        
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(MODEL_PATH, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(MODEL_PATH)
            if perplexity < best_perplexity:
                torch.save(model.state_dict(), f"{MODEL_PATH}/model.pt")
                print(f"{current_time_main} - saving best model")
                best_perplexity = perplexity
                
        # update the learning rate
        lr_scheduler.step()

if __name__ == '__main__':
    main()

# batch["input_ids"].to('cuda') why pass batch to gpu?
# torch.save(model.state_dict(), f"{output_dir}/model.pt")

# Adding Custom Layers on Top of a Hugging Face Model
# https://towardsdatascience.com/adding-custom-layers-on-top-of-a-hugging-face-model-f1ccdfc257bd
# https://huggingface.co/docs/transformers/custom_models
    
# 一定条件下，batchsize 越大训练效果越好，梯度累加则实现了 batchsize 的变相扩大
# 如果accumulation_steps 为 8，则batchsize '变相' 扩大了8倍, 使用时需要注意，学习率也要适当放大
# batch size的值通常设置在 8-32 之间

    