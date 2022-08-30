from transformers import BertTokenizer, BertConfig
from transformers import AdamW
from transformers import BertForSequenceClassification
# from modeling_bert import BertForSequenceClassification
import json
import torch 
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine
from torch import nn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

from data import get_inputdata



"""pretrained model of choice"""
# pretrained_model_name = 'SpanBERT/spanbert-base-cased'
pretrained_model_name = 'bert-base-cased'
# pretrained_model_name = 'bert-large-cased'


"""set up tokenizer"""
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
# tokenizer = BertTokenizer.from_pretrained('bert-large-cased-whole-word-masking')
tokenizer_max_len = 200


trainset_path = './dataset/review_subsets/trainset_subset_1.json'
devset_path = './dataset/review_subsets/devset_subset_1.json'

"""import data"""
input_ids_dev,attention_masks_dev,labels_dev = get_inputdata(devset_path,tokenizer,maxlength = tokenizer_max_len, dataset = 'review')
input_ids,attention_masks,labels = get_inputdata(trainset_path,tokenizer,maxlength = tokenizer_max_len, dataset = 'review')
print('labels:',labels)

from torch.utils.data import TensorDataset, random_split

# Combine the training inputs into a TensorDataset.
train_dataset = TensorDataset(input_ids, attention_masks, labels)
val_dataset = TensorDataset(input_ids_dev, attention_masks_dev, labels_dev)
 

# Calculate the number of samples to include in each set.
# train_size = int(0.9 * len(dataset))
# val_size = len(dataset) - train_size
train_size = len(train_dataset)
val_size = len(val_dataset)

# Divide the dataset by randomly selecting samples.
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))



"""prepare dataloader"""
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# The DataLoader needs to know our batch size for training, so we specify it 
# here. For fine-tuning BERT on a specific task, the authors recommend a batch 
# size of 16 or 32.
batch_size = 32

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )


"""setup model, optimizer"""
model = BertForSequenceClassification.from_pretrained(pretrained_model_name, num_labels = 5)
optimizer = AdamW(model.parameters(), lr=1e-5)
# model.config = model.config.from_dict(pretrain_config)

# use cuda

if torch.cuda.is_available():  
  dev = "cuda:3" 
else:  
  dev = "cpu"
CUDA_VISIBLE_DEVICES=0,1,2,3  
device = torch.device(dev)

model.cuda(device)



"""setup epoch, scheduler"""
from transformers import get_linear_schedule_with_warmup

# Number of training epochs. The BERT authors recommend between 2 and 4. 
# We chose to run for 4, but we'll see later that this may be over-fitting the
# training data.
epochs = 1

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs
print("size of train_dataloader:" ,len(train_dataloader))
# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
"""helper functions"""
import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def trim_batch(input_ids, pad_token_id, labels, attention_mask=None):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return (input_ids[:, keep_column_mask], None, labels)
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask], labels)




"""training step"""

import random
import numpy as np

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    # print(pred_flat)
    labels_flat = labels.flatten()
    # print(labels_flat)
    # type_pred = [idx_to_event[i] for i in pred_flat]
    # type_groundtruth = [idx_to_event[i] for i in labels_flat]
    # print('predict:',type_pred,'ground truth:',type_groundtruth)
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def flat_accuracy_top_k(preds, labels,k):
    topk_preds = []
    for pred in preds:
        topk = pred.argsort()[-k:][::-1]
        topk_preds.append(list(topk))
    # print(topk_preds)
    topk_preds = list(topk_preds)
    right_count = 0
    # print(len(labels))
    for i in range(len(labels)):
        l = labels[i][0]
        if l in topk_preds[i]:
            right_count+=1
    return right_count/len(labels)
    # pred_flat = np.argmax(preds, axis=1).flatten()
    # # print(pred_flat)
    # labels_flat = labels.flatten()
    # # print(labels_flat)
    # # type_pred = [idx_to_event[i] for i in pred_flat]
    # # type_groundtruth = [idx_to_event[i] for i in labels_flat]
    # # print('predict:',type_pred,'ground truth:',type_groundtruth)
    # return np.sum(pred_flat == labels_flat) / len(labels_flat)

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# We'll store a number of quantities such as training and validation loss, 
# validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

# For each epoch...
for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0

    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):
        
        #TODO: trim input!
        # trim_batch(input_ids, pad_token_id, role_type_ids, entity_type_ids, labels, attention_mask=None):
        if step == 0:
            print('sanity check trim batch')
            print('before trim')
            print('input size before:',len(batch[0][0]))
        batch = trim_batch(batch[0],tokenizer.pad_token_id,batch[2],attention_mask = batch[1])
        if step == 0:
            print('after trim')
            print('input size after:',len(batch[0][0]))
            print('')
        # Progress update every 20 batches.
        if step % 20 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))


        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        # print(b_input_ids)

        model.zero_grad()        

        outputs = model(b_input_ids, attention_mask=b_input_mask,labels=b_labels)
        loss = outputs[0] 

        total_train_loss += loss.item()


        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)            
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
        
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_evel_acc_at_3 = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    total_pred_labels = np.array([])
    total_true_labels = np.array([])
    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        batch = trim_batch(batch[0],tokenizer.pad_token_id,batch[2],attention_mask = batch[1])

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        

        with torch.no_grad():        

            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

            loss = outputs[0] 
            logits = outputs[1]
            
            
        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        
        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        
        # print(pred_flat)
        # print(labels_flat)
        total_pred_labels = np.concatenate((total_pred_labels,pred_flat))
        total_true_labels = np.concatenate((total_true_labels,labels_flat))

    # Report the final accuracy for this validation run.
   
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)

    # print("  Accuracy@3: {0:.2f}".format(avg_val_acc_at_3))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    print('total preds:',total_pred_labels)
    print('total truth:',total_true_labels)
    print('sklearn macro: precision, recall, F1:')
    print(precision_recall_fscore_support(total_true_labels, total_pred_labels, average='macro'))
    print()
    print('sklearn micro: precision, recall, F1:')
    print(precision_recall_fscore_support(total_true_labels, total_pred_labels, average='micro'))
    print()
    print('sklearn accuracy:')
    print(accuracy_score(total_true_labels,total_pred_labels))
    print()

    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Acc.': avg_val_accuracy,
            # 'Valid. Acc.@3': avg_val_acc_at_3,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))


"""save model"""
import os

# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

output_dir = './bert_baseline_one_epoch/'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

#dump training stat
output_training_stat = output_dir + 'training_stat' 
with open(output_training_stat,'w') as out_file:
    json.dump(training_stats, out_file, indent = 4, sort_keys = False)  
