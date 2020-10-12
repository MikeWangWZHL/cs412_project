from transformers import BertTokenizer, BertForMaskedLM
import torch 
import json
import re
import collections
import random

# datasets
review_path = './dataset/yelp_academic_dataset_review.json'
review_path_subset_0 = './dataset/review_subsets/review_subset_0.json'


#label dict
star_to_label = {'1.0':0,'2.0':1,'3.0':2,'4.0':3,'5.0':4}

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_subset(dataset,output_dir):
    data = []
    with open(dataset,encoding = 'utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    random.shuffle(data)
    n = int(len(data)/10)
    subsets = list(chunks(data,n))
    print('subset number: ',len(subsets))

    for i in range(len(subsets)):
        out_name = output_dir + '/review_subset_' + str(i) + '.json'
        with open(out_name,'w') as out_file:
            json.dump(subsets[i], out_file, indent = 4, sort_keys = False) 
        print('subset length:',len(subsets[i]))



def get_inputdata(tokenizer, maxlength = 200, dataset = 'review'):

    if dataset == 'review':
        data_path = review_path_subset_0
        print('note using review subset 0')
    # data = []
    # with open(data_path,encoding = 'utf-8') as f:
    #     for line in f:
    #         data.append(json.loads(line))
    
    with open(data_path, encoding = 'utf-8') as f:
        data = json.load(f)
    print('dataset length: ',len(data))
    
    input_ids_list = []
    attention_masks_list = []
    labelset = []
    count = 0
    for item in data:
        text = item['text']
        stars = item['stars']
        
        # labelset.append(star)
        if len(text) > 200:
            continue
        tokenized = tokenizer(text,return_tensors='pt', max_length = maxlength, padding= 'max_length')
        input_ids = tokenized['input_ids'][0]
        attention_masks = tokenized['attention_mask'][0]
        
        input_ids_list.append(input_ids)
        attention_masks_list.append(attention_masks)
        labelset.append(star_to_label[str(stars)])
        # if count == 2:
        #     break
        count+=1
    # print(input_ids_list)
    input_ids_tensor = torch.stack(input_ids_list)
    attention_masks_tensor = torch.stack(attention_masks_list)
    labels_tensor = torch.tensor(labelset)
    # print(input_ids_tensor)
    # print(attention_masks_tensor)
    # print(labels_tensor)
    return input_ids_tensor,attention_masks_tensor,labels_tensor

