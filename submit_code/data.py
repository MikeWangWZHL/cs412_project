from transformers import BertTokenizer, BertForMaskedLM
import torch 
import json
import re
import collections
import random
from tqdm import tqdm
import spacy
import nltk

# datasets
# review_path = './dataset/yelp_academic_dataset_review.json'
# review_path_subset_0 = './dataset/review_subsets/review_subset_0.json'

import itertools

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


def get_sub_train_dev_set(dataset):
    data = []
    with open(dataset,encoding = 'utf-8') as f:
        data = json.load(f)
    n = int(len(data)/10)
    subsubset = list(chunks(data,n))
    trainset = list(itertools.chain.from_iterable(subsubset[:-2]))
    devset = list(itertools.chain.from_iterable(subsubset[-2:]))
    print('train sub set length',len(trainset))
    print('dev sub set length',len(devset))
    

    with open('./dataset/review_subsets/devset_subset_1.json','w') as out:
        json.dump(devset,out,indent=4)

    with open('./dataset/review_subsets/trainset_subset_1.json','w') as out:
        json.dump(trainset,out,indent=4)
    
    return trainset,devset

# get_sub_train_dev_set('./dataset/review_subsets/review_subset_1.json')



stop_early = False
stop_count = 10240

def get_inputdata(dataset_path, tokenizer, maxlength = 200, dataset = 'review'):

    if dataset == 'review':
        data_path = dataset_path
        print('using: ',dataset_path)
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
    print('processing input...')
    for item in tqdm(data):
        text = item['text']
        stars = item['stars']
        
        # labelset.append(star)
        
        tokenized = tokenizer(text,return_tensors='pt', max_length = maxlength, padding= 'max_length',truncation = True)
        input_ids = tokenized['input_ids'][0]
        attention_masks = tokenized['attention_mask'][0]
        
        input_ids_list.append(input_ids)
        attention_masks_list.append(attention_masks)
        labelset.append(star_to_label[str(stars)])
        
        if stop_early:
            if count == stop_count:
                break
        
        count+=1
        
    
    # print(input_ids_list)
    input_ids_tensor = torch.stack(input_ids_list)
    attention_masks_tensor = torch.stack(attention_masks_list)
    labels_tensor = torch.tensor(labelset)
    # print(input_ids_tensor)
    # print(attention_masks_tensor)
    # print(labels_tensor)
    return input_ids_tensor,attention_masks_tensor,labels_tensor

def get_input_features(dataset_path, dataset = 'review'):
    
    negword_dict = json.load(open('negative_words.json'))
    positive_dict = json.load(open('positive_words.json'))
    user_dict = json.load(open('./dataset/user_subset_1.json'))
    if dataset == 'review':
        data_path = dataset_path
        print('using: ',dataset_path)
    
    with open(data_path, encoding = 'utf-8') as f:
        data = json.load(f)
    print('dataset length: ',len(data))
    # nlp = spacy.load('en_core_web_sm')
    
    feature_list = []
    count = 0
    print('getting input feature...')
    
    for item in tqdm(data):
        text = item['text']
        # review feature: useful, funny, cool
        review_useful = item['useful']
        review_funny = item['funny']
        review_cool = item['cool']

        # user feature:
        user_id = item['user_id']
        user_object = user_dict[user_id]

        user_review_count = user_object['review_count']
        user_useful = user_object['useful']
        user_funny = user_object['funny']
        user_cool = user_object['cool']
        user_fans = user_object['fans']
        user_avg_stars = user_object['average_stars']

        ## text feature
        parsed_sentence = nltk.word_tokenize(text)

        neg_count = 0
        pos_count = 0
        neg_word_position = []
        pos_word_position = []
        word_count = 0

        # TODO: add n_left and n_right, negating words
        for token in parsed_sentence:
            if token.lower() in negword_dict:
                neg_count += 1
                neg_word_position.append(word_count)
                # print('neg: ',token.text)
            if token.lower() in positive_dict:
                pos_count += 1
                pos_word_position.append(word_count)
                # print('postive: ',token.text)
            word_count += 1
        
        neg_pos_distance = 0
        if neg_count > 0 and pos_count > 0:
            neg_pos_mean = sum(neg_word_position)/neg_count
            pos_pos_mean = sum(pos_word_position)/pos_count
            neg_pos_distance = abs(neg_pos_mean-pos_pos_mean)


        if (neg_count+pos_count) == 0:
            # feature_list.append(torch.tensor([neg_count,pos_count,neg_pos_distance/word_count]))
            feature_list.append(torch.tensor([neg_count,pos_count,neg_pos_distance/word_count,review_useful,review_funny,review_cool,user_review_count,user_useful,user_funny,user_cool,user_fans,user_avg_stars/5]))
        else:
            # feature_list.append(torch.tensor([neg_count/(neg_count+pos_count),pos_count/(neg_count+pos_count),neg_pos_distance/word_count]))
            # feature_list.append(torch.tensor([neg_count/(neg_count+pos_count),pos_count/(neg_count+pos_count),neg_pos_distance/word_count]))
            feature_list.append(torch.tensor([neg_count/(neg_count+pos_count),pos_count/(neg_count+pos_count),neg_pos_distance/word_count,review_useful,review_funny,review_cool,user_review_count,user_useful,user_funny,user_cool,user_fans,user_avg_stars/5]))
        
        # feature_list.append(torch.tensor([review_useful,review_funny,review_cool,user_review_count,user_useful,user_funny,user_cool,user_fans,user_avg_stars/5]))
        # feature_list.append(torch.tensor([user_avg_stars/5]))

        # quit()
        if stop_early:
            if count == stop_count:
                break
        count+=1
        
    # print(feature_list)
    # quit()
    featurelist_tensor = torch.stack(feature_list)
    # print(featurelist_tensor)
    return featurelist_tensor

