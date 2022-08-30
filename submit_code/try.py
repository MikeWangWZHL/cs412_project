from transformers import BertTokenizer, BertConfig

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

tokenized = tokenizer('I can\'t do this this is not right, I won\'t do this')

print(tokenizer.convert_ids_to_tokens(tokenized['input_ids']))