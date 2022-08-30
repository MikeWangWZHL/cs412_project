import json

# word_list = []
# with open('positive_words',encoding='utf-8') as f:
#     for line in f:
#         word_list.append(line.strip())

# with open('positive_words.json','w') as out:
#     json.dump(word_list,out,indent=4)
# print(word_list)


word_dict = {}
with open('negation_words.json') as f:
    data = json.load(f)
for w in data:
    word_dict[w] = True
with open('negation_words.json','w') as out:
    json.dump(word_dict,out,indent=4)
