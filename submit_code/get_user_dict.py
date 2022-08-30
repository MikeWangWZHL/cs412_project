import json

user_set = set()
with open('./dataset/review_subsets/trainset_subset_1.json') as f:
    for item in json.load(f):
        user_set.add(item['user_id'])
with open('./dataset/review_subsets/devset_subset_1.json') as f:
    for item in json.load(f):
        user_set.add(item['user_id'])

print(len(user_set))

data = []
with open('./dataset/yelp_academic_dataset_user.json',encoding = 'utf-8') as f:
    for line in f:
        data.append(json.loads(line))
user_subset = {}
for item in data:
    if item['user_id'] in user_set:
        user_subset[item['user_id']] = item

print(user_subset['FOBRPlBHa3WPHFB5qYDlVg'])
# print(len(user_subset))
with open('./dataset/user_subset_1.json','w') as out:
    json.dump(user_subset,out,indent=4)



# with open('./dataset/user_subset_1.json') as f:
#     data = json.load(f)
# print(data[1])

