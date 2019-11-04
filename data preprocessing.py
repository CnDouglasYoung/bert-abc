import pandas as pd


data = pd.read_csv('data/train.csv',encoding='utf-8')
interesting_feat = ['text','class_label']
saving_data = data[interesting_feat]
saving_data = saving_data.rename(index=str, columns={"text": "question", "class_label": "label"})
print(saving_data)

from sklearn.model_selection import train_test_split

train_data,val_data = train_test_split(saving_data, test_size=0.20, random_state=42)

data = pd.read_csv('data/test.csv',encoding='utf-8')
interesting_feat = ['text','class_label']
test_data = data[interesting_feat]
test_data = test_data.rename(index=str, columns={"text": "question", "class_label": "label"})


train_data.to_json('data/train.json', orient='records')
test_data.to_json('data/test.json', orient='records')
val_data.to_json('data/val.json', orient='records')



import re

for s in ['train','test','val']:
    path = 'data/%s.json'%str(s)
    with open(path,'r') as r:
        content = r.read()
        content = re.sub(r'},{','}\n{',content[1:-1])
    with open(path, 'w') as r:
        r.write(content)