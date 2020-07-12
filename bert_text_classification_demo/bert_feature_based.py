import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import torch
import transformers as tfs
import warnings

warnings.filterwarnings('ignore')
train_df = pd.read_csv('F:/python_pycharm/python_projects/train.tsv', delimiter='\t', header=None)
train_set = train_df[:3000]

print("Train set shape:", train_set.shape)
model_class, tokenizer_class, pretrained_weights = (tfs.BertModel, tfs.BertTokenizer, 'bert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)
print(train_set[0].values)
train_tokenized = train_set[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
print(train_set[0])
train_max_len = 0
for i in train_tokenized.values:
    if len(i) > train_max_len:
        train_max_len = len(i)

train_padded = np.array([i + [0] * (train_max_len-len(i)) for i in train_tokenized.values])
print("train set shape:",train_padded.shape)
print(train_padded[0])
train_attention_mask = np.where(train_padded != 0, 1, 0)
print(train_attention_mask[0])
#训练集
train_input_ids = torch.tensor(train_padded).long()
train_attention_mask = torch.tensor(train_attention_mask).long()
with torch.no_grad():
    train_last_hidden_states = model(train_input_ids, attention_mask=train_attention_mask)
    train_features = train_last_hidden_states[0][:, 0, :].numpy()
    train_labels = train_set[1]
    train_features, test_features, train_labels, test_labels = train_test_split(train_features, train_labels)
    lr_clf = LogisticRegression()
    svm = SVC()
    svm.fit(train_features, train_labels)
    lr_clf.fit(train_features, train_labels)
    print(lr_clf.score(test_features, test_labels))
    print(svm.score(test_features, test_labels))
