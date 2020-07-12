import torch
from torch import nn
from torch import optim
import transformers as tfs
from sklearn.model_selection import train_test_split
import pandas as pd

train_df = pd.read_csv('F:/python_pycharm/python_projects/train.tsv', delimiter='\t', header=None)
train_set = train_df[:3000]
print("Train set shape:", train_set.shape)
train_set[1].value_counts()


class BertClassificationModel(nn.Module):
    def __init__(self):
        super(BertClassificationModel, self).__init__()
        model_class, tokenizer_class, pretrained_weights = (tfs.BertModel, tfs.BertTokenizer, 'bert-base-uncased')
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.bert = model_class.from_pretrained(pretrained_weights)
        self.dense = nn.Linear(768, 2)  # bert默认的隐藏单元数是768， 输出单元是2，表示二分类

    def forward(self, batch_sentences):
        # print(batch_sentences[4])
        batch_tokenized = self.tokenizer.batch_encode_plus(batch_sentences, add_special_tokens=True,
                                                           max_length=66,
                                                           pad_to_max_length=True,truncation = True)  # tokenize、add special token、pad
        input_ids = torch.tensor(batch_tokenized['input_ids'])
        attention_mask = torch.tensor(batch_tokenized['attention_mask'])
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[0][:, 0, :]  # 提取[CLS]对应的隐藏状态
        linear_output = self.dense(bert_cls_hidden_state)
        return linear_output

sentences = train_set[0].values
targets = train_set[1].values
train_inputs, test_inputs, train_targets, test_targets = train_test_split(sentences, targets)

batch_size = 64
batch_count = int(len(train_inputs) / batch_size)
batch_train_inputs, batch_train_targets = [], []
for i in range(batch_count):
    batch_train_inputs.append(train_inputs[i * batch_size: (i + 1) * batch_size])
    batch_train_targets.append(train_targets[i * batch_size: (i + 1) * batch_size])
# train the model
epochs = 6
lr = 0.01
print_every_batch = 10
bert_classifier_model = BertClassificationModel()
optimizer = optim.SGD(bert_classifier_model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    print_avg_loss = 0
    for i in range(batch_count):
        inputs = batch_train_inputs[i]
        labels = torch.tensor(batch_train_targets[i])
        optimizer.zero_grad()
        outputs = bert_classifier_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print_avg_loss += loss.item()
        if i % print_every_batch == (print_every_batch - 1):
            print("Batch: %d, Loss: %.4f" % ((i + 1), print_avg_loss / print_every_batch))
            print_avg_loss = 0

# eval the trained model
total = len(test_inputs)
hit = 0
with torch.no_grad():
    for i in range(total):
        outputs = bert_classifier_model([test_inputs[i]])
        _, predicted = torch.max(outputs, 1)
        if predicted == test_targets[i]:
            hit += 1

print("Accuracy: %.2f%%" % (hit / total * 100))
