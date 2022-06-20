import torch
import time
import math
import gzip
import csv
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence


class NameDataset(Dataset):
    def __init__(self, is_train_set=True):
        filename = './dataset/names_train.csv.gz' if is_train_set else './dataset/names_test.csv.gz'
        with gzip.open(filename, 'rt') as f:
            reader = csv.reader(f)
            rows = list(reader)
        self.names = [row[0] for row in rows]
        self.len = len(self.names)
        self.countries = [row[1] for row in rows]
        self.country_list = list(sorted(set(self.countries)))  # 去重
        self.country_dict = self.getCountryDict()
        self.country_num = len(self.country_list)

    def __getitem__(self, index):
        return self.names[index], self.country_dict[self.countries[index]]

    def __len__(self):
        return self.len

    # 建立国家名字和索引的对应关系
    def getCountryDict(self):
        country_dict = dict()
        for idx, country_name in enumerate(self.country_list, 0):
            country_dict[country_name] = idx  # 国家和索引对应
        return country_dict

    # 从索引获得国家的名字
    def idx2country(self, index):
        return self.country_list[index]

    # 获得国家的数量
    def getCountriesNum(self):
        return self.country_num


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# Parameters
HIDDEN_SIZE = 100
BATCH_SIZE = 256
N_LAYER = 2  # 网络层数
N_EPOCHS = 100  # 训练轮次
N_CHARS = 128  # 输入维度
USE_GPU = False


trainset = NameDataset()
trainLoader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testset = NameDataset(is_train_set=False)
testLoader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
N_COUNTRY = trainset.getCountriesNum()


class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_direction = 2 if bidirectional else 1

        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.gru = torch.nn.LSTM(hidden_size, hidden_size, n_layers, bidirectional=bidirectional)
        self.fc = torch.nn.Linear(hidden_size * self.n_direction, output_size)  # 双向传播会导致输出的 hidden_size 翻倍

    def forward(self, input, seq_lengths):
        # input shape: batchSize x seqLen -> seqLen x batchSize
        input = input.t()
        batch_size = input.size(1)

        hidden = self._init_hidden(batch_size)
        cell = self._init_cell(batch_size)
        embedding = self.embedding(input)

        # pack them up
        lstm_input = pack_padded_sequence(embedding, seq_lengths)

        output, (hidden, cell) = self.gru(lstm_input, (hidden, cell))
        if self.n_direction == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]
        fc_output = self.fc(hidden_cat)
        return fc_output

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_direction,
                             batch_size,
                             self.hidden_size)
        return hidden

    def _init_cell(self, batch_size):
        cell = torch.randn(self.n_layers * self.n_direction,
                           batch_size,
                           self.hidden_size)
        return cell


def name2list(name):
    arr = [ord(c) for c in name]  # 将字符转换成 ASCII 码
    return arr, len(arr)


def create_tensor(tensor):
    if USE_GPU:
        device = torch.device('cuda:0')
        tensor = tensor.to(device)
    return tensor


def make_tensors(names, countries):
    sequences_and_lengths = [name2list(name) for name in names]
    name_sequences = [s1[0] for s1 in sequences_and_lengths]
    seq_lengths = torch.LongTensor([s1[1] for s1 in sequences_and_lengths])
    countries = countries.long()

    # name tensor(batchSize x seqLen), padding
    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # 降序排序，为了使用 pack_padded_sequence
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]

    return create_tensor(seq_tensor), create_tensor(seq_lengths), create_tensor(countries)


def trainModel():
    total_loss = 0
    for i, (names, countries) in enumerate(trainLoader, 1):
        inputs, seq_lengths, target = make_tensors(names, countries)
        output = classifier(inputs, seq_lengths)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 10 == 0:
            print(f'[{time_since(start)}] Epoch {epoch}', end='')
            print(f'[{i * len(inputs)} / {len(trainset)}]', end='')
            print(f'loss={total_loss / (i * len(inputs))}')
    return total_loss


def predictModel():
    correct = 0
    total = len(testset)
    print("evaluating trained model ...")
    with torch.no_grad():
        for i, (names, countries) in enumerate(testLoader, 1):
            inputs, seq_lengths, target = make_tensors(names, countries)
            output = classifier(inputs, seq_lengths)
            _, pred = output.max(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        percent = '%.2f' % (100 * correct / total)
        print(f'Test set: Accuracy {correct}/{total} {percent}%')
    return correct / total


if __name__ == '__main__':
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER)
    if USE_GPU:
        device = torch.device('cuda:0')
        classifier.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    start = time.time()
    print("Training for %d epochs..." % N_EPOCHS)
    acc_list = []
    for epoch in range(1, N_EPOCHS + 1):
        trainModel()
        acc = predictModel()
        acc_list.append(acc)

    epoch_list = np.arange(1, len(acc_list) + 1, 1)
    acc_list = np.array(acc_list)
    plt.plot(epoch_list, acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()
