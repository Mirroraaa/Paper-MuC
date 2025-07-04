import os
from collections import defaultdict
import torch
from torch.utils.data.dataset import Dataset


class KnowledgeGraph:
    def __init__(self, path, dataset):
        entity_path = os.path.join(path, dataset, 'entity2id.txt')
        relation_path = os.path.join(path, dataset, 'relation2id.txt')
        train_path = os.path.join(path, dataset, 'train.txt')
        valid_path = os.path.join(path, dataset, 'valid.txt')
        test_path = os.path.join(path, dataset, 'test.txt')

        self.entity2id = {}
        self.relation2id = {}
        with open(entity_path, 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                entity, i = line.strip().split('\t')
                self.entity2id[str(entity)] = int(i)

        with open(relation_path, 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                relation, i = line.strip().split('\t')
                self.relation2id[str(relation)] = int(i)
        self.num_entities = len(self.entity2id)
        self.num_relations = len(self.relation2id)

        self.train_data = self.read_data(train_path)
        self.valid_data = self.read_data(valid_path)
        self.test_data = self.read_data(test_path)

        self.valid_hr_vocab = defaultdict(list)
        self.test_hr_vocab = defaultdict(list)
        for triplet in self.train_data + self.valid_data:
            self.valid_hr_vocab[(triplet[0], triplet[1])].append(triplet[2])
        for triplet in self.train_data + self.valid_data + self.test_data:
            self.test_hr_vocab[(triplet[0], triplet[1])].append(triplet[2])

    def read_data(self, file_path):
        data = []
        with open(file_path, 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                h, r, t = line.strip().split('\t')
                # 原始三元组
                data.append([
                    self.entity2id[h],
                    self.relation2id[r],
                    self.entity2id[t]
                ])
        return data


class KGDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(data):
        heads = [_[0] for _ in data]
        relations = [_[1] for _ in data]
        tails = [_[2] for _ in data]
        return heads, relations, tails

