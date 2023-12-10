import torch
from torch.utils.data import Dataset

import os
import csv

class ClassificationDataset(Dataset):
  def __init__(self, data_path, tokenizer, model):
    # Por padrão sempre escolhe a GPU se disponível
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    self.data_path = data_path
    self.tokenizer = tokenizer
    self.model = model.to(self.device)

    self.docs = self.__get_docs()

  def __len__(self):
    return len(self.docs)

  def __getitem__(self, idx):
    inputs, label = self.docs[idx]

    encoded_input = self.__get_encoded_input(inputs)
    encoded_label = self.__get_encoded_label(label)

    return encoded_input, encoded_label

  def __get_docs(self):
    docs = {}

    for f in os.listdir(self.data_path + 'less'):
        file_path = self.data_path + 'less' + '/' + f

        idx = int(f[10:f.find('.csv')])
        docs[idx] = ([], 0)

        with open(file_path, 'r', encoding='utf-8') as d:
            csv_reader = csv.DictReader(d, delimiter=';')

            for row in csv_reader:
                try:
                    doc = row['tweet']

                    docs[idx][0].append(doc)
                except Exception as e:
                    print(e)

    for f in os.listdir(self.data_path + 'more'):
        file_path = self.data_path + 'more' + '/' + f

        idx = int(f[10:f.find('.csv')])
        docs[idx] = ([], 1)

        with open(file_path, 'r', encoding='utf-8') as d:
            csv_reader = csv.DictReader(d, delimiter=';')

            for row in csv_reader:
                try:
                    doc = row['tweet']

                    docs[idx][0].append(doc)
                except Exception as e:
                    print(e)

    return docs

  def __get_encoded_input(self, docs):

    embedded_docs = []

    for doc in docs:
      if len(doc) > 0:
        encoded_doc = self.tokenizer(doc, return_tensors='pt').to(self.device)

        with torch.no_grad():
          embedded_doc = self.model(encoded_doc)[0][0, 1:-1]

        pooled_doc = torch.mean(embedded_doc, dim=0).unsqueeze(0)
        embedded_docs.append(pooled_doc)

    if embedded_docs:
      embedded_docs = torch.cat(embedded_docs)
    pooled_docs = torch.mean(embedded_docs, dim=0)

    return pooled_docs

  def __get_encoded_label(self, label):
    encoded = torch.tensor(label).to(self.device)

    return encoded
  
class SimpleDataset(Dataset):
  def __init__(self, data_path, tokenizer, model):
    # Por padrão sempre escolhe a GPU se disponível
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    self.data_path = data_path
    self.tokenizer = tokenizer
    self.model = model.to(self.device)

    self.docs = self.__get_docs()

  def __len__(self):
    return len(self.docs)

  def __getitem__(self, idx):
    inputs, label = self.docs[idx]

    encoded_input = self.__get_encoded_input(inputs)
    encoded_label = self.__get_encoded_label(label)

    return encoded_input, encoded_label

  def __get_docs(self):
    docs = {}

    for f in os.listdir(self.data_path + '/' + 'less'):
        file_path = self.data_path + '/' 'less' + '/' + f

        idx = int(f[10:f.find('.csv')])
        docs[idx] = ([], 0)

        with open(file_path, 'r', encoding='utf-8') as d:
            csv_reader = csv.DictReader(d, delimiter=';')

            for row in csv_reader:
                try:
                    doc = row['tweet']

                    docs[idx][0].append(doc)
                except Exception as e:
                    print(e)

    for f in os.listdir(self.data_path + '/' + 'more'):
        file_path = self.data_path + '/' 'more' + '/' + f

        idx = int(f[10:f.find('.csv')])
        docs[idx] = ([], 1)

        with open(file_path, 'r', encoding='utf-8') as d:
            csv_reader = csv.DictReader(d, delimiter=';')

            for row in csv_reader:
                try:
                    doc = row['tweet']

                    docs[idx][0].append(doc)
                except Exception as e:
                    print(e)

    return docs

  def __get_encoded_input(self, docs):

    doc = ' '.join(docs)
    encoded_doc = self.tokenizer(doc, return_tensors='pt').to(self.device)

    encoded_doc = encoded_doc[:, min(-512, encoded_doc.shape[1]):]

    with torch.no_grad():
      embedded_doc = self.model(encoded_doc)[0][0, 1:-1]

    pooled_doc = torch.mean(embedded_doc, dim=0)

    return pooled_doc

  def __get_encoded_label(self, label):
    encoded = torch.tensor(label).to(self.device)

    return encoded
