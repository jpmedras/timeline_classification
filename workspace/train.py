from transformers import AutoTokenizer  # Or BertTokenizer
from transformers import AutoModelForPreTraining  # Or BertForPreTraining for loading pretraining heads

from model import Classifier
from dataset import SimpleDataset

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Generator

from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt

tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False).encode
model = AutoModelForPreTraining.from_pretrained('neuralmind/bert-base-portuguese-cased')
dataset = SimpleDataset('../data/', tokenizer, model)

lengths = [0.8, 0.2]
batch_size = 16

train_set, test_set = random_split(dataset, lengths, Generator().manual_seed(4))

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, drop_last=True, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, drop_last=True, shuffle=False)

torch.manual_seed(4)

params = {
  'embedding_dim': 29794,
  'output_size': 2,
  'dropout_rate': 0.3
}

model = Classifier(**params)

num_epochs = 20

fit_params = {
    'train_loader': train_loader,
    'test_loader': test_loader,
    'num_epochs': num_epochs
}

train_losses, test_losses = model.fit(**fit_params)

# Plot the learning curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', linestyle='-', color='b', label='Training Loss')
plt.plot(range(1, num_epochs + 1), test_losses, marker='x', linestyle='-', color='g', label='Test Loss')

plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Average Loss')
plt.legend()
plt.show()

labels = ['Non-toxic', 'Toxic']

accuracy, cm  = model.evaluate(train_loader)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=labels)

disp.plot()
plt.show()

accuracy, cm  = model.evaluate(test_loader)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=labels)

disp.plot()
plt.show()
