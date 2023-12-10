import torch
from torch import nn

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

class Classifier(nn.Module):
  def __init__(self, embedding_dim, output_size, dropout_rate):
    super(Classifier,self).__init__()
    self.fc1 = nn.Linear(embedding_dim, embedding_dim//(2**2))
    self.fc2 = nn.Linear(embedding_dim//(2**2), embedding_dim//(2**4))
    self.fc3 = nn.Linear(embedding_dim//(2**4), embedding_dim//(2**6))
    self.fc4 = nn.Linear(embedding_dim//(2**6), output_size)
    self.dropout = nn.Dropout(dropout_rate)

    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.to(self.device)

  def forward(self,x):
    out = self.fc1(x)
    out = self.dropout(out)
    out = self.fc2(out)
    out = self.dropout(out)
    out = self.fc3(out)
    out = self.dropout(out)
    out = self.fc4(out)

    return out

  def fit(self, train_loader, test_loader, num_epochs=10, learnig_rate=0.001, outpath='tweetClassification.pth'):
    train_losses = []
    test_losses = []

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(self.parameters(),lr=learnig_rate)

    self.to(self.device)

    self.train()
    for epoch in range(num_epochs):
      total_loss = 0.0

      for inputs, labels in train_loader :
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        optimizer.zero_grad()

        outputs = self.forward(inputs)

        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

      average_train_loss = total_loss / len(train_loader)
      average_test_loss = self.__calc_loss(dataloader=test_loader, loss_function=loss_function)

      train_losses.append(average_train_loss)
      test_losses.append(average_test_loss)

      train_accuracy, _ = self.evaluate(train_loader)
      test_accuracy, _ = self.evaluate(test_loader)

      print(f"Epoch {epoch+1}/{num_epochs} Train Loss: {average_train_loss:.4f} ({train_accuracy}) | Test Loss: {average_test_loss} ({test_accuracy:.2f})")

    torch.save(self.state_dict(), outpath)
    return train_losses, test_losses

  def __calc_loss(self, dataloader, loss_function):
    with torch.no_grad():
        total_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)

            y_hat = self(x)

            loss = loss_function(y_hat, y)

            total_loss += loss.item()

        return total_loss/len(dataloader)

  def evaluate(self, dataloader):
    self.eval()

    with torch.no_grad():
      predictions = []
      labels = []

      for x, y in dataloader:
        x, y = x.to(self.device), y.to(self.device)

        y_hat = self(x)

        predicted = torch.argmax(y_hat, dim=1)

        predictions.extend(predicted.cpu().numpy())
        labels.extend(y.cpu().numpy())

    accuracy = accuracy_score(labels, predictions)
    confusion_mat = confusion_matrix(labels, predictions)

    return accuracy, confusion_mat
