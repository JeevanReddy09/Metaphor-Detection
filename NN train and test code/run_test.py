import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib
import csv
import nltk
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import joblib  

# Ensure necessary NLTK downloads
nltk.download('punkt')

# Define your metaphor words dictionary and other necessary functions
metaphor_words = {
    0: 'road',
    1: 'candle',
    2: 'light',
    3: 'spice',
    4: 'ride',
    5: 'train',
    6: 'boat'
}

def extract_context(text, metaphor_id, window_size=5):
    target_word = metaphor_words[metaphor_id]
    words = word_tokenize(text)
    if target_word in words:
        target_index = words.index(target_word)
        start = max(0, target_index - window_size)
        end = min(len(words), target_index + window_size + 1)
        return ' '.join(words[start:end])
    else:
        return ''

# PyTorch Dataset
class MetaphorDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx].astype(int), dtype=torch.long)

# Neural Network Model
class MetaphorNN(nn.Module):
    def __init__(self, input_size):
        super(MetaphorNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # 2 for binary classification

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def test_model(test_file, model_file):
    df = pd.read_csv(test_file)
    df['context'] = df.apply(lambda row: extract_context(row['text'], row['metaphorID']), axis=1)
    df = df[df['context'] != '']

    # Load the saved TF-IDF Vectorizer
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
    X_test = tfidf_vectorizer.transform(df['context']).toarray()
    y_test = df['label_boolean'].values

    test_dataset = MetaphorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = MetaphorNN(input_size=X_test.shape[1])
    model.load_state_dict(torch.load(model_file))
    model.eval()

    predictions, labels = [], []
    with torch.no_grad():
        for inputs, label in test_loader:
            output = model(inputs)
            _, predicted = torch.max(output.data, 1)
            predictions.extend(predicted.numpy())
            labels.extend(label.numpy())

    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')

    # Save predictions
    with open('predictions.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['True Label', 'Predicted Label'])
        writer.writerows(zip(labels, predictions))

    # Print metrics
    print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}")



if __name__ == "__main__":
    test_model('test.csv', 'metaphor_model.pth')
print(pd.read_csv('predictions.csv'))