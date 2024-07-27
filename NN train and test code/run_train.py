import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import joblib  # for saving the vectorizer

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

def train_model(train_file):
    df = pd.read_csv(train_file)
    df['context'] = df.apply(lambda row: extract_context(row['text'], row['metaphorID']), axis=1)
    df = df[df['context'] != '']

    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = tfidf_vectorizer.fit_transform(df['context']).toarray()
    y = df['label_boolean'].values

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.1, random_state=42)
    train_dataset = MetaphorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    model = MetaphorNN(input_size=X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(10):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Save the model and the vectorizer
    torch.save(model.state_dict(), 'metaphor_model.pth')
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')

if __name__ == "__main__":
    train_model('train.csv')
