#91.25%
import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Mapping labels to integer values
LABEL_MAP = {'anger': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'neutral': 5}

class AudioPreprocessor:
    def __init__(self, sr=22050, n_mels=128, max_len=128):
        self.sr = sr
        self.n_mels = n_mels
        self.max_len = max_len
    
    def pad_or_truncate(self, mel_spec):
        if mel_spec.shape[1] > self.max_len:
            return mel_spec[:, :self.max_len]
        else:
            pad_width = self.max_len - mel_spec.shape[1]
            return np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
    
    def process_audio(self, file_path):
        try:
            audio, sr = librosa.load(file_path, sr=self.sr)
            mel_spec = librosa.feature.melspectrogram(
                y=audio, 
                sr=sr, 
                n_mels=self.n_mels, 
                fmax=sr//2
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec_db = self.pad_or_truncate(mel_spec_db)
            return mel_spec_db
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

def load_dataset(data_path, preprocessor):
    X = []
    y = []
    total_files = sum(len(os.listdir(os.path.join(data_path, label))) 
                     for label in os.listdir(data_path))
    
    with tqdm(total=total_files, desc="Processing audio files") as pbar:
        for label in os.listdir(data_path):
            if label not in LABEL_MAP:
                continue
            
            label_path = os.path.join(data_path, label)
            for file_name in os.listdir(label_path):
                if not file_name.lower().endswith('.wav'):
                    continue
                
                file_path = os.path.join(label_path, file_name)
                mel_spec_db = preprocessor.process_audio(file_path)
                
                if mel_spec_db is not None:
                    X.append(mel_spec_db)
                    y.append(LABEL_MAP[label])
                
                pbar.update(1)
    
    return np.array(X), np.array(y)

class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ImprovedAudioEmotionModel(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.5):
        super(ImprovedAudioEmotionModel, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate)
        )
        
        self._to_linear = None
        self._get_conv_output((1, 128, 128))
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self._to_linear, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, num_classes)
        )
    
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.rand(batch_size, *shape)
        output = self.conv_layers(input)
        n_size = output.data.view(batch_size, -1).size(1)
        self._to_linear = n_size

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')  # Initialize with infinity instead of None
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss is None:
            raise ValueError("Validation loss is None. Check the validation data and model output.")
        
        if val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, 
                 scheduler, device, early_stopping):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.early_stopping = early_stopping
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'train_accuracy': []
        }

    def train_epoch(self, epoch, num_epochs):
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        with tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return train_loss / len(self.train_loader), correct / total

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(self.val_loader)
        val_acc = correct / total
        
        # Debug: Check if validation loss is None
        if avg_val_loss is None:
            raise ValueError("Validation loss is None. Check the validation data and model output.")
        
        return avg_val_loss, val_acc

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            avg_train_loss, train_acc = self.train_epoch(epoch, num_epochs)
            avg_val_loss, val_acc = self.validate()
            
            # Debug: Check if validation loss is None
            if avg_val_loss is None:
                raise ValueError("Validation loss is None. Check the validation data and model output.")
            
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['val_accuracy'].append(val_acc)
            
            self.scheduler.step(avg_val_loss)
            
            print(f'\nEpoch {epoch+1}/{num_epochs}:')
            print(f'Training Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.4f}')
            print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.4f}')
            
            if avg_val_loss < getattr(self.early_stopping, 'best_loss', float('inf')):
                torch.save(self.model.state_dict(), 'best_emotion_model.pth')
                print("Saved new best model")
            
            self.early_stopping(avg_val_loss)
            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
        return self.history

class ModelAnalyzer:
    def __init__(self, model, test_loader, device, class_names):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names

    def evaluate(self):
        self.model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        return np.array(y_true), np.array(y_pred)

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        return cm, cm_normalized

    def plot_training_history(self, history):
        plt.figure(figsize=(15, 5))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(history['train_accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()

    def analyze_overfitting(self, history):
        final_train_loss = history['train_loss'][-1]
        final_val_loss = history['val_loss'][-1]
        loss_diff = abs(final_train_loss - final_val_loss)
        
        final_train_acc = history['train_accuracy'][-1]
        final_val_acc = history['val_accuracy'][-1]
        acc_diff = abs(final_train_acc - final_val_acc)
        
        return {
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'loss_difference': loss_diff,
            'final_train_acc': final_train_acc,
            'final_val_acc': final_val_acc,
            'acc_difference': acc_diff
        }

def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize preprocessor and load data
    print("Loading and preprocessing audio data...")
    preprocessor = AudioPreprocessor()
    audio_data_path = r'D:\New\speech'  # Update this path
    X, y = load_dataset(audio_data_path, preprocessor)
    
    # Check for None values
    none_indices = [i for i, x in enumerate(X) if x is None]
    if none_indices:
        print(f"Found {len(none_indices)} None values in the dataset. Removing them.")
        X = [x for x in X if x is not None]
        y = [y[i] for i in range(len(y)) if i not in none_indices]
        X = np.array(X)
        y = np.array(y)
    
    # Normalize spectrograms
    X = (X - np.min(X)) / (np.max(X) - np.min(X) + 1e-8)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # Create data loaders
    train_dataset = AudioDataset(X_train, y_train)
    val_dataset = AudioDataset(X_val, y_val)
    test_dataset = AudioDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model and training components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedAudioEmotionModel().to(device)
    
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    early_stopping = EarlyStopping(patience=7)
    
    # Initialize trainer and train model
    print("\nTraining the model...")
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        early_stopping=early_stopping
    )
    
    history = trainer.train(num_epochs=50)
    
    # Initialize analyzer and evaluate model
    print("\nAnalyzing model performance...")
    class_names = list(LABEL_MAP.keys())
    analyzer = ModelAnalyzer(model, test_loader, device, class_names)
    
    # Load best model for evaluation
    model.load_state_dict(torch.load('best_emotion_model.pth', weights_only=True))
    
    # Get predictions
    y_true, y_pred = analyzer.evaluate()
    
    # Plot confusion matrix
    cm, cm_normalized = analyzer.plot_confusion_matrix(y_true, y_pred)
    
    # Plot training history
    analyzer.plot_training_history(history)
    
    # Analyze overfitting
    overfitting_metrics = analyzer.analyze_overfitting(history)
    
    # Print results
    print("\nModel Performance Results:")
    print("-" * 50)
    print(f"Test Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    print("\nOverfitting Analysis:")
    print("-" * 50)
    print(f"Final Training Loss: {overfitting_metrics['final_train_loss']:.4f}")
    print(f"Final Validation Loss: {overfitting_metrics['final_val_loss']:.4f}")
    print(f"Loss Difference: {overfitting_metrics['loss_difference']:.4f}")
    print(f"Final Training Accuracy: {overfitting_metrics['final_train_acc']:.4f}")
    print(f"Final Validation Accuracy: {overfitting_metrics['final_val_acc']:.4f}")
    print(f"Accuracy Difference: {overfitting_metrics['acc_difference']:.4f}")
    
    # Print confusion matrix insights
    print("\nConfusion Matrix Analysis:")
    print("-" * 50)
    for i, emotion in enumerate(class_names):
        correct = cm_normalized[i, i] * 100
        incorrect_indices = [j for j in range(len(class_names)) if j != i]
        most_confused_idx = incorrect_indices[np.argmax(cm_normalized[i, incorrect_indices])]
        confusion_rate = cm_normalized[i, most_confused_idx] * 100
        
        print(f"\n{emotion.capitalize()}:")
        print(f"Correct Classification Rate: {correct:.1f}%")
        print(f"Most confused with: {class_names[most_confused_idx]} ({confusion_rate:.1f}%)")

def predict_single_audio(model, file_path, preprocessor, device):
    """
    Make a prediction on a single audio file
    """
    model.eval()
    mel_spec_db = preprocessor.process_audio(file_path)
    
    if mel_spec_db is None:
        return None
    
    # Normalize and prepare input
    mel_spec_db = (mel_spec_db - np.min(mel_spec_db)) / (np.max(mel_spec_db) - np.min(mel_spec_db) + 1e-8)
    input_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1)
    
    emotion_idx = prediction.item()
    probability = probabilities[0][emotion_idx].item()
    
    # Get emotion label
    emotion = [k for k, v in LABEL_MAP.items() if v == emotion_idx][0]
    
    return emotion, probability

if __name__ == "__main__":
    try:
        main()
        
        # Example of how to use the model for prediction
        print("\nExample: Making prediction on a single audio file")
        print("-" * 50)
        
        # Load the trained model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ImprovedAudioEmotionModel().to(device)
        model.load_state_dict(torch.load('best_emotion_model.pth', weights_only=True))
        
        # Initialize preprocessor
        preprocessor = AudioPreprocessor()
        
        # Example prediction (replace with actual file path)
        test_file = "path_to_test_audio.wav"
        if os.path.exists(test_file):
            emotion, confidence = predict_single_audio(model, test_file, preprocessor, device)
            print(f"Predicted emotion: {emotion}")
            print(f"Confidence: {confidence:.2%}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")