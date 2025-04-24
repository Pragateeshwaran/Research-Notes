import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Function to load audio and generate spectrogram
def generate_spectrogram(audio_path, sr=22050, n_mels=128, hop_length=512):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Generate mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    
    # Convert to log scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return log_mel_spec

# Function to prepare spectrogram for CNN
def prepare_for_cnn(spectrogram, target_shape=(128, 128)):
    # Ensure spectrogram is 2D
    spec = np.expand_dims(spectrogram, axis=-1)  # Add channel dimension (height, width, 1)
    
    # Resize spectrogram to target shape if needed (using simple interpolation)
    spec_resized = tf.image.resize(spec, target_shape).numpy()
    
    # Normalize to [0, 1]
    spec_normalized = (spec_resized - np.min(spec_resized)) / (np.max(spec_resized) - np.min(spec_resized))
    
    return spec_normalized

# Function to build a simple CNN model
def build_cnn_model(input_shape=(128, 128, 1), num_classes=2):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# Main execution
if __name__ == "__main__":
    # Path to your audio file
    audio_path = 'audio.wav'
    
    # Generate spectrogram
    spectrogram = generate_spectrogram(audio_path)
    
    # Prepare spectrogram for CNN
    cnn_input = prepare_for_cnn(spectrogram)
    
    # Add batch dimension for TensorFlow (1, height, width, channels)
    cnn_input = np.expand_dims(cnn_input, axis=0)
    
    # Build CNN model
    model = build_cnn_model()
    
    # Print model summary
    model.summary()
    
    # Example: Predict (assuming model is trained)
    # For demonstration, we're just passing the input through the untrained model
    prediction = model.predict(cnn_input)
    print("Prediction:", prediction)
    
    # Optional: Save spectrogram visualization
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, sr=22050, hop_length=512, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.savefig('spectrogram.png')
    plt.close()










import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Sample corpus (replace with your own text data if needed)
corpus = """
The quick brown fox jumps over the lazy dog. 
The dog sleeps while the fox runs fast.
"""

# Parameters
n_words = 3  # Number of input words to predict the next word
max_vocab_size = 1000  # Maximum vocabulary size
embedding_dim = 50  # Size of embedding vectors
sequence_length = n_words  # Length of input sequences

# Step 1: Preprocess the text
# Split corpus into words
words = corpus.lower().split()
unique_words = list(set(words))

# Step 2: Create sequences of n words + next word
def create_sequences(words, n):
    sequences = []
    targets = []
    for i in range(len(words) - n):
        seq = words[i:i + n]
        target = words[i + n]
        sequences.append(' '.join(seq))
        targets.append(target)
    return sequences, targets

input_sequences, target_words = create_sequences(words, n_words)

# Step 3: Vectorize the text
# Use TextVectorization to convert words to integers
vectorize_layer = layers.TextVectorization(
    max_tokens=max_vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length
)

# Adapt the vectorizer to the input sequences
vectorize_layer.adapt(input_sequences)

# Vectorize input sequences
X = vectorize_layer(input_sequences).numpy()

# Vectorize target words (single words)
target_vectorizer = layers.TextVectorization(
    max_tokens=max_vocab_size,
    output_mode='int',
    output_sequence_length=1
)
target_vectorizer.adapt(target_words)
y = target_vectorizer(target_words).numpy().flatten()

# Step 4: Build the LSTM model
def build_lstm_model(vocab_size, embedding_dim, sequence_length):
    model = models.Sequential([
        # Embedding layer to convert word indices to dense vectors
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=sequence_length),
        # LSTM layer to learn sequential patterns
        layers.LSTM(64, return_sequences=False),
        # Dense layer for classification
        layers.Dense(64, activation='relu'),
        layers.Dense(vocab_size, activation='softmax')  # Output probability for each word
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Get vocabulary size
vocab_size = len(vectorize_layer.get_vocabulary())

# Build and summarize the model
model = build_lstm_model(vocab_size, embedding_dim, sequence_length)
model.summary()

# Step 5: Train the model
model.fit(X, y, epochs=50, batch_size=16, verbose=1)

# Step 6: Function to predict the next word
def predict_next_word(model, input_text, vectorize_layer, target_vectorizer):
    # Preprocess input text
    input_seq = vectorize_layer([input_text]).numpy()
    # Predict
    pred = model.predict(input_seq, verbose=0)
    # Get the predicted word index
    pred_index = np.argmax(pred[0])
    # Convert index to word
    vocab = target_vectorizer.get_vocabulary()
    return vocab[pred_index]

# Example prediction
test_input = "the quick brown"  # Example input of n_words
predicted_word = predict_next_word(model, test_input, vectorize_layer, target_vectorizer)
print(f"Input: {test_input}")
print(f"Predicted next word: {predicted_word}")







