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
