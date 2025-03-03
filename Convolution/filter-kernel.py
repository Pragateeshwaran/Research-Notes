import tensorflow as tf

class Net(tf.keras.Model):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=12, kernel_size=(2, 3), activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=10, kernel_size=(2, 3), activation='relu')

    def call(self, x):
        x = self.conv1(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        return x

# Define input shape
input_shape = (1, 28, 28, 1)  # (batch_size, height, width, channels)

# Initialize model
model = Net()

# Build model with a dummy input
model.build(input_shape)

# Print model summary
model.summary()

# Test with a random input
sample_input = tf.random.normal(input_shape)
output = model(sample_input)

print("Output shape:", output.shape)
