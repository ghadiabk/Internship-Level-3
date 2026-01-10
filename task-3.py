import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

print("Step 1: Loading and Preprocessing MNIST data...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    
    layers.Dense(128, activation='relu'),
    
    layers.Dropout(0.2),
    
    layers.Dense(10, activation='softmax')
])

custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=custom_optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\nStep 2: Starting Training...")
history = model.fit(x_train, y_train, 
                    epochs=10, 
                    batch_size=32, 
                    validation_split=0.2, 
                    verbose=1)

print("\nStep 3: Final Evaluation on Test Data...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nFinal Accuracy: {test_acc*100:.2f}%')

print("\nStep 4: Generating and saving performance plots...")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='#1f77b4', lw=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#ff7f0e', lw=2, linestyle='--')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', color='#d62728', lw=2)
plt.plot(history.history['val_loss'], label='Validation Loss', color='#9467bd', lw=2, linestyle='--')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('mnist_training_report.png', dpi=300)
print(f"Results saved successfully to: {os.getcwd()}/mnist_training_report.png")