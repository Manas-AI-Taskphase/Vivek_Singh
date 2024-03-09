# %%
import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.keras import utils, layers, models


# %% [markdown]
# # Loading datasets

# %%
# Load images from the directories
train_dataset = utils.image_dataset_from_directory("CNN/train",
                                                                      seed=123,
                                                                      image_size=(256,256),
                                                                      batch_size=32)

test_dataset = utils.image_dataset_from_directory("CNN/test",
                                                                      seed=123,
                                                                      image_size=(256,256),
                                                                      batch_size=32)

# Labels
training_labels = train_dataset.class_names
training_labels = train_dataset.class_names

"""
# Performance tuning
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)
"""

# %% [markdown]
# # Architecture of Model

# %%
# Defining
model = models.Sequential([
    # Convolution Layers
    layers.Conv2D(filters =16, kernel_size=(3, 3), activation='relu', padding="same", input_shape=(256, 256, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(filters =32, kernel_size=(3, 3), activation='relu', padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(filters =64, kernel_size=(3, 3), activation='relu', padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(filters =128, kernel_size=(3, 3), activation='relu', padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(filters =256, kernel_size=(3, 3), activation='relu', padding="same"),
    layers.MaxPooling2D((2, 2)),
    
    # Nural Network Layer
    layers.Flatten(),
    layers.Dense(100, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(12, activation='softmax')
])

# Compile 
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# %% [markdown]
# # Training
# 

# %%
model.fit(train_dataset, epochs=3, batch_size=32)

# %% [markdown]
# # Test

# %%
loss, accuracy = model.evaluate(
    test_dataset
)

# %%
print(f"Loss:{loss}\nAccuracy:{accuracy*100}%")

# %%
from sklearn.metrics import confusion_matrix, classification_report

y_true = []
y_pred = []

for images, labels in test_dataset:
    predictions = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))



# %%
conf_matrix = confusion_matrix(y_true, y_pred)
class_report = classification_report(y_true, y_pred, target_names=training_labels)

plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(training_labels))
plt.xticks(tick_marks, training_labels, rotation=45)
plt.yticks(tick_marks, training_labels)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print("\nClassification Report:")
print(class_report)


