from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Load your trained model
model = load_model('crop_disease_model.h5')  # Or 'crop_disease_model.keras' if you saved it in the new format

# Assuming you have a test generator already created (like in train_model.py)
# Make sure to import or recreate your test_generator here

# Evaluate the model
test_generator.reset()  # Reset the generator to start from the beginning
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Confusion Matrix
print("Confusion Matrix")
print(confusion_matrix(true_classes, predicted_classes))

# Classification Report
print("Classification Report")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Accuracy and Loss Plot (if you have the history object from model training)
# Make sure the 'history' object is accessible (i.e., save it in train_model.py and load it here)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()
