import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from matplotlib import pyplot as plt

def main():
    # Path to all training and validation datasets
    train_path = 'D:/Users/Leong/Documents/FYP/bank_note_recognition/dataset/train'
    validate_path = 'D:/Users/Leong/Documents/FYP/bank_note_recognition/dataset/validate'

    # preprocess the training and validation data based on MobileNet model input requirement
    train_batch = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
        directory=train_path, classes={'rm1':0,
                                       'rm5':1,
                                       'rm10':2,
                                       'rm20':3,
                                       'rm50':4,
                                       'rm100':5}, target_size=(224, 224), batch_size=10)
    validate_batch = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
        directory=validate_path, classes={'rm1':0,
                                          'rm5':1,
                                          'rm10':2,
                                          'rm20':3,
                                          'rm50':4,
                                          'rm100':5}, target_size=(224, 224), batch_size=10)

    # Import the original Mobilenet model
    mobilenet = tf.keras.applications.mobilenet.MobileNet()

    mobilenet.summary()

    # Creating new Functional model (instead of Sequential)
    # Deleting last 5 layers and introducing new output layer of unit 6 (6 classes)
    x = mobilenet.layers[-6].output
    output = Dense(units=6, activation='softmax')(x)  # Add softmax layer with 6 classification outputs
    model = Model(inputs=mobilenet.input, outputs=output)

    # Freeze the first to just before the last 23rd layer of the new model (Only train the parameters at the last 23 layers)
    for layer in model.layers[:-23]:
        layer.trainable = False

    model.summary()

    # Compile the modified and tuned mobilenet model "model"
    model.compile(optimizer=Adam(lr=0.0001), loss=categorical_crossentropy, metrics=['accuracy'])

    # Train the model
    history = model.fit(x=train_batch, validation_data=validate_batch, epochs=30, verbose=2)

    # Plot accuracy changes
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # Plot loss changes
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    print('Done Training')

    # Save the trained model as h5 file
    for version in range(50):
        if os.path.isfile(f'D:/Users/Leong/Documents/FYP/bank_note_recognition/models/trained_model_v1.{version}.h5') == False:
            model.save(f'D:/Users/Leong/Documents/FYP/bank_note_recognition/models/trained_model_v1.{version}.h5')
            print(f'Version 1.{version} saved')
            break


if __name__ == '__main__':
    main()