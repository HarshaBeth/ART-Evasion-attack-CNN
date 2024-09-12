# Adversarial Robustness Toolbox (ART) <br> ~~ Evasion Attacks on CNN classification model
![image](https://github.com/user-attachments/assets/73ca3d70-672b-415c-8b77-216dfc9e470e)

## Impact
Evasion attacks are made to drastically reduce the accuracy of a model by manipulating the input data. This works by adding small perturbations to the input, resulting in incorrect predictions and misclassification.
Moreover, many AI applications have potential threats, such as security systems that can incorrectly authorize if an image is predicted in the culprit's desire. It's essential to understand and replicate these attacks in order to develop secure and robust AI systems. This project aims to display the effects and the process of how an evasion attack is possible. 

<hr>

To begin, we need to have a model to attack hence we create a Convolutional Neural Network (CNN) that can predict handwritten numbers for our project. First, the dataset gets loaded, preprocessed, and split into training and testing datasets. Next, we build our CNN model and train it with 5 epochs.

```
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))
```
