# Adversarial Robustness Toolbox (ART) <br> ~~ Evasion Attacks and Defense on CNN classification model
![image](https://github.com/user-attachments/assets/73ca3d70-672b-415c-8b77-216dfc9e470e)

## Impact
Evasion attacks are made to drastically reduce the accuracy of a model by manipulating the input data. This works by adding small perturbations to the input, resulting in incorrect predictions and misclassification.
Moreover, many AI applications have potential threats, such as security systems that can incorrectly authorize if an image is predicted in the culprit's desire. It's essential to understand and replicate these attacks in order to develop secure and robust AI systems. This project aims to display the effects and the process of how an evasion attack is possible. 

<hr>

To begin, we need to have a model to attack hence we create a Convolutional Neural Network (CNN) that can predict handwritten numbers for our project. First, the dataset gets loaded, preprocessed, and split into training and testing datasets. Next, we build our CNN model and train it with 5 epochs.
### ~~ Model Creation ~~

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
. <br>
. <br>
. <br>
```
Epoch 5/5 
469/469 ━━━━━━━━━━━━━━━━━━━━ 15s 33ms/step - accuracy: 0.9923 - loss: 0.0244 - val_accuracy: 0.9903 - val_loss: 0.0301
```

We achieve an amazing accuracy of 99.23% at the end of our training process, to test our highly accurate model let's take a sample image from our testing dataset.

![image](https://github.com/user-attachments/assets/fb937b03-561d-43f9-8501-531a405a6dc7)

Our model is working as expected!


## Implementing Evasion Attack (FGSM)
Firstly, we need to wrap our model with a classifier to not affect our original model and allow us to test adversarial attacks.
```
classifier = TensorFlowV2Classifier(
    model=model,
    nb_classes=10,
    input_shape=(28, 28, 1),
    loss_object=tf.keras.losses.CategoricalCrossentropy(),
    clip_values=(min_pixel_value, max_pixel_value),
)
```

```
# Create Evasion(FGSM) attack
epsilon = 0.1
attack = FastGradientMethod(estimator=classifier, eps=epsilon)
```

With this completed, we can now start our attacks. We will assess 3 attacks with varying perturbations to showcase how increasing the noise added to our input data (epsilon, ε) fools our model to predict incorrectly. The findings are summarized as follows:

- <i>**Weak attack intensity (ε = 0.1):**</i> <br>
    With a small amount of noise in our input data, our CNN model is able to predict the image as expected. However, the model's accuracy has decreased gradually from 99.23% to 88.19%.
  
- <i>**Medium attack intensity (ε = 0.2):**</i> <br>
    With a moderate amount of noise introduced, our CNN model struggles but manages to predict the image correctly. This time, the model's accuracy has reduced immensely to 45.40%.
  
- <i>**Strong attack intensity (ε = 0.3):**</i> <br>
    In this attack our image is distorted highly with our attack, with an accuracy of a mere 11.17% the model fails to predict the image correctly.
  
![image](https://github.com/user-attachments/assets/6afd00d0-c933-49fa-ac34-a9e8997e4680)

 Our CNN model has shown us how its vulnerabilities can be exploited!

 ## Findings
 Comparison between these images helps to understand how an attack causes disturbance in the image to mislead the model.
 
 ![image](https://github.com/user-attachments/assets/8fa6cbbc-f34a-49ba-9b1e-601c7afa8c1d)

 With the help of this graph, we can visualize how easy and quick it is to attack Convolutional Neural Network (CNN) models in general. The graph depicts the correlation between our model's accuracy and the perturbation in our input data.
 
 ![image](https://github.com/user-attachments/assets/1d80367d-105c-4619-a8a9-eabfe4d235e2)

<br>
<hr>
<br>

# Defense ~ Model Training with Adversarial Data
After creating our model, it is crucial to train its defense against such evasion attacks. Utilizing the ```AdversarialTrainer``` function from the ART library, we can train our model on the previous attacks.  

```
trainer = AdversarialTrainer(classifier, attacks=[attack, attack2, attack3], ratio=0.5)
trainer.fit(x_train, y_train, nb_epochs=5, batch_size=128)
```
As seen in the code above, all 3 attack strengths are used to train our model to be robust. The ratio used is 0.5 to ensure both the original and adversarial data are being used equally for training.

## Model Robustness Evaluation
Moving forward, after successfully completing the training, the newly guarded model must be evaluated and the results obtained are as follows:
- Accuracy on original test data: 99.37%
- Accuracy on adversarial test data with epsilon 0.1: 97.86%
- Accuracy on adversarial test data with epsilon 0.2: 94.42%
- Accuracy on adversarial test data with epsilon 0.3: 89.70%

Our model has been trained well and gives us great accuracy albeit attempted attacks! The difference before and after adversarial training can be seen in the plotting graph below:

![image](https://github.com/user-attachments/assets/050cfd66-d344-4414-aa9c-3124c7f7cf30)

