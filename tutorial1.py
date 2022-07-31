#https://www.tensorflow.org/tutorials/quickstart/beginner
import tensorflow as tf
import numpy as np
print(tf.__version__)

mnist = tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train,x_test = x_train/255.0,x_test/255.0


# Building a machine Learning model

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)), #Flatten used as a dimentionality reduction of multidimentional vectors
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.2), #Nullifies certains neurons to mitigate overfitting
    tf.keras.layers.Dense(10)
])


'''
Outputs are logits scores
The logits are the unnormalized log probabilities output the model 
(the values output before the softmax normalization is applied to them)
'''
predictions = model(x_train[:1]).numpy()
print(predictions)

# using softmax to convert logits to probabilities
print(tf.nn.softmax(predictions).numpy())
# print(predictions)

'''
Note: It is possible to bake the tf.nn.softmax function 
into the activation function for the last layer of the network. 
While this can make the model output more directly interpretable, 
this approach is discouraged as it's impossible to 
provide an exact and numerically stable loss calculation for 
all models when using a softmax output.
'''

#Checking for the loss of training
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
print(loss_fn(y_train[:1],predictions).numpy())


model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)