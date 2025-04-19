import tensorflow as tf
import numpy as np

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)
"""
Design the model

In this section, you will design your model using TensorFlow.

You will use a machine learning algorithm called neural network to train your model. 
You will create the simplest possible neural network. 
It has 1 layer, and that layer has 1 neuron. The neural network's input is only one value at a time.
 Hence, the input shape must be [1].
"""
model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])

"""
Compile the model

Next, you will write the code to compile your neural network. When you do, you must specify 2 functions, a loss and an optimizer.

If you've seen lots of math for machine learning, this is where you would usually use it, but tf.keras nicely encapsulates it in functions for you.

From your previous examination, you know that the relationship between the numbers is y=3x+1.

When the computer is trying to learn this relationship, it makes a guess...maybe y=10x+10.
The loss function measures the guessed answers against the known correct answers and measures how well or how badly it did.
"""

model.compile(optimizer=tf.keras.optimizers.SGD(),loss=tf.keras.losses.MeanSquaredError())

"""
To train the neural network to 'learn' the relationship between the Xs and Ys, you will use model.fit.

This function will train the model in a loop where it will make a guess, measure how good or bad it is (aka the loss),
w use the optimizer to make another guess, etc. It will repeat this process for the number of epochs you specify, which in this lab is 50.
"""
model.fit(xs, ys, epochs=50)

print(str(model.predict(np.array([10.0]))))