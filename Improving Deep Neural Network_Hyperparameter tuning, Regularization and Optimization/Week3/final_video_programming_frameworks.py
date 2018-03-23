
# coding: utf-8

# In[30]:

import numpy as np
import tensorflow as tf


# In[40]:

w = tf.Variable(0, dtype = tf.float32)    #defined a variable
#cost = tf.add(tf.add(w**2, tf.multiply(-10.,w)),25)      #cost function equation
coefficients = np.array([[1.],[-20.],[100.]])
X = tf.placeholder(tf.float32,[3,1])

#cost = w**2 -10*w + 25
cost = X[0][0]*w**2 + X[1][0]*w + X[2][0]
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
print(session.run(w))


# In[41]:

session.run(train,feed_dict = {X:coefficients})   #runs one step of Gradient Descent
print(session.run(w))


# In[42]:

for i in range(1000):
    session.run(train, feed_dict = {X:coefficients})
print(session.run(w))


# In[55]:

r = np.random.rand()
beta = 1-10**(-r+1)
print(beta)


# In[ ]:



