
# coding: utf-8

# In[1]:


import tensorflow as tf
import keras 

mnist=keras.datasets.mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()


# In[4]:




x_train=keras.utils.normalize(x_train,axis=1)
x_test=keras.utils.normalize(x_test,axis=1)


model=keras.models.Sequential()
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128,activation='relu'))
model.add(keras.layers.Dense(128,activation='relu'))
model.add(keras.layers.Dense(128,activation='relu'))
model.add(keras.layers.Dense(128,activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy'
             ,metrics=['accuracy'])

model.fit(x_train,y_train,epochs=5)


# In[5]:


val_loss,val_accuracy=model.evaluate(x_test,y_test)
print("Loss=",val_loss)
print("Accuracy=",val_accuracy)


# In[7]:


import matplotlib.pyplot as plt

plt.imshow(x_train[1])
plt.show()

#print(x_train[1])



# In[8]:




model.save('final_model.h5')
#del model  # deletes the existing model

# returns a compiled model
# identical to the previous one


# In[9]:


model.summary()


# In[11]:


#Loading the saved model
from keras.models import load_model

loadednew=load_model('final_model.h5')
mypred1=loadednew.predict(x_test)


# In[15]:


import matplotlib.pyplot as plt

plt.imshow(x_test[2])
print(mypred1[2])
print(mypred1[2].argmax())

acc1,lss1=loadednew.evaluate(x_test,y_test)
print("Accuracy=",lss1)

