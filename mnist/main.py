from keras.datasets import mnist
from tensorflow.keras import models,layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

(train_imgs,train_labels),(test_imgs,test_labels) = mnist.load_data()

#plot
plt.imshow(train_imgs[4],cmap=plt.cm.binary)
plt.show()

#print(train_imgs)

nnet = models.Sequential()

# adding layers
nnet.add(
    layers.Dense(512,activation='relu',input_shape=(28*28,))
)
# Dense first param is the outputs the layer
nnet.add(layers.Dense(10,activation='softmax'))

#
train_imgs = train_imgs.reshape((60000,28*28))
train_imgs= train_imgs.astype('float32')/255

test_imgs= test_imgs.reshape((10000,28*28))
test_imgs= test_imgs.astype('float32')/255

train_labels = to_categorical(train_labels)
test_labels =to_categorical(test_labels)



#compiling /
nnet.compile(optimizer ='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# train the net
nnet.fit(
    train_imgs, train_labels,
    epochs=10,
    batch_size=128
)

