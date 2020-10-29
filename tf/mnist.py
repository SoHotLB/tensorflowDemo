
import tensorflow as tf 
 
mnist = tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()

import matplotlib.pyplot as plt 

image_index = 1234
print(y_train[image_index])
# plt.imshow(x_train[image_index])
plt.imshow(x_train[image_index], cmap='Greys') 


# 预处理 28 * 28  ->  32 * 32 
import numpy as np
x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2)), 'constant', constant_values=0)
x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2)), 'constant', constant_values=0)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

 

x_train = x_train.reshape(x_train.shape[0], 32, 32, 1)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 1)

# 构建神经网络模型 Lenet 模型


# 模型实例化
model = tf.keras.models.Sequential([
 tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='valid', activation=tf.nn.relu, input_shape=(32,32,1)),#relu
 tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='same'),
 tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation=tf.nn.relu),
 tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='same'),
 tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(units=120, activation=tf.nn.relu),
 tf.keras.layers.Dense(units=84, activation=tf.nn.relu),
 tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
 ])

# 模型展示
model.summary()

# 模型训练

"""**第三部分：模型训练**"""

import numpy as np

 
# 超参数设置
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# 优化器
adam_optimizer = tf.keras.optimizers.Adam(learning_rate)

model.compile(optimizer=adam_optimizer,
       loss=tf.keras.losses.sparse_categorical_crossentropy,
       metrics=['accuracy'])

import datetime
start_time = datetime.datetime.now()

model.fit(x=x_train,
     y=y_train,
     batch_size=batch_size,
     epochs=num_epochs)
end_time = datetime.datetime.now()
time_cost = end_time - start_time
print ("time_cost = ", time_cost)   

 

model.save('lenet_model.h5')

# 数据预测

# 预测
image_index = 4444

# plt.show(y_test[image_index])

print (y_test[image_index])
pred = model.predict(x_test[image_index].reshape(1, 32, 32, 1))
print(pred)
print(pred.argmax())  # argmax是numpy的函数，表示返回最大数的索引