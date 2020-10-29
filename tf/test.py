# 加载已经训练好的LeNet模型
import tensorflow as tf

model = tf.keras.models.load_model('lenet_model.h5')
 
model.summary()

import cv2
import matplotlib.pyplot as plt

# 第一步：读取图片
img = cv2.imread('8.png') 
print(img.shape)

# 第二步：将图片转为灰度图
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.shape)
plt.imshow(img,cmap='Greys')

 
# 第三步：将图片的底色和字的颜色取反
img = cv2.bitwise_not(img)
plt.imshow(img,cmap='Greys')


# 第四步：将底变成纯白色，将字变成纯黑色
img[img<=144]=0
img[img>140]=255  # 130

# 显示图片
plt.imshow(img,cmap='Greys')
 

# 第五步：将图片尺寸缩放为输入规定尺寸
img = cv2.resize(img,(32,32))
plt.show()


# 第六步：将数据类型转为float32
img = img.astype('float32')

# 第七步：数据正则化
img /= 255

# 第八步：增加维度为输入的规定格式
img = img.reshape(1, 32, 32, 1)
print(img.shape)
 

 
# 第九步：预测
pred = model.predict(img)

print(pred)
# 第十步：输出结果
print(pred.argmax())
