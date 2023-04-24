# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 21:11:55 2021

@author: Emon
"""

import matplotlib.pyplot as plt
plt.style.use('seaborn')
 
plt.figure(figsize=(10,10))
plt.subplot(4,4,1)
image_index = 2853
predict = x_test[image_index].reshape(28,28)
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
plt.title("Predicted Label: "+str(pred.argmax()))
 
plt.subplot(4,4,2)
image_index = 2000
predict = x_test[image_index].reshape(28,28)
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
plt.title("Predicted Label: "+str(pred.argmax()))
 
plt.subplot(4,4,3)
image_index = 1500
predict = x_test[image_index].reshape(28,28)
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
plt.title("Predicted Label: "+str(pred.argmax()))
 
plt.subplot(4,4,4)
image_index = 1345
predict = x_test[image_index].reshape(28,28)
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
plt.title("Predicted Label: "+str(pred.argmax()))