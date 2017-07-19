import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img=mpimg.imread('MNIST_0.png')
imgplot = plt.imshow(img)
plt.show()


import time
print("something")
time.sleep(2.5)    # pause 5.5 seconds
print("something")



img2=mpimg.imread('MNIST_1.png')
imgplot2 = plt.imshow(img2)
plt.show()