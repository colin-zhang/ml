import numpy as np
import matplotlib.pyplot as plt
'''
https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html?highlight=imshow#matplotlib.pyplot.imshow
'''

t = np.random.rand(128, 128) * 255
plt.imshow(t.astype('uint8'), cmap='gray')
plt.show()

t2 = np.random.rand(128, 128, 3) * 255
plt.imshow(t2.astype('uint8'), cmap=plt.cm.hot)
plt.show()

