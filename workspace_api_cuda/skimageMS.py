import numpy as np
import matplotlib.pyplot as plt
import time

#from skimage import measure
from _find_contours import find_contours

# Construct some test data
x, y = np.ogrid[-np.pi:np.pi:95j, -np.pi:np.pi:511j]
#r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))
t = np.load('./heatmaps_00000022_00000001.npy');
#r = t[:,:,8];

print(t.dtype," ",t.shape," ",t[:,:,0].size)

# get the start time
st = time.time()

# Find contours at a constant value of 0.8
#contours = measure.find_contours(r, 0.8)
for c in range(1): # range(11) 
    r=t[:,:,1]
    #r=t[50:65,115:130,9]
    print("Lancio ",c)
    contours = find_contours(r, 3.38694965e-03) # circa 0.00674

# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')


# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(r, cmap=plt.cm.gray)

print(contours)

for contour in contours:
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()
