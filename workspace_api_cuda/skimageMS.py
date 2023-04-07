import numpy as np
import matplotlib.pyplot as plt
import time

#from skimage import measure
from _find_contours import find_contours

# Construct some test data
#x, y = np.ogrid[-np.pi:np.pi:95j, -np.pi:np.pi:511j]
#r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))
t = np.load('./heatmaps_00000001_00000001.npy');
#r = t[:,:,8];

print(t.dtype," ",t.shape," ",t[:,:,0].size)
print("MIN t: ", t[:,:,10].min())
print("MAX t: ", t[:,:,10].max())
print("AVG t: ", t[:,:,10].mean())

r=t[:,:,10]

# get the start time
st = time.time()


# TEST confronto risultato con libreria
contours = find_contours(r, 0.5) 
#contours = measure.find_contours(r, 0.5)  # skimage

# Find contours at a constant value of 0.8
#contours = measure.find_contours(r, 0.8)
#for c in range(11): # range(11) 
    #r=t[:,:,c]
    #print("Lancio ",c)
    #contours = find_contours(r, 0.5) 
    #contours = measure.find_contours(r, 0.5)  # skimage

# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')


# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(r, cmap=plt.cm.gray)

#print(contours)
for contour in contours:
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()
