import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


################################################################################################
#
# Mariusz Wisniewski KD-712
#
# Computer Vision
# 4606-ES-000000C-0128
#
# Measuring the position of the scale line on the camera image
#
# Histogram generation
#
# 2023-06-07
#
################################################################################################


#data = pd.read_csv('detekcja_analitic0.txt',sep=' ',header=None)
data = pd.read_csv('detekcja_hough0.txt', sep=' ', header=None)
#data = pd.read_csv('detekcja_tf0.txt',sep=' ',header=None)

data = pd.DataFrame(data)
x = data[7]
y = []
for i in range(len(x)):
    if -10 < x[i] < 10:
        y.append(x[i])

print("total, ok, % ", len(x), len(y), len(y)/len(x))
print("median std [pix]", np.median(y), np.std(y))
scale = 0.00302323
print("median std [mm]", np.median(y)*scale, np.std(y)*scale, len(y))

plt.xlim(-10, 10)
plt.xticks(np.arange(-10, 10+1, 1.0))
plt.hist(y, bins=np.arange(-10, 10, 1.0))
plt.savefig('histogram.png')
plt.show()
