from matplotlib import pyplot as plt
import numpy as np
import csv

data1 = []
data2 = []
data3 = []
data4 = []

filename = '../scores/fast_plot/fast_model_plot_1.csv'
with open(filename, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
       data1.extend(row)


filename = '../scores/fast_plot/fast_model_plot_2.csv'
with open(filename, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
       data2.extend(row)

filename = '../scores/fast_plot/fast_model_plot_3.csv'
with open(filename, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
       data3.extend(row)

filename = '../scores/fast_plot/fast_model_plot_4.csv'
with open(filename, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
       data4.extend(row)

length = 1250


mean = []
for i in range(length):
    mean.append(np.mean([int(data1[i]),int(data2[i]),int(data3[i]),int(data4[i])]))

mean_value = np.mean(mean[750:length])

std = []
for i in range(length):
    std.append( np.std([[int(data1[i]),int(data2[i]),int(data3[i]),int(data4[i])]]) )


x = np.linspace(0, length, length)

mean = np.array(mean)
std = np.array(std)

#mean_value = [mean_value]*length

fig, ax = plt.subplots()
ax.plot(x, mean, color="red",  linewidth=0.25)
ax.fill_between(x, mean-std, mean+std,)
ax.set_ylim([0,500])
ax.hlines(y=mean_value, xmin=750, xmax=length, color='black', label="mean of last 500")
ax.legend(['mean', "standard deviation", "mean of last 500"])
#plt.grid()
plt.xlabel('Episodes', fontsize=12)
plt.ylabel('Performance', fontsize=14)

plt.savefig('fast_mean_std.png')
