import numpy as np
import matplotlib.pyplot as plt
import sys
import os
caffe_root = '/opt/caffe/'
sys.path.insert(0,caffe_root+'python')
import caffe

caffe.set_mode_cpu()
#caffe.set_device(0) #use gpu id
solver = caffe.SGDSolver('/opt/caffe/examples/mnist/lenet_solver.prototxt') #use solver.prototxt not train.prototxt

niter = 1000
test_interval = 200
train_loss = np.zeros(niter)
test_acc = np.zeros(int(np.ceil(niter/test_interval)))

for it in range(niter):
    solver.step(1)
    train_loss[it] = solver.net.blobs['loss'].data
    solver.test_nets[0].forward(start='conv1')
    if it % test_interval == 0:
        acc = solver.test_nets[0].blobs['accuracy'].data
        print('Iteration',it,'testing...','accuracy:',acc)
        test_acc[it // test_interval] = acc
print(test_acc)
_,ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(np.arange(niter),train_loss)
ax2.plot(test_interval*np.arange(len(test_acc)),test_acc,'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
plt.show()

