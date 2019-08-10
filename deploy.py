import numpy as np
import sys
caffe_root = '/opt/caffe/'
sys.path.insert(0,caffe_root+'python')
import caffe


caffe.set_mode_cpu()
model_def = '/opt/caffe/models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = '/opt/caffe/models/bvlc_reference_caffenet/deploy.caffemodel'

net = caffe.Net(model_def,model_weights,caffe.TEST)
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)
print('mean-subtracted values:',zip('BGR',mu))

transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})

transformer.set_transpose('data',(2,0,1))
transformer.set_mean('data',mu)
transformer.set_raw_scale('data',255)
transformer.set_channel_swap('data',(2,1,0))

net.blobs['data'].reshape(1,3,227,227)
image = caffe.io.load_image(caffe_root+'examples/images/cat.jpg')
transformed_image = transformer.preprocess('data',image)

net.blobs['data'].data[...] = transformed_image

output = net.forward()
output_prob = output['prob'][0]

print('predicted class is:',output_prob.argmax())


