import sys
sys.path.append('/home/arg_ws3/caffe/python')
from surgery import transplant
#caffe_root = '/home/arg_ws3/caffe'  # this file should be run from {caffe_root}/examples (otherwise change this line)
#sys.path.insert(0, caffe_root + 'python')


import caffe
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.
import os
caffe.set_mode_gpu()

old_model_def = '/home/arg_ws3/caffe/models/bvlc_reference_caffenet/deploy.prototxt'
old_model_weights = '/home/arg_ws3/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

old_net = caffe.Net(old_model_def,      # defines the structure of the model
                old_model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)


new_model_def = '/home/arg_ws3/david_trainings/fcn.berkeleyvision.org/deploy.prototxt'


new_net = caffe.Net(new_model_def,      # defines the structure of the model
                caffe.TEST)     # use test mode (e.g., don't perform dropout)


transplant(new_net, old_net, suffix='')

new_net.save('initmodel.caffemodel')
