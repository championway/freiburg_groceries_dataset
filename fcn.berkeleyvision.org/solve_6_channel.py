import sys
sys.path.append('/home/arg_ws3/caffe/python')
sys.path.append('/home/arg_ws3/david_trainings/fcn.berkeleyvision.org')
print sys.path

import caffe
import surgery, score

import numpy as np
import os
import sys

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

weights = "/home/arg_ws3/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"

# init
#caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver_6_channel.prototxt')
#solver.net.copy_from(weights)

# surgeries
#interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
#surgery.interp(solver.net, interp_layers)

# scoring
#val = np.loadtxt('/home/peter/caffe/data/new_brand/full_mask_final/predict_mask/val.txt', dtype=str)

for i in range(25):
    solver.step(1000)
    print "===================== Round:", i+1, "====================="
    #score.seg_tests(solver, False, val, layer='score')
