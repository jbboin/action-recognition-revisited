import os
import glob
import numpy as np
import caffe

import evaluate
import config

caffe.set_mode_gpu()
caffe.set_device(0)

model_weights = os.path.join(config.LRCN_MODELS_DIR, 'flow_lstm_model_iter_50000.caffemodel')
h5Files = glob.glob(os.path.join(config.DATASET_DIR, 'extracted_features_lstm_flow/*.h5'))
model_lstm = 'models/deploy_lstm.prototxt'
model_fstm = 'models/deploy_fstm.prototxt'

net_lstm = caffe.Net(model_lstm, model_weights, caffe.TEST) 
net = caffe.Net(model_fstm, model_weights, caffe.TEST) 
d_out = 256
net.params['lstm-fc'][0].data[...] = net_lstm.params['lstm1'][0].data[3*d_out:4*d_out,:]
net.params['lstm-fc'][1].data[...] = net_lstm.params['lstm1'][1].data[3*d_out:4*d_out]

evaluate.evaluate(net, h5Files, sub_mean=True)
