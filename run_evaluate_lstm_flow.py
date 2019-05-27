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
model = 'models/deploy_lstm.prototxt'

net = caffe.Net(model, model_weights, caffe.TEST) 

evaluate.evaluate(net, h5Files)
