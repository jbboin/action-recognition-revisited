import os
import glob
import numpy as np
import caffe

import evaluate
import config

caffe.set_mode_gpu()
caffe.set_device(0)

model_weights = os.path.join(config.LRCN_MODELS_DIR, 'RGB_lstm_model_iter_30000.caffemodel')
h5Files = glob.glob(os.path.join(config.DATASET_DIR, 'extracted_features_lstm_RGB/*.h5'))
model = 'models/deploy_lstm.prototxt'

net = caffe.Net(model, model_weights, caffe.TEST) 

np.random.seed(0)
evaluate.evaluate(net, h5Files, num_shuffles=20)
