import os
import glob
import numpy as np
import caffe

import evaluate
import config

caffe.set_mode_gpu()
caffe.set_device(0)

model_weights = os.path.join(config.LRCN_MODELS_DIR, 'single_frame_all_layers_hyb_RGB_iter_5000.caffemodel')
h5Files = glob.glob(os.path.join(config.DATASET_DIR, 'extracted_features_baseline_RGB/*.h5'))
model = 'models/deploy_baseline.prototxt'

net = caffe.Net(model, model_weights, caffe.TEST) 

evaluate.evaluate_single(net, h5Files)
