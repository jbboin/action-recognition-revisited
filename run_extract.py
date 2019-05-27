import argparse
import evaluate
import glob
import h5py
import os
import caffe

caffe.set_mode_gpu()
caffe.set_device(0)

parser = argparse.ArgumentParser()
#Model to evaluate
parser.add_argument("--model", type=str, default=None)
#folder in which to save extracted features; features will be deleted at the end unless --save_features flag included at run time
parser.add_argument("--save_folder", type=str, default='extracted_features')
#path for image frames (flow or RGB)
parser.add_argument("--im_path", type=str, default='frames')

#add flow tag if evaluating flow images
parser.add_argument("--flow", dest='flow', action='store_true')
parser.set_defaults(flow=False)

args = parser.parse_args()

if not args.model:
  raise Exception("Must input trained model for evaluation") 
if not os.path.isdir(args.save_folder):
  print 'Creating save folder %s.' %args.save_folder
  os.mkdir(args.save_folder)

#extract features
evaluate.extract_features(args.model, flow=args.flow, save_folder=args.save_folder, im_path=args.im_path)
