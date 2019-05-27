import h5py
import os
import numpy as np
import caffe
import time
import sys
import glob

import config

def evaluate(net, h5Files, sub_mean=False, num_shuffles=0):
  num_frames = 16
  stride = 8
  batch_size = 500

  #create label dict
  video_label_dict = {}
  video_labels = open('ucf101_split/ucf101_split1_testVideos.txt').readlines()
  video_labels = [(v.split(' ')[0].split('/')[1], v.split(' ')[1].strip()) for v in video_labels]

  for v, l in video_labels:
    video_label_dict[v] = l

  gt_labels = []
  predict_labels = []
 
  for ih, h5File in enumerate(h5Files):
    print 'On %d/%d.' %(ih, len(h5Files))

    f = h5py.File(h5File)
    h5_dict = {}
    total_frames = 0
    for key in f.keys():
      h5_dict[key] = np.array(f[key])
      total_frames += h5_dict[key].shape[0]
    f.close()

    data_in = np.zeros((int(total_frames*2.5),4096))
    video_boundaries = [0]
    added_frames = 0
    for video in h5_dict.keys():
      feats = h5_dict[video]
      num_video_frames = feats.shape[0] 
      for iv in range(0, num_video_frames, stride):
        if iv+num_frames > num_video_frames:
          feats_chunk = feats[-1*num_frames:, ...]
        else:
          feats_chunk = feats[iv:iv+num_frames, ...]
        data_in[added_frames:added_frames+num_frames,...] = feats_chunk
        added_frames += num_frames
      video_boundaries.append(added_frames)
      gt_labels.append(int(video_label_dict[video]))
    data_in = data_in[:video_boundaries[-1],...]

    num_clips = data_in.shape[0]/num_frames
    #run forward pass and build up predicted labels
    num_samples = data_in.shape[0]
    out_probs = np.zeros((num_shuffles+1, num_samples, 101))
    for i in range(0, num_samples, batch_size*num_frames):
      data_in_chunk = np.zeros((batch_size*num_frames, 4096))
      data_in_chunk[:min(num_samples-i, batch_size*num_frames),...] = data_in[i:min(num_samples, i+batch_size*num_frames),...]
      data_in_chunk_reshape = np.zeros((batch_size*num_frames, 4096))
      for shuffle_idx in range(num_shuffles+1):
        for nc in range(batch_size):
          if sub_mean:
            mean_data = np.mean(data_in_chunk[nc*num_frames:(nc+1)*num_frames,:], axis=0)
            for nf in range(num_frames):
              new_idx = nf*batch_size + nc
              data_in_chunk_reshape[new_idx,:] = mean_data
          else:
            rand_perm = np.random.permutation(num_frames)
            for nf in range(num_frames):
              if shuffle_idx == 0:
                old_idx = nc*num_frames+nf
              else:
                old_idx = nc*num_frames+rand_perm[nf]
              new_idx = nf*batch_size + nc
              data_in_chunk_reshape[new_idx,:] = data_in_chunk[old_idx,:]
        net.blobs['data'].data[...] = data_in_chunk_reshape.reshape((8000, 4096, 1,1))
        clip_markers_chunk = np.ones((8000, 1, 1,1))
        clip_markers_chunk[:batch_size,...] = 0
        net.blobs['clip_markers'].data[...] = clip_markers_chunk
        out = net.forward()
        out_probs_reshape = np.zeros((8000,101))
        count = 0
        for nc in range(min(num_samples-i, batch_size*num_frames)/num_frames):
          for nf in range(num_frames):
            out_probs_reshape[count,:] = out['probs'][nf, nc, :]
            count += 1
        out_probs[shuffle_idx, i:min(i+batch_size*num_frames,num_samples):] = out_probs_reshape[:min(num_samples-i, batch_size*num_frames),:]
    for video_increment in range(len(video_boundaries)-1):
      vid_start = video_boundaries[video_increment]
      vid_end = video_boundaries[video_increment+1]
      predict_labels_shuffle = np.argmax(np.mean(out_probs[:, vid_start:vid_end,...], axis=1), axis=1)
      predict_labels.append(predict_labels_shuffle)

  predict_labels_np = np.array(predict_labels)
  gt_labels_np = np.array(gt_labels)
  for shuffle_idx in range(num_shuffles+1):
    if shuffle_idx == 0:
      print('No input sequence shuffling')
    else:
      print('Input sequence shuffling trial #%d' % (shuffle_idx))
    num_correct = len(np.where(gt_labels_np-predict_labels_np[:,shuffle_idx] == 0)[0])
    print 'accuracy = %f (num_correct = %d)' %(float(num_correct)/len(gt_labels), num_correct)

def evaluate_single(net, h5Files):
  batch_size = 8000
  
  #create label dict
  video_label_dict = {}
  video_labels = open('ucf101_split/ucf101_split1_testVideos.txt').readlines()
  video_labels = [(v.split(' ')[0].split('/')[1], v.split(' ')[1].strip()) for v in video_labels]
  
  for v, l in video_labels:
    video_label_dict[v] = l
  
  gt_labels = []
  predict_labels = []
   
  for ih, h5File in enumerate(h5Files):
    print 'On %d/%d.' %(ih, len(h5Files))
  
    f = h5py.File(h5File)
    h5_dict = {}
    total_frames = 0
    for key in f.keys():
      h5_dict[key] = np.array(f[key])
      total_frames += h5_dict[key].shape[0]
    f.close()
  
    data_in = np.zeros((total_frames,4096))
    video_boundaries = [0]
    added_frames = 0
    for video in h5_dict.keys():
      feats = h5_dict[video]
      num_video_frames = feats.shape[0]
      data_in[added_frames:added_frames+num_video_frames,...] = feats
      added_frames += num_video_frames
      video_boundaries.append(added_frames)
      gt_labels.append(int(video_label_dict[video]))
    assert added_frames == total_frames
  
    #run forward pass and build up predicted labels
    out_probs = np.zeros((total_frames, 101))
    for i in range(0, total_frames, batch_size):
      data_in_chunk = np.zeros((batch_size, 4096))
      data_in_chunk[:min(total_frames-i, batch_size),...] = data_in[i:min(total_frames, i+batch_size),...]
      net.blobs['data'].data[...] = data_in_chunk
      out = net.forward()
      out_probs[i:min(i+batch_size,total_frames) :] = out['probs'][:min(total_frames-i, batch_size),:] 
    for video_increment in range(len(video_boundaries)-1):
      vid_start = video_boundaries[video_increment]
      vid_end = video_boundaries[video_increment+1]
      predict_labels.append(np.argmax((np.mean(out_probs[vid_start:vid_end,...], axis=0)))) 
  
  num_correct = len(np.where(np.array(gt_labels)-np.array(predict_labels) == 0)[0])
  print 'accuracy = %f (num_correct = %d)' %(float(num_correct)/len(gt_labels), num_correct)

def image_processor_helper(transformer, data_in, image_dim=227):
  resize = (240, 320)
  if not ((data_in.shape[0] == resize[0]) & (data_in.shape[1] == resize[1])):
    data_in = caffe.io.resize_image(data_in, resize)
  
  shift_x = (resize[0] - image_dim)/2
  shift_y = (resize[1] - image_dim)/2
  shift_data_in = data_in[shift_x:shift_x+image_dim,shift_y:shift_y+image_dim,:] 
  processed_image = transformer.preprocess('data',shift_data_in)
  return processed_image
  
def image_processor(transformer, input_im, image_dim=227):
  data_in = caffe.io.load_image(input_im)
  return image_processor_helper(transformer, data_in, image_dim)

def create_transformer(net, image_dim, flow):
  shape = (128,3,image_dim,image_dim)
  transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
  transformer.set_raw_scale('data', 255)
  image_mean = [103.939, 116.779, 128.68]
  if flow:
    image_mean = [128, 128, 128]
  channel_mean = np.zeros((3,image_dim,image_dim))
  for channel_index, mean_val in enumerate(image_mean):
    channel_mean[channel_index, ...] = mean_val
  transformer.set_mean('data', channel_mean)
  transformer.set_channel_swap('data', (2, 1, 0))
  transformer.set_transpose('data', (2, 0, 1))
  return transformer

def extract_features(model_weights, flow=False, save_folder='extracted_features', im_path='images'):

  videos = open('ucf101_split/ucf101_split1_testVideos.txt').readlines()
  videos = [im_path + '/' + v.split(' ')[0].split('/')[1]  for v in videos]
  
  image_dim = 227
  
  if not os.path.isdir(save_folder):
    print 'Creating folder %s.' %save_folder
    os.mkdir(save_folder)
  
  model_file = 'models/deploy_feature_extract.prototxt'
  
  #max number frames per h5
  max_save_frames = 50000
   
  #layer to extract
  feature_extract = 'fc6'
  feature_size = 4096
  
  net = caffe.Net(model_file, model_weights, caffe.TEST)
  transformer = create_transformer(net, image_dim, flow) 
  
  
  def save_frames(feats, num_h5):
    f = h5py.File('%s/extracted_features_%d.h5' %(save_folder, num_h5), "w")
    for key in feats.keys():
      dset = f.create_dataset(key, data=feats[key])
    f.close()
    print 'Saved to: ', '%s/extracted_features_%d.h5' %(save_folder, num_h5)
  
  num_h5 = 0
  num_frames = 0
  features = {} 
  t = time.time()
  for iv, video in enumerate(videos):
    sys.stdout.write('\rOn video %d/%d.' %(iv, len(videos)))
    sys.stdout.flush()
    if float(num_frames) / max_save_frames > 1:
      sys.stdout.write('\n')
      print 'Saving files. Time between saves: %f.\n' %(time.time()-t)
      t = time.time()
  
      save_frames(features, num_h5)
  
      features = {}
      num_h5 += 1
      num_frames = 0
    
    frames = glob.glob('%s/*jpg' %video)
      
    data = []
    #t = time.time()
    for frame in sorted(frames):
      data.append(image_processor(transformer, frame))
  
    min_batch = 400 
    video_key = video.split('/')[-1] 
  
    features[video_key] = np.zeros((len(data), feature_size)) 
    for d in range(0, len(data), min_batch):
      end_point = min(d+min_batch, len(data))
      net.blobs['data'].reshape(end_point-d,3,image_dim,image_dim)
      net.blobs['data'].data[...] = data[d:end_point]
      out = net.forward()
      features[video_key][d:end_point,...] = net.blobs[feature_extract].data
  
    num_frames += len(frames)
    
  print 'Saving files. Time between saves: %f.\n' %(time.time()-t)
  save_frames(features, num_h5)
   
  sys.stdout.write('\n')
