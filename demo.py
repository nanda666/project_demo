import scipy.io as scipi
import cv2
import numpy as np 
from glob import glob 
import argparse
from helpers import *
from matplotlib import pyplot as plt 
import _pickle

# feature = scipi.loadmat('attfeat.mat')
# #tran = scipi.loadmat('attrann.mat')
# bov_helper = BOVHelpers(100)
# print(feature.get("features").shape)
# #print(tran)
# descriptor_list = feature.get("features")
# resize_list = descriptor_list[:150,:512]
# print(resize_list.shape)
# bov_descriptor_stack = bov_helper.formatND(resize_list)
# bov_helper.cluster()
# bov_helper.developVocabulary(n_images =150, descriptor_list=resize_list)
# # show vocabulary trained
# bov_helper.plotHist()
im=cv2.imread("/home/ubuntu/Documents/Program/Bag-of-Visual-Words-Python-master/images/Default/106400.jpg")
print(im.shape)
gr=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
print(gr.shape)
sift = cv2.xfeatures2d.SIFT_create()
(locs, int_descriptors) = sift.detectAndCompute(gr, None)
print(int_descriptors.shape)
print(len(locs))

# np.save('/home/ubuntu/Documents/Program/Bag-of-Visual-Words-Python-master/images/Default/106400.sift', np.hstack(int_descriptors))


# nfeatures = int_descriptors.shape[1]
# padding = np.zeros((2, nfeatures))
# locs = np.vstack((locs, padding))
# header = ' '.join([str(nfeatures), str(128)])
# temp = int_descriptors.astype('float')  # convert descriptors to float
# descriptors = temp[:]
# with open(resultname, 'wb') as f:
#     pickle.dump([locs.T, descriptors.T], f, protocol=_pickle.HIGHEST_PROTOCOL)
# print ("features saved in", resultname)

#int_descriptors.tofile("/home/ubuntu/Documents/Program/Bag-of-Visual-Words-Python-master/images/Default/106400.sift","")
# np.savetxt("/home/ubuntu/Documents/Program/Bag-of-Visual-Words-Python-master/images/Default/106400.sift",int_descriptors,delimiter='\t')
# f = open("/home/ubuntu/Documents/Program/Bag-of-Visual-Words-Python-master/images/Default/106400.sift", "w")
# f.write(_pickle.dumps(int_descriptors[0] ))
# f.close()

f_feat_out = open('/home/ubuntu/Documents/Program/Bag-of-Visual-Words-Python-master/images/Default/106400.sift', 'wb')
for i in range(len(locs)):
    feat_str = ''
    feat_tmp = int_descriptors[i]
    for j in range(len(feat_tmp)):
        feat_str += str(j) + ':' + str(feat_tmp[j]) + ' '
    feat_str.strip(' ')
    feat_str += '\t' + str(label[i])
    f_feat_out.write(_pickle.dumps(feat_str + '\n')
f_feat_out.close()



https://github.com/wangg12/cup_proj/blob/master/sift_bow/findFeatures.py
https://stackoverflow.com/questions/25680529/store-the-extracted-surf-descriptors-and-keypoints-in-npy-file
https://singhgaganpreet.wordpress.com/category/opencv/reading-n-writing-sift-descriptors/
https://www.codeproject.com/Articles/619039/Bag-of-Features-Descriptor-on-SIFT-Features-with-O
https://stackoverflow.com/questions/3685265/how-to-write-a-multidimensional-array-to-a-text-file/18145279
https://github.com/ansible/ansible/issues/9413
http://ai.stanford.edu/~olga/papers/eccv10workshop-Attributes.pdf
