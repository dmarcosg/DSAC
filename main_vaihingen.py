import tensorflow as tf
import scipy.misc
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from active_contours_fast import draw_poly,derivatives_poly,draw_poly_fill
from snake_utils import imrotate, plot_snakes, CNN_B, snake_graph, plot_for_figure
from scipy import interpolate
from skimage.filters import gaussian
import scipy
import time
import math
from PIL import Image, ImageOps
from tensorflow.python.client import timeline

import matplotlib.pyplot as plt

model_path = 'models/vaihingen/'
do_plot = False
do_train = True
start_test = 100




def snake_process (mapE, mapA, mapB, mapK, init_snake):

    for i in range(mapE.shape[3]):
        Du = np.gradient(mapE[:,:,0,i], axis=0)
        Dv = np.gradient(mapE[:,:,0,i], axis=1)
        u = init_snake[:,0:1]
        v = init_snake[:,1:2]
        du = np.zeros(u.shape)
        dv = np.zeros(v.shape)
        snake_hist = []
        snake_hist.append(np.array([u[:, 0], v[:, 0]]).T)
        tic = time.time()
        for j in range(1):
            u, v, du, dv = sess2.run([tf_u, tf_v, tf_du, tf_dv], feed_dict={tf_Du: Du, tf_Dv: Dv,
                                                                               tf_u0: u, tf_v0: v, tf_du0: du, tf_dv0: dv,
                                                                               tf_alpha: mapA[:,:,0,i], tf_beta: mapB[:,:,0,i],
                                                                               tf_kappa: mapK[:,:,0,i]}) #,options=run_options, run_metadata=run_metadata
            snake_hist.append(np.array([u[:, 0], v[:, 0]]).T)

        #print('%.2f' % (time.time() - tic) + ' s snake')

    return np.array([u[:,0],v[:,0]]).T,snake_hist




#Load data
L = 60
batch_size = 1
numfilt = [32,64,128,128,256,256]
im_size = 512
out_size = 256
data_path = '/mnt/bighd/Data/Vaihingen/buildings/'
csvfile=open(data_path+'polygons.csv', newline='')
reader = csv.reader(csvfile)
images = np.zeros([im_size,im_size,3,168])
masks = np.zeros([out_size,out_size,1,168])
GT = np.zeros([L,2,168])
for i in range(168):
    corners = reader.__next__()
    num_points = np.int32(corners[0])
    poly = np.zeros([num_points, 2])
    for c in range(num_points):
        poly[c, 0] = np.float(corners[1+2*c])*out_size/im_size
        poly[c, 1] = np.float(corners[2+2*c])*out_size/im_size
    [tck, u] = interpolate.splprep([poly[:, 0], poly[:, 1]], s=2, k=1, per=1)
    [GT[:,0,i], GT[:,1,i]] = interpolate.splev(np.linspace(0, 1, L), tck)
    this_im  = scipy.misc.imread(data_path+'building_'+str(i+1).zfill(3)+'.tif')
    images[:,:,:,i] = np.float32(this_im)/255
    img_mask = scipy.misc.imread(data_path+'building_mask_' + str(i+1).zfill(3) + '.tif')/255
    masks[:,:,0,i] = scipy.misc.imresize(img_mask,[out_size,out_size])/255
GT = np.minimum(GT,out_size-1)
GT = np.maximum(GT,0)

with tf.device('/gpu:0'):
    tvars, grads, predE, predA, predB, predK, l2loss, grad_predE, \
    grad_predA, grad_predB, grad_predK, grad_l2loss, x, y_ = CNN_B(im_size, out_size, L, batch_size=1,wd=0.01,layers=len(numfilt),numfilt=numfilt)


#Prepare folder to save network
if not os.path.isdir(model_path):
    os.makedirs(model_path)

if not do_train and not os.path.isdir(model_path+'results'):
    os.makedirs(model_path+'results')
elif os.path.isdir(model_path+'results/polygons.csv'):
    os.remove(model_path+'results/polygons.csv')


# Add ops to save and restore all the variables.
saver = tf.train.Saver()

#Initialize CNN
optimizer = tf.train.AdamOptimizer(1e-4, epsilon=1e-6)
apply_gradients = optimizer.apply_gradients(zip(grads, tvars))

with tf.device('/cpu:0'):
    tf_u, tf_v, tf_du, tf_dv, tf_Du, tf_Dv, tf_u0, tf_v0, tf_du0, tf_dv0, \
    tf_alpha, tf_beta, tf_kappa = snake_graph(out_size, L)



def epoch(n,i,mode):
    # mode (str): train or test
    batch_ind = np.arange(i,i+batch_size)
    batch = np.copy(images[:, :, :, batch_ind])
    batch_mask = np.copy(masks[:, :, :, batch_ind])
    thisGT = np.copy(GT[:, :, batch_ind[0]])
    if mode is 'train':
        ang = np.random.rand() * 360
        for j in range(len(batch_ind)):
            for b in range(batch.shape[2]):
                batch[:, :, b, j] = imrotate(batch[:, :, b, j], ang)
            batch_mask[:, :, 0, j] = imrotate(batch_mask[:, :, 0, j], ang, resample='nearest')
        R = [[np.cos(ang * np.pi / 180), np.sin(ang * np.pi / 180)],
             [-np.sin(ang * np.pi / 180), np.cos(ang * np.pi / 180)]]
        thisGT -= out_size / 2
        thisGT = np.matmul(thisGT, R)
        thisGT += out_size / 2
    # prediction_np = sess.run(prediction,feed_dict={x:batch})
    tic = time.time()
    [mapE, mapA, mapB, mapK, l2] = sess.run([predE, predA, predB, predK, l2loss], feed_dict={x: batch})
    mapA = np.maximum(mapA, 0)
    mapB = np.maximum(mapB,0)
    mapK = np.maximum(mapK, 0)
    #print('%.2f' % (time.time() - tic) + ' s tf inference')
    if mode is 'train':
        for j in range(mapK.shape[3]):
            mapK[:, :, 0, j] -= batch_mask[:, :, 0, j] * 0.5 - 0.5 / 2
        # mapE_aug[:,:,0,j] = mapE[:,:,0,j]+np.maximum(0,20-batch_dists[:,:,0,j])*max_val/50
    # Do snake inference
    s = np.linspace(0, 2 * np.pi, L)
    init_u = out_size / 2 + 20 * np.cos(s)
    init_v = out_size / 2 + 20 * np.sin(s)
    init_u = init_u.reshape([L, 1])
    init_v = init_v.reshape([L, 1])
    init_snake = np.array([init_u[:, 0], init_v[:, 0]]).T
    for j in range(batch_size):
        snake, snake_hist = snake_process(mapE, mapA, mapB, mapK, init_snake)
        # Get last layer gradients
        M = mapE.shape[0]
        N = mapE.shape[1]
        der1, der2 = derivatives_poly(snake)


        der1_GT, der2_GT = derivatives_poly(thisGT)

        grads_arrayE = mapE * 0.01
        grads_arrayA = mapA * 0.01
        grads_arrayB = mapB * 0.01
        grads_arrayK = mapK * 0.01
        grads_arrayE[:, :, 0, 0] -= draw_poly(snake, 1, [M, N],12) - draw_poly(thisGT, 1, [M, N],12)
        grads_arrayA[:, :, 0, 0] -= (np.mean(der1) - np.mean(der1_GT))
        grads_arrayB[:, :, 0, 0] -= (draw_poly(snake, der2, [M, N],12) - draw_poly(thisGT, der2_GT, [M, N],12))
        mask_gt = draw_poly_fill(thisGT, [M, N])
        mask_snake = draw_poly_fill(snake, [M, N])
        grads_arrayK[:, :, 0, 0] -= mask_gt - mask_snake

        intersection = (mask_gt+mask_snake) == 2
        union = (mask_gt + mask_snake) >= 1
        iou = np.sum(intersection) / np.sum(union)
    if mode is 'train':
        tic = time.time()
        apply_gradients.run(
            feed_dict={x: batch, grad_predE: grads_arrayE, grad_predA: grads_arrayA, grad_predB: grads_arrayB,
                       grad_predK: grads_arrayK, grad_l2loss: 1})
        #print('%.2f' % (time.time() - tic) + ' s apply gradients')
        #print('IoU = %.2f' % (iou))
    #if mode is 'test':
        #print('IoU = %.2f' % (iou))
    if do_plot:
        plot_snakes(snake, snake_hist, thisGT, mapE, mapA, mapB, mapK, \
                grads_arrayE, grads_arrayA, grads_arrayB, grads_arrayK, batch, batch_mask)
        #plt.show()
    return iou,snake



with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
    sess2 = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
    save_path = tf.train.latest_checkpoint(model_path)
    init = tf.global_variables_initializer()
    sess.run(init)
    start_epoch = 0
    if save_path is not None:
        saver.restore(sess,save_path)
        start_epoch = int(save_path.split('-')[-1].split('.')[0])+1

    if do_train:
        end_epoch = 100
    else:
        end_epoch = start_epoch + 1
        polygons_csvfile = open(model_path + 'results/' 'polygons.csv', 'a', newline='')
        polygons_writer = csv.writer(polygons_csvfile)

    for n in range(start_epoch,end_epoch):
        iou_test = 0
        iou_train = 0
        iter_count = 0
        if do_train:
            for i in range(0,100,batch_size):
                #print(i)
                #Do CNN inference
                new_iou_train,snake = epoch(n,i,'train')
                iou_train += new_iou_train
                iter_count += 1
                print('Train. Epoch ' + str(n) + '. Iter ' + str(iter_count) + '/' + str(100) + ', IoU = %.2f' % (
                iou_train / iter_count))
            iou_train /= 100

            saver.save(sess,model_path+'model', global_step=n)
        iter_count = 0
        for i in range(start_test,168):
            new_iou_test, snake = epoch(n, i, 'test')
            if not do_train:
                list_to_write = [len(snake)]
                snake = np.reshape(snake,2*len(snake)).tolist()
                for el in snake:
                    list_to_write.append(el)
                polygons_writer.writerow(list_to_write)
            iou_test += new_iou_test
            iter_count += 1
            print('Test. Epoch ' + str(n) + '. Iter ' + str(iter_count) + '/' + str(68) + ', IoU = %.2f' % (
            iou_test / iter_count))
        iou_test /= iter_count
        if not do_train:
            iou_csvfile = open(model_path + 'iuo_train_test.csv', 'a', newline='')
            iou_writer = csv.writer(iou_csvfile)
            iou_writer.writerow([n,iou_train,iou_test])
            iou_csvfile.close()
            polygons_csvfile.close()








#if os.path.isfile(model_path+'iuo_train_test.csv'):

#else:






