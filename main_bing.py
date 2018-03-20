import tensorflow as tf
import scipy.misc
import numpy as np
import csv
import os
from active_contour_maps_GD_fast import draw_poly,derivatives_poly,draw_poly_fill
from snake_utils import imrotate, plot_snakes, CNN_B, snake_graph
from scipy import interpolate
from skimage.filters import gaussian
import scipy
import time
import matplotlib.pyplot as plt

model_path = 'models/bing/'
do_plot = False
do_train = True


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
L = 30
if do_train:
    num_ims = 335
else:
    num_ims = 271
numfilt = [32,64,128,128]
batch_size = 1
im_size = 80
out_size = 80
if do_train:
    data_path = '/mnt/bighd/Data/BingJohn/buildings_osm/single_buildings/train/'
else:
    data_path = '/mnt/bighd/Data/BingJohn/buildings_osm/single_buildings/test/'

csvfile=open(data_path+'building_coords.csv', newline='')
reader = csv.reader(csvfile)
images = np.zeros([im_size,im_size,3,num_ims])
dists = np.zeros([im_size,im_size,1,num_ims])
masks = np.zeros([im_size,im_size,1,num_ims])
GT = np.zeros([L,2,num_ims])
for i in range(num_ims):
    poly = np.zeros([5, 2])
    corners = reader.__next__()
    for c in range(4):
        poly[c, 0] = np.float(corners[1+2*c])
        poly[c, 1] = np.float(corners[2+2*c])
    poly[4,:] = poly[0,:]
    [tck, u] = interpolate.splprep([poly[:, 0], poly[:, 1]], s=2, k=1, per=1)
    [GT[:,0,i], GT[:,1,i]] = interpolate.splev(np.linspace(0, 1, L), tck)
    this_im  = scipy.misc.imread(data_path+'building_'+str(i).zfill(3)+'.png')
    images[:,:,:,i] = np.float32(this_im)/255
    img_mask = scipy.misc.imread(data_path+'building_mask_' + str(i).zfill(3) + '.png')/65535
    masks[:,:,0,i] = img_mask
    img_dist = scipy.ndimage.morphology.distance_transform_edt(img_mask) + \
               scipy.ndimage.morphology.distance_transform_edt(1 - img_mask)
    img_dist = gaussian(img_dist, 10)
    dists[:,:,0,i] =  img_dist
GT = np.minimum(GT,im_size-1)
GT = np.maximum(GT,0)


###########################################################################################
# DEFINE CNN ARCHITECTURE
###########################################################################################
print('Creating CNN...',flush=True)
with tf.device('/gpu:0'):
    tvars, grads, predE, predA, predB, predK, l2loss, grad_predE, \
    grad_predA, grad_predB, grad_predK, grad_l2loss, x, y_ = CNN_B(im_size, out_size, L,
    batch_size=1,wd=0.01,layers=len(numfilt),numfilt=numfilt,E_blur=2,stack_from=0)

#Initialize CNN
optimizer = tf.train.AdamOptimizer(1e-5, epsilon=1e-7)
apply_gradients = optimizer.apply_gradients(zip(grads, tvars))

###########################################################################################
# DEFINE SNAKE INFERENCE
###########################################################################################
niter = 50
print('Creating snake inference graph...',flush=True)
with tf.device('/cpu:0'):
    tf_u, tf_v, tf_du, tf_dv, tf_Du, tf_Dv, tf_u0, tf_v0, tf_du0, tf_dv0, \
    tf_alpha, tf_beta, tf_kappa = snake_graph(out_size, L,niter=niter)

###########################################################################################
#Prepare folder to save network
###########################################################################################
start_epoch = 0
if not os.path.isdir(model_path):
    os.makedirs(model_path)

if not do_train and not os.path.isdir(model_path+'results'):
    os.makedirs(model_path+'results')
elif os.path.isdir(model_path+'results/polygons.csv'):
    os.remove(model_path+'results/polygons.csv')

# Add ops to save and restore all the variables.
saver = tf.train.Saver()


###########################################################################################
# DEFINE EPOCH
###########################################################################################
def epoch(n,i,mode):
    # mode (str): train or test
    batch_ind = np.arange(i,i+batch_size)
    batch = np.float32(np.copy(images[:, :, :, batch_ind]))/255
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
        thisGT = np.minimum(thisGT, out_size - 1)
        thisGT = np.maximum(thisGT, 0)
    # prediction_np = sess.run(prediction,feed_dict={x:batch})
    [mapE, mapA, mapB, mapK, l2] = sess.run([predE, predA, predB, predK, l2loss], feed_dict={x: batch})
    mapA = np.maximum(mapA, 0)
    mapB = np.maximum(mapB, 0)
    mapK = np.maximum(mapK, 0)
    #print('%.2f' % (time.time() - tic) + ' s tf inference')
    if mode is 'train':
        for j in range(mapK.shape[3]):
            mapK[:, :, 0, j] -= batch_mask[:, :, 0, j] * 0.5 - 0.5 / 2
    # Do snake inference
    s = np.linspace(0, 2 * np.pi, L)
    init_u = out_size / 2 + 5 * np.cos(s)
    init_v = out_size / 2 + 5 * np.sin(s)
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
        grads_arrayE[:, :, 0, 0] -= draw_poly(snake, 1, [M, N],4) - draw_poly(thisGT, 1, [M, N],4)
        grads_arrayA[:, :, 0, 0] -= (np.mean(der1) - np.mean(der1_GT))
        grads_arrayB[:, :, 0, 0] -= (draw_poly(snake, der2, [M, N],4) - draw_poly(thisGT, der2_GT, [M, N],4))
        mask_gt = draw_poly_fill(thisGT, [M, N])
        mask_snake = draw_poly_fill(snake, [M, N])
        grads_arrayK[:, :, 0, 0] -= mask_gt - mask_snake

        intersection = (mask_gt+mask_snake) == 2
        union = (mask_gt + mask_snake) >= 1
        iou = np.sum(intersection) / np.sum(union)
        area_gt = np.sum(mask_gt>0)
        area_snake = np.sum(mask_snake > 0)
    if mode is 'train':
        tic = time.time()
        apply_gradients.run(
            feed_dict={x: batch, grad_predE: grads_arrayE, grad_predA: grads_arrayA, grad_predB: grads_arrayB,
                       grad_predK: grads_arrayK, grad_l2loss: 1})
        #print('%.2f' % (time.time() - tic) + ' s apply gradients')
        #print('IoU = %.2f' % (iou))
    #if mode is 'test':
        #print('IoU = %.2f' % (iou))
    if do_plot and n >=50  and mode is 'test':
        plot_snakes(snake, snake_hist, thisGT, mapE, np.maximum(mapA, 0), np.maximum(mapB, 0), mapK, \
                grads_arrayE, grads_arrayA, grads_arrayB, grads_arrayK, batch, batch_mask)
        #plt.show()
    return iou,area_gt,area_snake,snake


###########################################################################################
# RUN THE TRAINING
###########################################################################################
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
    sess2 = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
    save_path = tf.train.latest_checkpoint(model_path)
    init = tf.global_variables_initializer()
    sess.run(init)
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
            for i in range(0,num_ims,batch_size):
                #print(i)
                #Do CNN inference
                new_iou, new_area_gt, new_area_snake, snake = epoch(n,i,'train')
                iou_train += new_iou
                iter_count += 1
                print('Train. Epoch ' + str(n) + '. Iter ' + str(iter_count) + '/' + str(num_ims) + ', IoU = %.2f' % (
                iou_train / iter_count))
            iou_train /= num_ims

            saver.save(sess,model_path+'model', global_step=n)
        iter_count = 0
        areas_gt = []
        areas_snake = []
        for i in range(num_ims):
            new_iou, new_area_gt, new_area_snake, snake = epoch(n,i, 'test')
            if not do_train:
                list_to_write = [len(snake)]
                snake = np.reshape(snake, 2 * len(snake)).tolist()
                for el in snake:
                    list_to_write.append(el)
                polygons_writer.writerow(list_to_write)
            areas_gt.append(new_area_gt)
            areas_snake.append(new_area_snake)
            iou_test += new_iou
            iter_count += 1
            print('Test. Epoch ' + str(n) + '. Iter ' + str(iter_count) + '/' + str(num_ims) + ', IoU = %.2f' % (
            iou_test / iter_count))
        areas_gt = np.stack(areas_gt)
        areas_snake = np.stack(areas_snake)
        diff = areas_gt - areas_snake
        rmse = np.sqrt(np.sum(diff ** 2) / len(diff))
        print(rmse)
        iou_test /= num_ims
        if not do_train:
            iou_csvfile = open(model_path + 'iuo_train_test.csv', 'a', newline='')
            iou_writer = csv.writer(iou_csvfile)
            iou_writer.writerow([n,iou_train,iou_test])
            iou_csvfile.close()
            polygons_csvfile.close()




