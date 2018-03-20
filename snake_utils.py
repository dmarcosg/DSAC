from PIL import Image, ImageOps
import math
import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
import tensorflow as tf



def active_contour_step(Fu, Fv, du, dv, snake_u, snake_v, alpha, beta,kappa,
                    gamma,max_px_move, delta_s):

    L = snake_u.shape[0]
    M = Fu.shape[0]
    N = Fu.shape[1]
    u = tf.cast(tf.round(snake_u),tf.int32)
    v = tf.cast(tf.round(snake_v), tf.int32)


    # Explicit time stepping for image energy minimization:

    fu = tf.gather(tf.reshape(Fu, tf.TensorShape([M * N])), u*M + v)
    fv = tf.gather(tf.reshape(Fv, tf.TensorShape([M * N])), u * M + v)
    a = tf.gather(tf.reshape(alpha, tf.TensorShape([M * N])), u * M + v)
    b = tf.gather(tf.reshape(beta, tf.TensorShape([M * N])), u * M + v)
    a = tf.squeeze(a)
    b = tf.squeeze(b)
    am1 = tf.concat([a[L-1:L],a[0:L-1]],0)
    a0d0 = tf.diag(a)
    am1d0 = tf.diag(am1)
    a0d1 = tf.concat([a0d0[0:L,L-1:L], a0d0[0:L,0:L-1]], 1)
    am1dm1 = tf.concat([am1d0[0:L, 1:L], am1d0[0:L, 0:1]], 1)

    bm1 = tf.concat([b[L - 1:L], b[0:L - 1]],0)
    b1 = tf.concat([b[1:L], b[0:1]],0)
    b0d0 = tf.diag(b)
    bm1d0 = tf.diag(bm1)
    b1d0 = tf.diag(b1)
    b0dm1 = tf.concat([b0d0[0:L, 1:L], b0d0[0:L, 0:1]], 1)
    b0d1 = tf.concat([b0d0[0:L, L-1:L], b0d0[0:L, 0:L-1]], 1)
    bm1dm1 = tf.concat([bm1d0[0:L, 1:L], bm1d0[0:L, 0:1]], 1)
    b1d1 = tf.concat([b1d0[0:L, L - 1:L], b1d0[0:L, 0:L - 1]], 1)
    bm1dm2 = tf.concat([bm1d0[0:L, 2:L], bm1d0[0:L, 0:2]], 1)
    b1d2 = tf.concat([b1d0[0:L, L - 2:L], b1d0[0:L, 0:L - 2]], 1)


    A = -am1dm1  + (a0d0 + am1d0) - a0d1
    B = bm1dm2 - 2*(bm1dm1+b0dm1) + (bm1d0+4*b0d0+b1d0) - 2*(b0d1+b1d1) + b1d2

    # Get kappa values between nodes
    s = 10
    range_float = tf.cast(tf.range(s),tf.float32)
    snake_u1 = tf.concat([snake_u[L-1:L],snake_u[0:L-1]],0)
    snake_v1 = tf.concat([snake_v[L-1:L],snake_v[0:L-1]],0)
    snake_um1 = tf.concat([snake_u[1:L], snake_u[0:1]], 0)
    snake_vm1 = tf.concat([snake_v[1:L], snake_v[0:1]], 0)
    u_interps = tf.cast(snake_u+tf.round(tf.multiply(range_float ,(snake_um1 - snake_u)) / s), tf.int32)
    v_interps = tf.cast(snake_v+tf.round(tf.multiply(range_float ,(snake_vm1 - snake_v)) / s), tf.int32)
    kappa_collection = tf.gather(tf.reshape(kappa, tf.TensorShape([M * N])), u_interps*M  + v_interps)
    #kappa_collection = tf.reshape(kappa_collection,tf.TensorShape([L,s]))
    #kappa_collection = tf.Print(kappa_collection,[kappa_collection],summarize=1000)
    # Get the derivative of the balloon energy
    js = tf.cast(tf.range(1, s + 1),tf.float32)
    s2 = 1 / (s * s)
    int_ends_u_next = s2 * (snake_um1 - snake_u)  # snake_u[next_i] - snake_u[i]
    int_ends_u_prev = s2 * (snake_u1 - snake_u)  # snake_u[prev_i] - snake_u[i]
    int_ends_v_next = s2 * (snake_vm1 - snake_v)  # snake_v[next_i] - snake_v[i]
    int_ends_v_prev = s2 * (snake_v1 - snake_v)  # snake_v[prev_i] - snake_v[i]
    # contribution from the i+1 triangles to dE/du

    dEb_du = tf.multiply(tf.reduce_sum(tf.multiply(js,
              tf.gather(kappa_collection,
              tf.range(s-1,-1,delta=-1),axis=1)),axis=1),
              tf.squeeze(int_ends_v_next))
    dEb_du -= tf.multiply(tf.reduce_sum(tf.multiply(js,
              kappa_collection), axis=1),
              tf.squeeze(int_ends_v_prev))

    dEb_dv = -tf.multiply(tf.reduce_sum(tf.multiply(js,
             tf.gather(tf.gather(kappa_collection,tf.concat([tf.range(L-1,L),tf.range(L-1)],0),axis=0),
             tf.range(s - 1, -1, delta=-1), axis=1)), axis=1),
             tf.squeeze(int_ends_u_next))
    dEb_dv += tf.multiply(tf.reduce_sum(tf.multiply(js,
              tf.gather(kappa_collection,
              tf.concat([tf.range(L-1, L ), tf.range(L - 1)],0), axis=0),), axis=1),
              tf.squeeze(int_ends_u_prev))
    #dEb_du = np.sum(js * kappa_collection[:, np.arange(s - 1, -1, -1)], axis=1) * int_ends_v_next.squeeze()
    #dEb_du -= np.sum(js * kappa_collection[:, js - 1], axis=1) * int_ends_v_prev.squeeze()
    #dEb_dv = -np.sum(js * kappa_collection[np.roll(np.arange(L), 1), :][:, np.arange(s - 1, -1, -1)],
    #                 axis=1) * int_ends_u_next.squeeze()
    #dEb_dv += np.sum(js * kappa_collection[np.roll(np.arange(L), 1), :][:, js - 1], axis=1) * int_ends_u_prev.squeeze()




    # Movements are capped to max_px_move per iteration:
    #du = -max_px_move*tf.tanh( (fu - tf.reshape(dEb_du,fu.shape) + 2*tf.matmul(A/delta_s+B/tf.square(delta_s),snake_u))*gamma )*0.5 + du*0.5
    #dv = -max_px_move*tf.tanh( (fv - tf.reshape(dEb_dv,fv.shape) + 2*tf.matmul(A/delta_s+B/tf.square(delta_s),snake_v))*gamma )*0.5 + dv*0.5
    du = -max_px_move * tf.tanh((fu  - tf.reshape(dEb_du,fu.shape))*gamma )*0.5 + du*0.5
    dv = -max_px_move*tf.tanh( (fv  - tf.reshape(dEb_dv,fv.shape))*gamma )*0.5 + dv*0.5
    snake_u = tf.matmul(tf.matrix_inverse(tf.eye(L._value) + 2*gamma*(A/delta_s + B/(delta_s*delta_s))), snake_u + gamma * du)
    snake_v = tf.matmul(tf.matrix_inverse(tf.eye(L._value) + 2 * gamma * (A / delta_s + B / (delta_s * delta_s))), snake_v + gamma * dv)

    #snake_u = np.matmul(np.linalg.inv(np.eye(L, L) + 2 * gamma * (A / delta_s + B / np.square(delta_s))),
    #                    snake_u + gamma * du)
    #snake_v = np.matmul(np.linalg.inv(np.eye(L, L) + 2 * gamma * (A / delta_s + B / np.square(delta_s))),
    #                    snake_v + gamma * dv)

    #snake_u += du
    #snake_v += dv
    snake_u = tf.minimum(snake_u, tf.cast(M,tf.float32)-1)
    snake_v = tf.minimum(snake_v, tf.cast(N,tf.float32)-1)
    snake_u = tf.maximum(snake_u, 1)
    snake_v = tf.maximum(snake_v, 1)

    return snake_u,snake_v,du,dv

def imrotate(img, angle, fill='black',resample='bilinear'):
    """Rotate the given PIL.Image counter clockwise around its centre by angle degrees.
    Empty region will be padded with color specified in fill."""
    img = Image.fromarray(img)
    theta = math.radians(angle)
    w, h = img.size
    diameter = math.sqrt(w * w + h * h)
    theta_0 = math.atan(float(h) / w)
    w_new = diameter * max(abs(math.cos(theta-theta_0)), abs(math.cos(theta+theta_0)))
    h_new = diameter * max(abs(math.sin(theta-theta_0)), abs(math.sin(theta+theta_0)))
    pad = math.ceil(max(w_new - w, h_new - h) / 2)
    img = ImageOps.expand(img, border=int(pad), fill=fill)
    if resample is 'bicubic':
        img = img.rotate(angle, resample=Image.BICUBIC)
    elif resample is 'bilinear':
        img = img.rotate(angle, resample=Image.BILINEAR)
    elif resample is 'nearest':
        img = img.rotate(angle, resample=Image.NEAREST)
    else:
        print('Dunno what interpolation method ' + resample + ' is.')
    return np.array(img.crop((pad, pad, w + pad, h + pad)))

def polygon_area(u,v):
    return np.sum(u[:-1]*v[1:]) - np.sum(u[1:]*v[:-1])

def plot_snakes(snake,snake_hist,GT,mapE, mapA, mapB, mapK, grads_arrayE, grads_arrayA, grads_arrayB, grads_arrayK, image, mask):

    # Plot result
    fig0, (ax) = plt.subplots(ncols=5,nrows=1)
    im = ax[0].imshow(scipy.misc.imresize(np.abs(image[:,:,:,0]),mapE[:, :, 0, 0].shape))
    for i in range(0,len(snake_hist),5):
        ax[0].plot(snake_hist[i][:, 1], snake_hist[i][:, 0], '-.', color=[i / len(snake_hist), i / len(snake_hist), 1-i / len(snake_hist)], lw=3)
    if not GT is None:
        ax[0].plot(GT[:, 1], GT[:, 0], '-', color=[0.2, 1, 0.2], lw=3)
    ax[0].plot(snake[:, 1], snake[:, 0], '--', lw=3,color=[1,1,0])
    ax[0].axis('off')
    ax[0].set_title(r'a) image $\mathbf{x}$', y=-0.2)
    plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04).remove()

    #plt.colorbar(im, ax=ax)
    #fig0.suptitle('Image, GT (red) and converged snake (black)', fontsize=20)

    im0 = ax[1].imshow(mapE[:, :, 0, 0])
    plt.colorbar(im0, ax=ax[1],fraction=0.046, pad=0.04)
    ax[1].axis('off')
    ax[1].set_title(r'b) data term $D(\mathbf{x})$', y=-0.2)

    im1 = ax[2].imshow(mapK[:, :, 0, 0])
    plt.colorbar(im1, ax=ax[2],fraction=0.046, pad=0.04)
    ax[2].axis('off')
    ax[2].set_title(r'c) balloon term $\kappa(\mathbf{x})$', y=-0.2)
    im2 = ax[3].imshow(mapB[:, :, 0, 0])
    plt.colorbar(im2, ax=ax[3],fraction=0.046, pad=0.04)
    ax[3].axis('off')
    ax[3].set_title(r'd) thin plate term $ \beta(\mathbf{x})$', y=-0.2)

    im3 = ax[4].imshow(mapA[:, :, 0, 0])
    plt.colorbar(im3, ax=ax[4], fraction=0.046, pad=0.04)
    ax[4].axis('off')
    ax[4].set_title(r'd) membrane term $ \alpha(\mathbf{x})$', y=-0.2)


    if not grads_arrayE is None:
        fig0, (ax) = plt.subplots(ncols=1)
        im = ax.imshow(mask[:, :, 0, 0])
        ax.plot(GT[:, 1], GT[:, 0], '--r', lw=3)
        plt.colorbar(im, ax=ax)

        fig1, ax = plt.subplots(ncols=2,nrows=2)
        im0 = ax[0,0].imshow(mapE[:, :, 0, 0])
        plt.colorbar(im0, ax=ax[0,0])
        ax[0, 0].set_title('D')
        im1 = ax[0,1].imshow(mapA[:, :, 0, 0])
        plt.colorbar(im1, ax=ax[0,1])
        ax[0, 1].set_title('alpha')
        im2 = ax[1,0].imshow(mapB[:, :, 0, 0])
        plt.colorbar(im2, ax=ax[1,0])
        ax[1, 0].set_title('beta')
        im3 = ax[1,1].imshow(mapK[:, :, 0, 0])
        plt.colorbar(im3, ax=ax[1,1])
        ax[1, 1].set_title('kappa')
        fig1.suptitle('Output maps', fontsize=20)

        fig2, ax = plt.subplots(ncols=2,nrows=2)
        im0 = ax[0,0].imshow(grads_arrayE[:, :, 0, 0])
        plt.colorbar(im0, ax=ax[0,0])
        ax[0, 0].set_title('D')
        im1 = ax[0,1].imshow(grads_arrayA[:, :, 0, 0])
        plt.colorbar(im1, ax=ax[0,1])
        ax[0, 1].set_title('alpha')
        im2 = ax[1,0].imshow(grads_arrayB[:, :, 0, 0])
        plt.colorbar(im2, ax=ax[1,0])
        ax[1, 0].set_title('beta')
        im3 = ax[1,1].imshow(grads_arrayK[:, :, 0, 0])
        plt.colorbar(im3, ax=ax[1,1])
        ax[1, 1].set_title('kappa')
        fig2.suptitle('Gradient maps', fontsize=20)

    plt.show()

def plot_for_figure(snake, snake_hist, GT, mapE, mapA, mapB, mapK, grads_arrayE, grads_arrayA, grads_arrayB,
                    grads_arrayK, image, mask):
    # Plot result
    fig0, (ax) = plt.subplots(ncols=4, nrows=2)
    im = ax[0,0].imshow(scipy.misc.imresize(np.abs(image[:, :, :, 0]), mapE[:, :, 0, 0].shape))

    im1 = ax[0, 1].imshow(scipy.misc.imresize(np.abs(image[:, :, :, 0])*0+255, mapE[:, :, 0, 0].shape))
    ax[0,1].plot(snake_hist[0][:, 1], snake_hist[0][:, 0], '--', color=[0.6, 0.2, 0], lw=3)


    im2 = ax[0, 2].imshow(scipy.misc.imresize(np.abs(image[:, :, :, 0]) * 0 + 255, mapE[:, :, 0, 0].shape))
    ax[0,2].plot(snake_hist[1][:, 1], snake_hist[1][:, 0], '--', color=[0, 0, 0.5], lw=3)


    im3 = ax[0, 3].imshow(scipy.misc.imresize(np.abs(image[:, :, :, 0]) * 0 + 255, mapE[:, :, 0, 0].shape))
    ax[0,3].plot(GT[:, 1], GT[:, 0], '--', color=[0.2, 1, 0.2], lw=3)

    # ax[0].plot(snake[:, 1], snake[:, 0], '--b', lw=3)
    # ax[0].axis('off')
    # ax[0].set_title(r'a) image $\mathbf{x}$', y=-0.2)
    # plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04).remove()

    # plt.colorbar(im, ax=ax)
    # fig0.suptitle('Image, GT (red) and converged snake (black)', fontsize=20)

    im0 = ax[1,0].imshow(mapE[:, :, 0, 0])
    # plt.colorbar(im0, ax=ax[4],fraction=0.046, pad=0.04)
    # ax[1].axis('off')
    # ax[1].set_title(r'b) data term $D(\mathbf{x})$', y=-0.2)

    im1 = ax[1,1].imshow(mapK[:, :, 0, 0])
    # plt.colorbar(im1, ax=ax[5],fraction=0.046, pad=0.04)
    # ax[2].axis('off')
    # ax[2].set_title(r'c) balloon term $\kappa(\mathbf{x})$', y=-0.2)
    im2 = ax[1,2].imshow(mapB[:, :, 0, 0])
    # plt.colorbar(im2, ax=ax[3],fraction=0.046, pad=0.04)
    # ax[6].axis('off')
    # ax[6].set_title(r'd) thin plate term $ \beta(\mathbf{x})$', y=-0.2)

    im3 = ax[1,3].imshow(mapA[:, :, 0, 0])
    # plt.colorbar(im3, ax=ax[7], fraction=0.046, pad=0.04)
    # ax[7].axis('off')
    # ax[7].set_title(r'd) membrane term $ \alpha(\mathbf{x})$', y=-0.2)
    for i in range(4):
        ax[0,i].axis('off')
        ax[1, i].axis('off')

    plt.show()

def weight_variable(shape,wd=0.0):
    initial = tf.truncated_normal(shape, stddev=0.1)
    var = tf.Variable(initial)
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    return var

def gaussian_filter(shape,sigma):
    x, y = [int(np.floor(edge / 2)) for edge in shape]
    grid = np.array([[((i ** 2 + j ** 2) / (2.0 * sigma ** 2)) for i in range(-x, x + 1)] for j in range(-y, y + 1)])
    filt = np.exp(-grid) / (2 * np.pi * sigma ** 2)
    filt /= np.sum(filt)
    var = np.zeros((shape[0],shape[1],1,1))
    var[:,:,0,0] = filt
    return tf.constant(np.float32(var))

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def batch_norm(x):
    batch_mean, batch_var = tf.nn.moments(x, [0,1,2])
    scale = tf.Variable(tf.ones(batch_mean.shape))
    beta = tf.Variable(tf.zeros(batch_mean.shape))
    return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, 1e-7)

def CNN(im_size,out_size,L,batch_size=1,layers = 5, wd=0.001, numfilt=0):

    #Input and output
    x = tf.placeholder(tf.float32, shape=[im_size, im_size, 3, batch_size])
    x_image = tf.reshape(x, [-1, im_size, im_size, 3])
    y_ = tf.placeholder(tf.float32, shape=[L,2])

    W_conv = []
    b_conv = []
    h_conv = []
    h_pool = []
    resized_out = []
    W_conv.append(weight_variable([3, 3, 3, 32], wd=wd))
    b_conv.append(bias_variable([32]))
    h_conv.append(tf.nn.relu(conv2d(x_image, W_conv[-1],padding='VALID') + b_conv[-1]))
    h_pool.append(batch_norm(max_pool_2x2(h_conv[-1])))
    for layer in range(1,layers):
        W_conv.append(weight_variable([3, 3, 32+(layer-1)*numfilt, 32+layer*numfilt],wd=wd))
        b_conv.append(bias_variable([32+layer*numfilt]))
        h_conv.append(tf.nn.relu(conv2d(h_pool[-1], W_conv[-1],padding='VALID') + b_conv[-1]))
        h_pool.append(batch_norm(max_pool_2x2(h_conv[-1])))
        if layer > layers - 3:
            resized_out.append(tf.image.resize_images(h_conv[-1], [out_size, out_size]))

    h_concat = tf.concat(resized_out,3)

    # MLP for dimension reduction
    W_convd = weight_variable([1, 1, int(h_concat.shape[3]), 32+2*numfilt], wd=wd)
    b_convd = bias_variable([32+2*numfilt])
    h_convd = batch_norm(tf.nn.relu(conv2d(h_concat, W_convd) + b_convd))

    #Final conv layer
    W_convf = weight_variable([3, 3, 32+2*numfilt, 32],wd=wd)
    b_convf = bias_variable([32])
    h_convf = batch_norm(tf.nn.relu(conv2d(h_convd, W_convf) + b_convf))

    #Predict energy
    W_fcE = weight_variable([1, 1, 32, 1],wd=wd)
    b_fcE = bias_variable([1])
    h_fcE = conv2d(h_convf, W_fcE) + b_fcE
    G_filt = gaussian_filter((15,15), 2)
    predE = tf.reshape(conv2d(h_fcE,G_filt), [out_size, out_size, 1, -1])

    # Predict alpha
    W_fcA = weight_variable([1, 1, 32, 1],wd=wd)
    b_fcA = bias_variable([1])
    h_fcA = conv2d(h_convf, W_fcA) + b_fcA
    h_fcA = tf.reduce_mean(h_fcA) + h_fcA * 0
    # predA = tf.nn.softplus(tf.reshape(h_fcA,[im_size,im_size,1,-1]))
    predA = tf.reshape(h_fcA, [out_size, out_size, 1, -1])
    # Predict beta
    W_fcB = weight_variable([1, 1, 32, 1],wd=wd)
    b_fcB = bias_variable([1])
    h_fcB = conv2d(h_convf, W_fcB) + b_fcB
    #h_fcB = tf.log(1+tf.exp(h_fcB))
    predB = tf.reshape(h_fcB, [out_size, out_size, 1, -1])
    # Predict kappa
    W_fcK = weight_variable([1, 1, 32, 1],wd=wd)
    b_fcK = bias_variable([1])
    h_fcK = conv2d(h_convf, W_fcK) + b_fcK
    #h_fcK = tf.log(1+tf.exp(h_fcK))
    predK = tf.reshape(h_fcK, [out_size, out_size, 1, -1])

    #Inject the gradients
    grad_predE = tf.placeholder(tf.float32, shape=[out_size, out_size, 1, batch_size])
    grad_predA = tf.placeholder(tf.float32, shape=[out_size, out_size, 1, batch_size])
    grad_predB = tf.placeholder(tf.float32, shape=[out_size, out_size, 1, batch_size])
    grad_predK = tf.placeholder(tf.float32, shape=[out_size, out_size, 1, batch_size])
    l2loss = tf.add_n(tf.get_collection('losses'), name='l2_loss')
    grad_l2loss = tf.placeholder(tf.float32, shape=[])
    tvars = tf.trainable_variables()
    grads = tf.gradients([predE,predA,predB,predK,l2loss], tvars, grad_ys = [grad_predE,grad_predA,grad_predB,grad_predK,grad_l2loss])

    return tvars,grads,predE, predA, predB, predK, l2loss, grad_predE, grad_predA, grad_predB, grad_predK, grad_l2loss, x,y_

def CNN_B(im_size,out_size,L,batch_size=1,layers = 5, wd=0.001, numfilt=None, E_blur=2,stack_from=2):

    if numfilt is None:
        numfilt = np.ones(layers,dtype=np.int32)*32
    #Input and output
    x = tf.placeholder(tf.float32, shape=[im_size, im_size, 3, batch_size])
    x_image = tf.reshape(x, [-1, im_size, im_size, 3])
    y_ = tf.placeholder(tf.float32, shape=[L,2])

    W_conv = []
    b_conv = []
    h_conv = []
    h_pool = []
    resized_out = []
    W_conv.append(weight_variable([7, 7, 3, numfilt[0]], wd=wd))
    b_conv.append(bias_variable([numfilt[0]]))
    h_conv.append(tf.nn.relu(conv2d(x_image, W_conv[-1],padding='SAME') + b_conv[-1]))
    h_pool.append(batch_norm(max_pool_2x2(h_conv[-1])))

    for layer in range(1,layers):
        if layer == 1:
            W_conv.append(weight_variable([5, 5, numfilt[layer-1], numfilt[layer]],wd=wd))
        else:
            W_conv.append(weight_variable([3, 3, numfilt[layer-1], numfilt[layer]], wd=wd))
        b_conv.append(bias_variable([numfilt[layer]]))
        h_conv.append(tf.nn.relu(conv2d(h_pool[-1], W_conv[-1],padding='SAME') + b_conv[-1]))
        h_pool.append(batch_norm(max_pool_2x2(h_conv[-1])))
        if layer >= stack_from:
            resized_out.append(tf.image.resize_images(h_conv[-1], [out_size, out_size]))

    h_concat = tf.concat(resized_out,3)

    # MLP for dimension reduction
    W_convd = weight_variable([1, 1, int(h_concat.shape[3]), 256], wd=wd)
    b_convd = bias_variable([256])
    h_convd = batch_norm(tf.nn.relu(conv2d(h_concat, W_convd) + b_convd))

    # MLP for dimension reduction
    W_convf = weight_variable([1, 1, 256, 64],wd=wd)
    b_convf = bias_variable([64])
    h_convf = batch_norm(tf.nn.relu(conv2d(h_convd, W_convf) + b_convf))

    #Predict energy
    W_fcE = weight_variable([1, 1, 64, 1],wd=wd)
    b_fcE = bias_variable([1])
    h_fcE = conv2d(h_convf, W_fcE) + b_fcE
    G_filt = gaussian_filter((9,9), E_blur)
    predE = tf.reshape(conv2d(h_fcE,G_filt), [out_size, out_size, 1, -1])

    # Predict alpha
    W_fcA = weight_variable([1, 1, 64, 1],wd=wd)
    b_fcA = bias_variable([1])
    h_fcA = conv2d(h_convf, W_fcA) + b_fcA
    h_fcA = tf.reduce_mean(h_fcA) + h_fcA * 0
    # predA = tf.nn.softplus(tf.reshape(h_fcA,[im_size,im_size,1,-1]))
    predA = tf.reshape(h_fcA, [out_size, out_size, 1, -1])
    # Predict beta
    W_fcB = weight_variable([1, 1, 64, 1],wd=wd)
    b_fcB = bias_variable([1])
    h_fcB = conv2d(h_convf, W_fcB) + b_fcB
    #h_fcB = tf.log(1+tf.exp(h_fcB))
    predB = tf.reshape(h_fcB, [out_size, out_size, 1, -1])
    # Predict kappa
    W_fcK = weight_variable([1, 1, 64, 1],wd=wd)
    b_fcK = bias_variable([1])
    h_fcK = conv2d(h_convf, W_fcK) + b_fcK
    #h_fcK = tf.log(1+tf.exp(h_fcK))
    predK = tf.reshape(h_fcK, [out_size, out_size, 1, -1])

    #Inject the gradients
    grad_predE = tf.placeholder(tf.float32, shape=[out_size, out_size, 1, batch_size])
    grad_predA = tf.placeholder(tf.float32, shape=[out_size, out_size, 1, batch_size])
    grad_predB = tf.placeholder(tf.float32, shape=[out_size, out_size, 1, batch_size])
    grad_predK = tf.placeholder(tf.float32, shape=[out_size, out_size, 1, batch_size])
    l2loss = tf.add_n(tf.get_collection('losses'), name='l2_loss')
    grad_l2loss = tf.placeholder(tf.float32, shape=[])
    tvars = tf.trainable_variables()
    grads = tf.gradients([predE,predA,predB,predK,l2loss], tvars, grad_ys = [grad_predE,grad_predA,grad_predB,grad_predK,grad_l2loss])

    return tvars,grads,predE, predA, predB, predK, l2loss, grad_predE, grad_predA, grad_predB, grad_predK, grad_l2loss, x,y_

def CNN_B_alpha(im_size,out_size,L,batch_size=1,layers = 5, wd=0.001, numfilt=None, E_blur=2,stack_from=2):

    if numfilt is None:
        numfilt = np.ones(layers,dtype=np.int32)*32
    #Input and output
    x = tf.placeholder(tf.float32, shape=[im_size, im_size, 3, batch_size])
    x_image = tf.reshape(x, [-1, im_size, im_size, 3])
    y_ = tf.placeholder(tf.float32, shape=[L,2])

    W_conv = []
    b_conv = []
    h_conv = []
    h_pool = []
    resized_out = []
    W_conv.append(weight_variable([7, 7, 3, numfilt[0]], wd=wd))
    b_conv.append(bias_variable([numfilt[0]]))
    h_conv.append(tf.nn.relu(conv2d(x_image, W_conv[-1],padding='SAME') + b_conv[-1]))
    h_pool.append(batch_norm(max_pool_2x2(h_conv[-1])))

    for layer in range(1,layers):
        if layer == 1:
            W_conv.append(weight_variable([5, 5, numfilt[layer-1], numfilt[layer]],wd=wd))
        else:
            W_conv.append(weight_variable([3, 3, numfilt[layer-1], numfilt[layer]], wd=wd))
        b_conv.append(bias_variable([numfilt[layer]]))
        h_conv.append(tf.nn.relu(conv2d(h_pool[-1], W_conv[-1],padding='SAME') + b_conv[-1]))
        h_pool.append(batch_norm(max_pool_2x2(h_conv[-1])))
        if layer >= stack_from:
            resized_out.append(tf.image.resize_images(h_conv[-1], [out_size, out_size]))

    h_concat = tf.concat(resized_out,3)

    # MLP for dimension reduction
    W_convd = weight_variable([1, 1, int(h_concat.shape[3]), 256], wd=wd)
    b_convd = bias_variable([256])
    h_convd = batch_norm(tf.nn.relu(conv2d(h_concat, W_convd) + b_convd))

    # MLP for dimension reduction
    W_convf = weight_variable([1, 1, 256, 64],wd=wd)
    b_convf = bias_variable([64])
    h_convf = batch_norm(tf.nn.relu(conv2d(h_convd, W_convf) + b_convf))

    #Predict energy
    W_fcE = weight_variable([1, 1, 64, 1],wd=wd)
    b_fcE = bias_variable([1])
    h_fcE = conv2d(h_convf, W_fcE) + b_fcE
    G_filt = gaussian_filter((9,9), E_blur)
    predE = tf.reshape(conv2d(h_fcE,G_filt), [out_size, out_size, 1, -1])

    # Predict alpha
    W_fcA = weight_variable([1, 1, 64, 1],wd=wd)
    b_fcA = bias_variable([1])
    h_fcA = conv2d(h_convf, W_fcA) + b_fcA
    #h_fcA = tf.reduce_mean(h_fcA) + h_fcA * 0
    # predA = tf.nn.softplus(tf.reshape(h_fcA,[im_size,im_size,1,-1]))
    predA = tf.reshape(h_fcA, [out_size, out_size, 1, -1])
    # Predict beta
    W_fcB = weight_variable([1, 1, 64, 1],wd=wd)
    b_fcB = bias_variable([1])
    h_fcB = conv2d(h_convf, W_fcB) + b_fcB
    #h_fcB = tf.log(1+tf.exp(h_fcB))
    predB = tf.reshape(h_fcB, [out_size, out_size, 1, -1])
    # Predict kappa
    W_fcK = weight_variable([1, 1, 64, 1],wd=wd)
    b_fcK = bias_variable([1])
    h_fcK = conv2d(h_convf, W_fcK) + b_fcK
    #h_fcK = tf.log(1+tf.exp(h_fcK))
    predK = tf.reshape(h_fcK, [out_size, out_size, 1, -1])

    #Inject the gradients
    grad_predE = tf.placeholder(tf.float32, shape=[out_size, out_size, 1, batch_size])
    grad_predA = tf.placeholder(tf.float32, shape=[out_size, out_size, 1, batch_size])
    grad_predB = tf.placeholder(tf.float32, shape=[out_size, out_size, 1, batch_size])
    grad_predK = tf.placeholder(tf.float32, shape=[out_size, out_size, 1, batch_size])
    l2loss = tf.add_n(tf.get_collection('losses'), name='l2_loss')
    grad_l2loss = tf.placeholder(tf.float32, shape=[])
    tvars = tf.trainable_variables()
    grads = tf.gradients([predE,predA,predB,predK,l2loss], tvars, grad_ys = [grad_predE,grad_predA,grad_predB,grad_predK,grad_l2loss])

    return tvars,grads,predE, predA, predB, predK, l2loss, grad_predE, grad_predA, grad_predB, grad_predK, grad_l2loss, x,y_

def CNN_B_scalar(im_size,out_size,L,batch_size=1,layers = 5, wd=0.001, numfilt=None, E_blur=2,stack_from=2):

    if numfilt is None:
        numfilt = np.ones(layers,dtype=np.int32)*32
    #Input and output
    x = tf.placeholder(tf.float32, shape=[im_size, im_size, 3, batch_size])
    x_image = tf.reshape(x, [-1, im_size, im_size, 3])
    y_ = tf.placeholder(tf.float32, shape=[L,2])

    W_conv = []
    b_conv = []
    h_conv = []
    h_pool = []
    resized_out = []
    W_conv.append(weight_variable([7, 7, 3, numfilt[0]], wd=wd))
    b_conv.append(bias_variable([numfilt[0]]))
    h_conv.append(tf.nn.relu(conv2d(x_image, W_conv[-1],padding='SAME') + b_conv[-1]))
    h_pool.append(batch_norm(max_pool_2x2(h_conv[-1])))

    for layer in range(1,layers):
        if layer == 1:
            W_conv.append(weight_variable([5, 5, numfilt[layer-1], numfilt[layer]],wd=wd))
        else:
            W_conv.append(weight_variable([3, 3, numfilt[layer-1], numfilt[layer]], wd=wd))
        b_conv.append(bias_variable([numfilt[layer]]))
        h_conv.append(tf.nn.relu(conv2d(h_pool[-1], W_conv[-1],padding='SAME') + b_conv[-1]))
        h_pool.append(batch_norm(max_pool_2x2(h_conv[-1])))
        if layer >= stack_from:
            resized_out.append(tf.image.resize_images(h_conv[-1], [out_size, out_size]))

    h_concat = tf.concat(resized_out,3)

    # MLP for dimension reduction
    W_convd = weight_variable([1, 1, int(h_concat.shape[3]), 256], wd=wd)
    b_convd = bias_variable([256])
    h_convd = batch_norm(tf.nn.relu(conv2d(h_concat, W_convd) + b_convd))

    # MLP for dimension reduction
    W_convf = weight_variable([1, 1, 256, 64],wd=wd)
    b_convf = bias_variable([64])
    h_convf = batch_norm(tf.nn.relu(conv2d(h_convd, W_convf) + b_convf))

    #Predict energy
    W_fcE = weight_variable([1, 1, 64, 1],wd=wd)
    b_fcE = bias_variable([1])
    h_fcE = conv2d(h_convf, W_fcE) + b_fcE
    G_filt = gaussian_filter((9,9), E_blur)
    predE = tf.reshape(conv2d(h_fcE,G_filt), [out_size, out_size, 1, -1])

    # Predict alpha
    W_fcA = weight_variable([1, 1, 64, 1],wd=wd)
    b_fcA = bias_variable([1])
    h_fcA = conv2d(h_convf, W_fcA) + b_fcA
    h_fcA = tf.reduce_mean(h_fcA) + h_fcA * 0
    # predA = tf.nn.softplus(tf.reshape(h_fcA,[im_size,im_size,1,-1]))
    predA = tf.reshape(h_fcA, [out_size, out_size, 1, -1])
    # Predict beta
    W_fcB = weight_variable([1, 1, 64, 1],wd=wd)
    b_fcB = bias_variable([1])
    h_fcB = conv2d(h_convf, W_fcB) + b_fcB
    h_fcB = tf.reduce_mean(h_fcB) + h_fcB * 0
    #h_fcB = tf.log(1+tf.exp(h_fcB))
    predB = tf.reshape(h_fcB, [out_size, out_size, 1, -1])
    # Predict kappa
    W_fcK = weight_variable([1, 1, 64, 1],wd=wd)
    b_fcK = bias_variable([1])
    h_fcK = conv2d(h_convf, W_fcK) + b_fcK
    h_fcK = tf.reduce_mean(h_fcK) + h_fcK * 0
    #h_fcK = tf.log(1+tf.exp(h_fcK))
    predK = tf.reshape(h_fcK, [out_size, out_size, 1, -1])

    #Inject the gradients
    grad_predE = tf.placeholder(tf.float32, shape=[out_size, out_size, 1, batch_size])
    grad_predA = tf.placeholder(tf.float32, shape=[out_size, out_size, 1, batch_size])
    grad_predB = tf.placeholder(tf.float32, shape=[out_size, out_size, 1, batch_size])
    grad_predK = tf.placeholder(tf.float32, shape=[out_size, out_size, 1, batch_size])
    l2loss = tf.add_n(tf.get_collection('losses'), name='l2_loss')
    grad_l2loss = tf.placeholder(tf.float32, shape=[])
    tvars = tf.trainable_variables()
    grads = tf.gradients([predE,predA,predB,predK,l2loss], tvars, grad_ys = [grad_predE,grad_predA,grad_predB,grad_predK,grad_l2loss])

    return tvars,grads,predE, predA, predB, predK, l2loss, grad_predE, grad_predA, grad_predB, grad_predK, grad_l2loss, x,y_

def snake_graph(out_size,L,niter=100):
    tf_alpha = tf.placeholder(tf.float32, shape=[out_size, out_size])
    tf_beta = tf.placeholder(tf.float32, shape=[out_size, out_size])
    tf_kappa = tf.placeholder(tf.float32, shape=[out_size, out_size])
    tf_Du = tf.placeholder(tf.float32, shape=[out_size, out_size])
    tf_Dv = tf.placeholder(tf.float32, shape=[out_size, out_size])
    tf_u0 = tf.placeholder(tf.float32, shape=[L, 1])
    tf_v0 = tf.placeholder(tf.float32, shape=[L, 1])
    tf_du0 = tf.placeholder(tf.float32, shape=[L, 1])
    tf_dv0 = tf.placeholder(tf.float32, shape=[L, 1])
    gamma = tf.constant(1, dtype=tf.float32)
    max_px_move = tf.constant(1, dtype=tf.float32)
    delta_s = tf.constant(out_size / L, dtype=tf.float32)

    tf_u = tf_u0
    tf_du = tf_du0
    tf_v = tf_v0
    tf_dv = tf_dv0

    for i in range(niter):
        tf_u, tf_v, tf_du, tf_dv = active_contour_step(tf_Du, tf_Dv, tf_du, tf_dv, tf_u, tf_v,
                                                       tf_alpha, tf_beta, tf_kappa,
                                                       gamma, max_px_move, delta_s)
    return tf_u, tf_v, tf_du, tf_dv, tf_Du, tf_Dv, tf_u0, tf_v0, tf_du0, tf_dv0, tf_alpha, tf_beta, tf_kappa