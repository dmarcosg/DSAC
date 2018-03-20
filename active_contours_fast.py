from scipy import interpolate
from PIL import Image, ImageDraw, ImageMath
import numpy as np

def active_contour_step(step_n, Fu, Fv, du, dv, snake_u, snake_v, alpha, beta,
                    kappa, gamma,max_px_move, delta_s):
    """"" Perform one step in the minimization of the snake energy.
    Parameters
    ---------
    Fu, Fv: a MxN numpy arrays with the force fields in u and v
    du, dv: Lx1 numpy arrays with the previous steps (for momentum)
    snake_u, snake_v: Lx1 numpy arrays with the current snake
    alpha, beta: MxN numpy arays with the penalizations
    gamma: time step
    max_px_move: cap to the final step
    delta_s: desired distance between nodes
    Returns
    ----
    snake_u, snake_v: Lx1 numpy arrays with the new snake
    du, dv: Lx1 numpy arrays with the current steps (for momentum)
    """
    L = snake_u.shape[0]
    M = Fu.shape[0]
    N = Fu.shape[1]
    u = np.int32(np.round(snake_u))
    v = np.int32(np.round(snake_v))

    # Explicit time stepping for image energy minimization:

    a = np.zeros(L)
    b = np.zeros(L)
    fu = np.zeros(L)
    fv = np.zeros(L)
    snake_hist = []
    for step in range(step_n):
        for i in range(L):
            a[i] = alpha[u[i,0],v[i,0]]
            b[i] = beta[u[i,0], v[i,0]]
            fu[i] = Fu[u[i,0], v[i,0]]
            fv[i] = Fv[u[i,0], v[i,0]]

        fu = np.reshape(fu,u.shape)
        fv = np.reshape(fv,v.shape)
        am1 = np.concatenate([a[L-1:L],a[0:L-1]],axis=0)
        a0d0 = np.diag(a)
        am1d0 = np.diag(am1)
        a0d1 = np.concatenate([a0d0[0:L,L-1:L], a0d0[0:L,0:L-1]], axis=1)
        am1dm1 = np.concatenate([am1d0[0:L, 1:L], am1d0[0:L, 0:1]], axis=1)

        bm1 = np.concatenate([b[L - 1:L], b[0:L - 1]],axis=0)
        b1 = np.concatenate([b[1:L], b[0:1]],axis=0)
        b0d0 = np.diag(b)
        bm1d0 = np.diag(bm1)
        b1d0 = np.diag(b1)
        b0dm1 = np.concatenate([b0d0[0:L, 1:L], b0d0[0:L, 0:1]], axis=1)
        b0d1 = np.concatenate([b0d0[0:L, L-1:L], b0d0[0:L, 0:L-1]], axis=1)
        bm1dm1 = np.concatenate([bm1d0[0:L, 1:L], bm1d0[0:L, 0:1]], axis=1)
        b1d1 = np.concatenate([b1d0[0:L, L - 1:L], b1d0[0:L, 0:L - 1]], axis=1)
        bm1dm2 = np.concatenate([bm1d0[0:L, 2:L], bm1d0[0:L, 0:2]], axis=1)
        b1d2 = np.concatenate([b1d0[0:L, L - 2:L], b1d0[0:L, 0:L - 2]], axis=1)


        A = -am1dm1  + (a0d0 + am1d0) - a0d1
        B = bm1dm2 - 2*(bm1dm1+b0dm1) + (bm1d0+4*b0d0+b1d0) - 2*(b0d1+b1d1) + b1d2

        #Get kappa values between nodes
        kappa_collection = []
        s = 10
        for i in range(L):
            next_i = i + 1
            if next_i == L:
                next_i = 0
            u_interp = np.int32(np.round(snake_u[i]+range(s)*(snake_u[next_i]-snake_u[i])/s))
            v_interp = np.int32(np.round(snake_v[i]+range(s)*(snake_v[next_i]-snake_v[i])/s))
            kappa_in_segment = []
            for j in range(s):
                kappa_in_segment.append(kappa[u_interp[j], v_interp[j]])
            kappa_collection.append(kappa_in_segment)

        kappa_collection.append(kappa_collection[0])

        #Get the derivative of the balloon energy
        dEb_du = []
        dEb_dv = []
        for i in range(L):
            next_i = i + 1
            prev_i = i - 1
            if next_i == L:
                next_i = 0
            if prev_i == -1:
                prev_i = L-1
            val = 0
            #contribution from the i+1 triangle to dE/du
            int_end = snake_v[next_i] - snake_v[i]
            dh = np.abs(int_end/s)
            for j in range(s):
                val += np.sign(int_end)*(j+1)/s * kappa_collection[i][s-j-1] * dh
            #contribution from the i-1 triangle to dE/du
            int_end = snake_v[prev_i] - snake_v[i]
            dh = np.abs(int_end / s)
            for j in range(s):
                val += -np.sign(int_end)*(j+1)/s * kappa_collection[i][j] * dh
            dEb_du.append(val)

            val = 0
            # contribution from the i+1 triangle to dE/dv
            int_end = snake_u[next_i] - snake_u[i]
            dh = np.abs(int_end / s)
            for j in range(s):
                val += -np.sign(int_end)*(j+1) / s  * kappa_collection[prev_i][s-j-1] * dh
            # contribution from the i-1 triangle to dE/dv
            int_end = snake_u[prev_i] - snake_u[i]
            dh = np.abs(int_end / s)
            for j in range(s):
                val += np.sign(int_end)*(j+1) / s  * kappa_collection[prev_i][j] * dh
            dEb_dv.append(val)
        dEb_du = np.stack(dEb_du)
        dEb_dv = np.stack(dEb_dv)




        # Movements are capped to max_px_move per iteration:
        du = -max_px_move*np.tanh( (fu + 2*np.matmul(A/delta_s+B/np.square(delta_s),snake_u))*gamma )*0.1 + du*0.9
        dv = -max_px_move*np.tanh( (fv + 2*np.matmul(A/delta_s+B/np.square(delta_s),snake_v))*gamma )*0.1 + dv*0.9

        #du += np.multiply(k,n_u)
        #dv += np.multiply(k,n_v)
        du += dEb_du
        dv += dEb_dv

        snake_u += du
        snake_v += dv
        snake_u = np.minimum(snake_u, np.float32(M)-1)
        snake_v = np.minimum(snake_v, np.float32(N)-1)
        snake_u = np.maximum(snake_u, 1)
        snake_v = np.maximum(snake_v, 1)
        snake_hist.append(np.array([snake_u[:, 0], snake_v[:, 0]]).T)

    return snake_u,snake_v,du,dv,snake_hist

def draw_poly(poly,values,im_shape,brush_size):
    """ Returns a MxN (im_shape) array with values in the pixels crossed
    by the edges of the polygon (poly). total_points is the maximum number
    of pixels used for the linear interpolation.
    """
    u = poly[:,0]
    v = poly[:,1]
    b = np.round(brush_size/2)
    image = Image.fromarray(np.zeros(im_shape))
    image2 = Image.fromarray(np.zeros(im_shape))
    d = ImageDraw.Draw(image)
    if type(values) is int:
        values = np.ones(np.shape(u)) * values
    for n in range(len(poly)):
        d.ellipse([(v[n]-b,u[n]-b),(v[n]+b,u[n]+b)], fill=values[n])
        image2 = ImageMath.eval("convert(max(a, b), 'F')", a=image, b=image2)
    return np.array(image2)

def derivatives_poly(poly):
    """
    :param poly: the Lx2 polygon array [u,v]
    :return: der1, der1, Lx2 derivatives arrays
    """
    u = poly[:, 0]
    v = poly[:, 1]
    L = len(u)
    der1_mat = -np.roll(np.eye(L), -1, axis=1) + \
               np.roll(np.eye(L), -1, axis=0)  # first order derivative, central difference
    der2_mat = np.roll(np.eye(L), -1, axis=0) + \
               np.roll(np.eye(L), -1, axis=1) - \
               2 * np.eye(L)  # second order derivative, central difference
    der1 = np.sqrt(np.power(np.matmul(der1_mat, u), 2) + \
                   np.power(np.matmul(der1_mat, v), 2))
    der2 = np.sqrt(np.power(np.matmul(der2_mat, u), 2) + \
                   np.power(np.matmul(der2_mat, v), 2))
    return der1,der2

def draw_poly_fill(poly,im_shape):
    """Returns a MxN (im_shape) array with 1s in the interior of the polygon
    defined by (poly) and 0s outside."""
    u = poly[:, 0]
    v = poly[:, 1]
    image = Image.fromarray(np.zeros(im_shape))
    d = ImageDraw.Draw(image)
    d.polygon(np.column_stack((v, u)).reshape(-1).tolist(), fill=1, outline=1)
    return np.array(image)

def active_countour_gradients(snake,im_shape):
    L = snake.shape[0]
    der1_mat = -np.roll(np.eye(L), -1, axis=1) + \
               np.roll(np.eye(L), -1, axis=0)  # first order derivative, central difference
    der2_mat = np.roll(np.eye(L), -1, axis=0) + \
               np.roll(np.eye(L), -1, axis=1) - \
               2 * np.eye(L)  # second order derivative, central difference
    der1 = np.sqrt(np.power(np.matmul(der1_mat, snake[:, 0]), 2) + \
                   np.power(np.matmul(der1_mat, snake[:, 1]), 2))
    der2 = np.sqrt(np.power(np.matmul(der2_mat, snake[:, 0]), 2) + \
                   np.power(np.matmul(der2_mat, snake[:, 1]), 2))
    der0_img = np.zeros(im_shape)
    der1_img = np.zeros(im_shape)
    der2_img = np.zeros(im_shape)

    [tck, u] = interpolate.splprep([snake[:, 0], snake[:, 1]], s=2, k=1, per=1)
    [xi, yi] = interpolate.splev(np.linspace(0, 1, 200), tck)

    intp_der1 = interpolate.interp1d(u, der1)
    intp_der2 = interpolate.interp1d(u, der2)
    vals_der1 = intp_der1(np.linspace(0, 1, 200))
    vals_der2 = intp_der2(np.linspace(0, 1, 200))
    for n in range(len(xi)):
        print(n)
        der0_img[int(xi[n]), int(yi[n])] = 1
        der1_img[int(xi[n]), int(yi[n])] = vals_der1[n]
        der2_img[int(xi[n]), int(yi[n])] = vals_der2[n]

    gradients = np.zeros([im_shape[0], im_shape[1], 3])
    gradients[:, :, 0] = der0_img
    gradients[:, :, 1] = der1_img
    gradients[:, :, 2] = der2_img
    return gradients