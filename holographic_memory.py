import numpy as np
import theano
import theano.tensor as T
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


# shape: C x B x F (F=2n)
def complex_mult(r, u, inverse_r=False, moduli_1=False):
    _, _, F = u.shape
    r_rl = r[:, :, :F / 2]
    r_im = r[:, :, F / 2:]
    if inverse_r:
        if moduli_1:
            r_im = -r_im
        else:
            tmp = r_rl / (r_rl ** 2 + r_im ** 2)
            r_im = -r_im / (r_rl ** 2 + r_im ** 2)
            r_rl = tmp
    u_rl = u[:, :, :F / 2]
    u_im = u[:, :, F / 2:]
    res_rl = r_rl * u_rl - r_im * u_im
    res_im = r_rl * u_im + r_im * u_rl
    res = T.concatenate([res_rl, res_im], axis=2)
    return res


# key: C x B x F
# mem: C x F
def read(key, mem):
    value = complex_mult(
        key,
        mem.dimshuffle(0, 'x', 1),
        inverse_r=True, moduli_1=True)
    return value.mean(axis=0)


# key: C x B x F
# value: B x F
# mem: C x F
def write(key, value):
    coded_value = complex_mult(key, value.dimshuffle('x', 0, 1))
    return coded_value.sum(axis=1)

key = T.tensor3('key')
value = T.matrix('value')
mem = T.matrix('mem')

read_func = theano.function([key, mem], read(key, mem))
write_func = theano.function([key, value], write(key, value))


# Let's test it!
B = 5
F = 110 * 110 * 3
C = 20

# shape: 20 x 110 x 110 x 3
data = np.load('20_images_from_imagenet.npy')[:B]
VALUES = data.reshape(B, F) - 0.5

phis = np.random.random((C, B, F / 2)) * 2 * np.pi
KEYS = np.concatenate([np.cos(phis), np.sin(phis)], axis=2)

MEM = write_func(KEYS, VALUES)
all_imgs = read_func(KEYS, MEM)

data = VALUES.reshape(B, 110, 110, 3)
data = np.swapaxes(data, 0, 1)
data = np.reshape(data, (110, 110 * B, 3))
plt.imshow(data[:, :110 * B] + 0.5)
plt.show()

all_imgs = all_imgs.reshape(B, 110, 110, 3)
all_imgs = np.swapaxes(all_imgs, 0, 1)
all_imgs = np.reshape(all_imgs, (110, 110 * B, 3))
plt.imshow(all_imgs[:, :110 * B] + 0.5)
plt.show()

import ipdb; ipdb.set_trace()
