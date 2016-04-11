import numpy as np
import theano
import theano.tensor as T
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


# shape: B x F (F=2n)
def complex_mult(r, u, inverse_r=False, moduli_1=False):
    B, F = u.shape
    r_rl = r[:, :F / 2]
    r_im = r[:, F / 2:]
    if inverse_r:
        if moduli_1:
            r_im = -r_im
        else:
            tmp = r_rl / (r_rl ** 2 + r_im ** 2)
            r_im = -r_im / (r_rl ** 2 + r_im ** 2)
            r_rl = tmp
    u_rl = u[:, :F / 2]
    u_im = u[:, F / 2:]
    res_rl = r_rl * u_rl - r_im * u_im
    res_im = r_rl * u_im + r_im * u_rl
    res = T.concatenate([res_rl, res_im], axis=1)
    return res


# shape: B x F (F=2n)
def read(key, mem):
    value = complex_mult(key, mem, inverse_r=True, moduli_1=True)
    return value


# shape: B x F (F=2n)
def write(key, value, mem):
    coded_value = complex_mult(key, value)
    return mem + coded_value

key = T.matrix('key')
value = T.matrix('value')
mem = T.matrix('mem')

read_func = theano.function([key, mem], read(key, mem))
write_func = theano.function([key, value, mem], write(key, value, mem))


# shape: 20 x 110 x 110 x 3
data = np.load('20_images_from_imagenet.npy')

B = 20
F = 110 * 110 * 3
Num_copies = 1

phis = np.random.random((B * Num_copies, F / 2)) * 2 * np.pi
KEYS = np.concatenate([np.cos(phis), np.sin(phis)], axis=1)

# VALUES
VALUES = data.reshape(B, F) - 0.5

MEM = np.zeros((Num_copies, F))
for i in range(2):
        MEM = write_func(KEYS[i:i + 1], VALUES[i:i + 1], MEM)

all_imgs = []
for i in range(B):
    img = read_func(KEYS[i:i + 1], MEM)
    all_imgs.append(img)

all_imgs = np.concatenate(all_imgs, axis=0)

data = VALUES.reshape(20, 110, 110, 3)
data = np.swapaxes(data, 0, 1)
data = np.reshape(data, (110, 110 * 20, 3))
plt.imshow(data[:, :110 * 2] + 0.5)
plt.show()

all_imgs = all_imgs.reshape(20, 110, 110, 3)
all_imgs = np.swapaxes(all_imgs, 0, 1)
all_imgs = np.reshape(all_imgs, (110, 110 * 20, 3))
plt.imshow(all_imgs[:, :110 * 2] + 0.5)
plt.show()

print np.max(data[:, :110 * 2])
print np.max(all_imgs[:, :110 * 2])

import ipdb; ipdb.set_trace()
