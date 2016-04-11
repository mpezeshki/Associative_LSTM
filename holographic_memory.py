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


KEYS = np.array([[0.50],
                 [0.03]])
KEYS = np.concatenate([KEYS, np.sqrt(1 - KEYS ** 2)], axis=1)
random_sign = np.random.randint(2, size=KEYS.shape) * 2 - 1
KEYS = random_sign * KEYS

VALUES = np.array([[0.43, 0.20],
                   [0.23, 0.60]])
MEM = np.zeros((1, 2))
MEM = write_func(KEYS[0:1], VALUES[0:1], MEM)
MEM = write_func(KEYS[1:2], VALUES[1:2], MEM)
print VALUES[0:1]
print read_func(KEYS[0:1], MEM)


# shape: 20 x 110 x 110 x 3
data = np.load('20_images_from_imagenet.npy')

B = 20
F = 110 * 110 * 3
Num_copies = 1
# random KEYS with a moduli of one
# KEYS = np.random.random((B, F / 2))
# KEYS = np.concatenate([KEYS, np.sqrt(1 - KEYS ** 2)], axis=1)
# random_sign = np.random.randint(B, size=KEYS.shape) * 2 - 1
# KEYS = random_sign * KEYS

# KEYS = np.random.randn(B * Num_copies, F)
# KEYS = KEYS / np.linalg.norm(KEYS)

phis = np.random.random((B * Num_copies, F / 2)) * 2 * np.pi
KEYS_rl = np.array([np.cos(phi) for phi in phis.T]).T
KEYS_im = np.array([np.sin(phi) for phi in phis.T]).T
KEYS = np.concatenate([KEYS_rl, KEYS_im], axis=1)

# VALUES
VALUES = data.reshape(B, F)
VALUES = VALUES - 0.5

MEM = np.zeros((1, F))
for i in range(1):
    for j in range(Num_copies):
        MEM = write_func(KEYS[i * Num_copies + j:i * Num_copies + j + 1],
                         VALUES[i:i + 1], MEM)

all_imgs = []
for i in range(B):
    copies = []
    for j in range(Num_copies):
        img = read_func(KEYS[Num_copies * i + j:Num_copies * i + j + 1], MEM)
        copies.append(img)
    all_imgs.append(np.mean(copies, axis=0))

all_imgs = np.concatenate(all_imgs, axis=0)

data = VALUES.reshape(20, 110, 110, 3)
data = np.swapaxes(data, 0, 1)
data = np.reshape(data, (110, 110 * 20, 3))
plt.imshow(data[:, :110 * 2])
plt.show()

all_imgs = all_imgs.reshape(20, 110, 110, 3)
all_imgs = np.swapaxes(all_imgs, 0, 1)
all_imgs = np.reshape(all_imgs, (110, 110 * 20, 3))
plt.imshow(all_imgs[:, :110 * 2])
plt.show()

print VALUES[0]
print all_imgs[0]

import ipdb; ipdb.set_trace()
