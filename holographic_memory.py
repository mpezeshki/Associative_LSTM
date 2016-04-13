import numpy as np
import theano
import theano.tensor as T
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

B = 10
F = 110 * 110 * 3
C = 20

# shape: C x F/2
permutations = []
indices = np.arange(F / 2)
for i in range(C):
    np.random.shuffle(indices)
    permutations.append(np.concatenate(
        [indices,
         [ind + F / 2 for ind in indices]]))
# C x F (numpy)
PERMUTATIONS = np.vstack(permutations)


# input: B x F
# output: C x B x F
def permute(input):
    inputs_permuted = []
    for i in range(PERMUTATIONS.shape[0]):
        inputs_permuted.append(
            input[:, PERMUTATIONS[i]].dimshuffle('x', 0, 1))
    return T.concatenate(inputs_permuted, axis=0)


# r: C x B x F
# u: if mem: C x 1 x F
# u: if value: 1 x B x F
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
    # C x B x F
    return res


# key: C x B x F
# mem: C x F
def read(key, mem):
    value = complex_mult(
        permute(key),
        mem.dimshuffle(0, 'x', 1),
        inverse_r=True, moduli_1=True)
    return value.mean(axis=0)


# key: C x B x F
# value: B x F
# mem: C x F
def write(key, value):
    coded_value = complex_mult(permute(key), value.dimshuffle('x', 0, 1))
    # C x F
    return coded_value.sum(axis=1)

if __name__ == "__main__":
    # B x F
    key = T.matrix('key')
    # B x F
    value = T.matrix('value')
    # C x F
    mem = T.matrix('mem')

    read_func = theano.function([key, mem], read(key, mem))
    write_func = theano.function([key, value], write(key, value))

    # shape: 20 x 110 x 110 x 3
    data = np.load('20_images_from_imagenet.npy')[:B]
    VALUES = data.reshape(B, F) - np.mean(data.reshape(B, F),
                                          axis=1, keepdims=True)

    phis = np.random.random((B, F / 2)) * 2 * np.pi
    KEYS = np.concatenate([np.cos(phis), np.sin(phis)], axis=1)

    MEM = write_func(KEYS, VALUES)

    all_imgs = read_func(KEYS, MEM)

    VALUES = VALUES + np.mean(data.reshape(B, F), axis=1, keepdims=True)
    VALUES = VALUES.reshape(B, 110, 110, 3)
    VALUES = np.swapaxes(VALUES, 0, 1)
    VALUES = np.reshape(VALUES, (110, 110 * B, 3))
    plt.imshow(VALUES[:, :110 * B])
    plt.show()

    all_imgs = all_imgs + np.mean(data.reshape(B, F), axis=1, keepdims=True)
    all_imgs = all_imgs.reshape(B, 110, 110, 3)
    all_imgs = np.swapaxes(all_imgs, 0, 1)
    all_imgs = np.reshape(all_imgs, (110, 110 * B, 3))
    plt.imshow(all_imgs[:, :110 * B])
    plt.show()
