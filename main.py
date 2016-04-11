import theano
import numpy as np
from theano import tensor
from blocks.initialization import IsotropicGaussian, Constant
from blocks.bricks import Linear, Tanh
from bricks import AssociativeLSTM
floatX = theano.config.floatX


n_epochs = 90
x_dim = 10
h_dim = 20
o_dim = 12

print 'Building model ...'
# T x B x F
x = tensor.tensor3('x', dtype=floatX)

x_to_h = Linear(name='x_to_h',
                input_dim=x_dim,
                output_dim=4.5 * h_dim)
x_transform = x_to_h.apply(x)
lstm = AssociativeLSTM(activation=Tanh(),
                       dim=h_dim, name="lstm")
h, c = lstm.apply(x_transform)
h_to_o = Linear(name='h_to_o',
                input_dim=h_dim,
                output_dim=o_dim)
o = h_to_o.apply(h)

for brick in (lstm, x_to_h, h_to_o):
    brick.weights_init = IsotropicGaussian(0.01)
    brick.biases_init = Constant(0)
    brick.initialize()

f = theano.function([x], o)
X = np.random.random((15, 100, 10))
print f(X).shape
