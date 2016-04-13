import theano
import os
import numpy as np
from theano import tensor
from blocks.initialization import Constant
from blocks.bricks import Linear, Tanh
from bricks import AssociativeLSTM
from fuel.datasets import IterableDataset
from fuel.streams import DataStream
from blocks.model import Model
from blocks.bricks.cost import SquaredError
from blocks.algorithms import (GradientDescent, Scale,
                               StepClipping, CompositeRule,
                               Adam)
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.extensions import Printing
from blocks.graph import ComputationGraph
import logging
from utils import SaveLog, Glorot
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
floatX = theano.config.floatX


def get_episodic_copy_data(time_steps, n_data, n_sequence, batch_size):
    seq = np.random.randint(1, high=9, size=(n_data, n_sequence))
    zeros1 = np.zeros((n_data, time_steps - 1))
    zeros2 = np.zeros((n_data, time_steps))
    marker = 9 * np.ones((n_data, 1))
    zeros3 = np.zeros((n_data, n_sequence))

    x = np.concatenate((seq, zeros1, marker, zeros3), axis=1).astype('int32')
    y = np.concatenate((zeros3, zeros2, seq), axis=1).astype('int32')

    x = x.reshape(n_data / batch_size, batch_size, 1, -1)
    x = np.swapaxes(x, 2, 3)
    y = y.reshape(n_data / batch_size, batch_size, 1, -1)
    y = np.swapaxes(y, 2, 3)

    return x, y

batch_size = 2
num_copies = 1
x_dim = 1
h_dim = 128
o_dim = 1
save_path = 'test_path'

print 'Building model ...'
# T x B x F
x = tensor.tensor3('x', dtype=floatX)
# T x B x F
y = tensor.tensor3('y', dtype=floatX)

x_to_h = Linear(name='x_to_h',
                input_dim=x_dim,
                output_dim=4.5 * h_dim)
x_transform = x_to_h.apply(x)
lstm = AssociativeLSTM(activation=Tanh(),
                       dim=h_dim,
                       num_copies=num_copies,
                       name="lstm")
h, c = lstm.apply(x_transform)
h_to_o = Linear(name='h_to_o',
                input_dim=h_dim,
                output_dim=o_dim)
o = h_to_o.apply(h)

for brick in (lstm, x_to_h, h_to_o):
    brick.weights_init = Glorot()
    brick.biases_init = Constant(0)
    brick.initialize()

cost = SquaredError().apply(y[-10:], o[-10:])
cost.name = 'SquaredError'

print 'Bulding training process...'
shapes = []
for param in ComputationGraph(cost).parameters:
    # shapes.append((param.name, param.eval().shape))
    shapes.append(np.prod(list(param.eval().shape)))
print "Total number of parameters: " + str(np.sum(shapes))

if not os.path.exists(save_path):
    os.makedirs(save_path)
log_path = save_path + '/log.txt'
fh = logging.FileHandler(filename=log_path)
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

algorithm = GradientDescent(cost=cost,
                            parameters=ComputationGraph(cost).parameters,
                            step_rule=CompositeRule([StepClipping(10.0),
                                                     Adam(1e-3)]))  # 3e-4
monitor_cost = TrainingDataMonitoring([cost],
                                      prefix='train',
                                      after_epoch=False,
                                      before_training=True,
                                      every_n_batches=1000)

data = get_episodic_copy_data(100, int(1e6), 10, batch_size)
dataset = IterableDataset({'x': data[0] / 10.0,
                           'y': data[1] / 10.0})
stream = DataStream(dataset)

model = Model(cost)
main_loop = MainLoop(data_stream=stream, algorithm=algorithm,
                     extensions=[monitor_cost,
                                 Printing(after_epoch=False,
                                          every_n_batches=1000),
                                 SaveLog(every_n_batches=1000)],
                     model=model)

print 'Starting training ...'
main_loop.run()
