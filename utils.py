import theano
import numpy as np
from blocks.initialization import NdarrayInitialization
import logging
from blocks.extensions import SimpleExtension
logger = logging.getLogger('main.utils')


class SaveLog(SimpleExtension):
    def __init__(self, show=None, **kwargs):
        super(SaveLog, self).__init__(**kwargs)
        self.add_condition(('before_training',), self.do)
        self.add_condition(('after_training',), self.do)
        self.add_condition(('on_interrupt',), self.do)

    def do(self, which_callback, *args):
        epoch = self.main_loop.status['iterations_done']
        current_row = self.main_loop.log.current_row
        logger.info("\niterations_done:%d" % epoch)
        for element in current_row:
            logger.info(str(element) + ":%f" % current_row[element])


class Glorot(NdarrayInitialization):
    def generate(self, rng, shape):
        if len(shape) == 2:
            input_size, output_size = shape

            # if it is lstm's concatenated weight
            if (input_size * 4.5 == output_size):
                output_size = output_size / 4.5
            elif (input_size * 4 == output_size):
                output_size = output_size / 4

            high = np.sqrt(6) / np.sqrt(input_size + output_size)
            m = rng.uniform(-high, high, size=shape)
        return m.astype(theano.config.floatX)
