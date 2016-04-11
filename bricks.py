from blocks.bricks import Initializable, Tanh, Logistic
from blocks.bricks.base import application, lazy
from blocks.roles import add_role, WEIGHT, INITIAL_STATE
from blocks.utils import shared_floatx_nans, shared_floatx_zeros
from blocks.bricks.recurrent import BaseRecurrent, recurrent
import theano.tensor as tensor
from holographic_memory import complex_mult


class AssociativeLSTM(BaseRecurrent, Initializable):
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation=None, gate_activation=None, **kwargs):
        self.dim = dim

        if not activation:
            activation = Tanh()
        if not gate_activation:
            gate_activation = Logistic()
        self.activation = activation
        self.gate_activation = gate_activation

        children = ([self.activation, self.gate_activation] +
                    kwargs.get('children', []))
        super(AssociativeLSTM, self).__init__(children=children, **kwargs)

    def get_dim(self, name):
        if name == 'inputs':
            return self.dim * 4.5
        if name in ['states', 'cells']:
            return self.dim
        if name == 'mask':
            return 0
        return super(AssociativeLSTM, self).get_dim(name)

    def _allocate(self):
        self.W_state = shared_floatx_nans((self.dim, 4.5 * self.dim),
                                          name='W_state')
        # The underscore is required to prevent collision with
        # the `initial_state` application method
        self.initial_state_ = shared_floatx_zeros((self.dim,),
                                                  name="initial_state")
        self.initial_cells = shared_floatx_zeros((self.dim,),
                                                 name="initial_cells")
        add_role(self.W_state, WEIGHT)
        add_role(self.initial_state_, INITIAL_STATE)
        add_role(self.initial_cells, INITIAL_STATE)

        self.parameters = [
            self.W_state, self.initial_state_, self.initial_cells]

    def _initialize(self):
        self.weights_init.initialize(self.parameters[0], self.rng)

    # The activation function that bound values between 0 and 1
    def bound(self, input_):
        sq = input_ ** 2
        d = tensor.maximum(1, tensor.sqrt(
            sq[:, :self.dim / 2] + sq[:, self.dim / 2:]))
        d = tensor.concatenate([d, d], axis=1)
        return input_ / d

    @recurrent(sequences=['inputs', 'mask'], states=['states', 'cells'],
               contexts=[], outputs=['states', 'cells'])
    def apply(self, inputs, states, cells, mask=None):
        def slice_(x, no):
            # Gates dimension is dim/2.
            if no in [0, 1, 2]:
                return x[:, no * self.dim / 2: (no + 1) * self.dim / 2]
            # Keys and u dimension is dim.
            elif no in [3, 4, 5]:
                return x[:, int((no - 1.5) * self.dim):
                         int((no - 0.5) * self.dim)]

        activation = tensor.dot(states, self.W_state) + inputs

        in_gate = self.gate_activation.apply(slice_(activation, 0))
        in_gate = tensor.concatenate([in_gate, in_gate], axis=1)
        forget_gate = self.gate_activation.apply(slice_(activation, 1))
        forget_gate = tensor.concatenate([forget_gate, forget_gate], axis=1)
        out_gate = self.gate_activation.apply(slice_(activation, 2))
        out_gate = tensor.concatenate([out_gate, out_gate], axis=1)

        in_key = self.bound(slice_(activation, 3))
        out_key = self.bound(slice_(activation, 4))
        u = self.bound(slice_(activation, 5))

        next_cells = (forget_gate * cells +
                      complex_mult(in_key, in_gate * u))

        next_states = out_gate * self.bound(complex_mult(out_key, next_cells))

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
            next_cells = (mask[:, None] * next_cells +
                          (1 - mask[:, None]) * cells)

        return next_states, next_cells

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return [tensor.repeat(self.initial_state_[None, :], batch_size, 0),
                tensor.repeat(self.initial_cells[None, :], batch_size, 0)]
