from chainer.functions.activation import slstm
from chainer import link
from chainer.links.connection import linear
from chainer import variable


class StatelessSLSTM(link.Chain):

    """Stateless S-LSTM layer.

    This is a fully-connected S-LSTM layer as a chain. Unlike the
    :func:`~chainer.functions.slstm` function, which is defined as a stateless
    activation function.

    Args:
        in_size (int): Dimensionality of input vectors.
        out_size (int): Dimensionality of output vectors.

    Attributes:
        W (chainer.links.Linear): Linear layer of upward connections.

    """
    def __init__(self, in_size, out_size):
        super(StatelessSLSTM, self).__init__(
            W1=linear.Linear(out_size, 4 * out_size),
            W2=linear.Linear(out_size, 4 * out_size),
        )
        self.state_size = out_size

    def _make_input(self, c, h, W):
        lstm_in = W(h)
        if c is None:
            c = self.xp.zeros((len(h.data), self.state_size), dtype=h.dtype)
        return c, lstm_in

    def __call__(self, c1, c2, h1, h2):
        """Returns S-LSTM output.

        Args:
            c1 (~chainer.Variable): Left cell state of S-LSTM units.
            c2 (~chainer.Variable): Right cell state.
            h1 (~chainer.Variable): Left hidden state.
            h2 (~chainer.Variable): Right hidden state.

        Returns:
            ~chainer.Variable: Outputs of updated S-LSTM units.

        """
        c1, lstm_in1 = self._make_input(c1, h1, self.W1)
        c2, lstm_in2 = self._make_input(c2, h2, self.W2)
        c, h = slstm.slstm(c1, c2, lstm_in1, lstm_in2)
        return c, h
