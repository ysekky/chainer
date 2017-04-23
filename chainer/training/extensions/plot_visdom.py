import numpy
import six
try:
    import visdom
    available = True
except ImportError:
    available = False


from chainer.training.extensions import plot_base


class PlotVisdom(plot_base.PlotBase):

    def __init__(self, y_keys, x_key='iteration', trigger=(1, 'epoch'),
                 vis=None, opts=None):
        if not available:
            raise RuntimeError('visdom is not available')

        super(PlotVisdom, self).__init__(y_keys, x_key, trigger)
        self._visdom = vis if vis else visdom.Visdom()
        self._opts = opts if opts else {}

    def plot(self, trainer, summary, x, ys):
        x = numpy.array([x])

        ys_data = [y for _, y in six.iteritems(ys) if y is not None]
        keys = [key for key, y in six.iteritems(ys) if y is not None]

        if hasattr(self, '_win'):
            for key, y in six.moves.zip(keys, ys_data):
                y = numpy.array([y])
                self._visdom.updateTrace(X=x, Y=y, win=self._win, name=key)
        elif len(ys) > 0:
            # visdom does not accept empty list.
            y = numpy.array([ys_data])
            opts = self._opts.copy()
            opts['legend'] = keys
            self._win = self._visdom.line(X=x, Y=y, opts=opts)
