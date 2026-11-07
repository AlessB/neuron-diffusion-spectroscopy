"""
A tiny numpy stand-in for the parts of torch that nds.activations.collect_neuron_moments uses.
For testing the collection loop's bookkeeping without PyTorch. Not a real torch.
"""
import contextlib
import numpy as np

long = np.int64
float32 = np.float32
float64 = np.float64


class _MaxResult:
    def __init__(self, values):
        self.values = values


class Tensor:
    def __init__(self, array):
        self.a = np.asarray(array)

    # conversions
    def float(self): return Tensor(self.a.astype(np.float32))
    def double(self): return Tensor(self.a.astype(np.float64))
    def detach(self): return Tensor(self.a.copy())
    def cpu(self): return self
    def numpy(self): return self.a
    def numel(self): return int(self.a.size)
    def __float__(self): return float(self.a)
    def __int__(self): return int(self.a)

    @property
    def shape(self): return self.a.shape
    @property
    def T(self): return Tensor(self.a.T)

    # reductions
    def sum(self, dim=None): return Tensor(self.a.sum(axis=dim))
    def mean(self, dim=None): return Tensor(self.a.mean(axis=dim))
    def max(self, dim=None): return _MaxResult(Tensor(self.a.max(axis=dim)))
    def abs(self): return Tensor(np.abs(self.a))
    def norm(self, dim=None, keepdim=False): return Tensor(np.linalg.norm(self.a, axis=dim, keepdims=keepdim))
    def clamp_min(self, value): return Tensor(np.maximum(self.a, value))
    def reshape(self, *shape): return Tensor(self.a.reshape(*shape))

    # indexing and arithmetic
    def __getitem__(self, key):
        if isinstance(key, Tensor):
            key = key.a
        return Tensor(self.a[key])
    def _v(self, other): return other.a if isinstance(other, Tensor) else other
    def __sub__(self, o): return Tensor(self.a - self._v(o))
    def __add__(self, o): return Tensor(self.a + self._v(o))
    def __mul__(self, o): return Tensor(self.a * self._v(o))
    __rmul__ = __mul__
    def __truediv__(self, o): return Tensor(self.a / self._v(o))
    def __gt__(self, o): return Tensor(self.a > self._v(o))
    def __matmul__(self, o): return Tensor(self.a @ self._v(o))


def as_tensor(data, dtype=None, device=None):
    return Tensor(np.asarray(data, dtype=dtype))


@contextlib.contextmanager
def no_grad():
    yield
