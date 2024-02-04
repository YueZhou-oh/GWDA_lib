import numpy as np


def dot(a, b):
    """Dot product on the last axis.

    Args:
        a (array-like): input array
        b (array-like): input array
    """
    return np.einsum("...j, ...j", a, b)


def norm(a):
    """Norm on the last axis.

    Args:
        a (array-like): input array
    """
    return np.linalg.norm(a, axis=-1)


def arrayindex(a, x):
    """Return the array indices for ``x`` in ``a``.

    >>> a = [1, 3, 5, 7]
    >>> indices = arrayindex([3, 7, 1], a)
    [1, 3, 0]

    Args:
        x (array-like): elements to search for
        a (array-like): array in which elements are searched for

    Returns:
        :obj:`ndarray`: List of array indices

    Raises:
        ValueError: if not all elements of ``x`` cannot be found in ``a``.
    """
    if not np.all(np.isin(x, a)):
        raise ValueError("cannot find all items")
    return np.searchsorted(a, x)


def atleast_2d(a):
    """View inputs as arrays with at least two dimensions.

    Contrary to numpy's function, we here add the missing dimension
    on the last axis if needed.

    >>> np.atleast_2d(3.0)
    array([[3.]])
    >>> x = np.arange(3.0)
    >>> np.atleast_2d(x)
    array([[0., 1., 2.]])
    >>> np.atleast_2d(x).base is x
    True
    >>> np.atleast_2d(1, [1, 2], [[1, 2]])
    [array([[1]]), array([[1, 2]]), array([[1, 2]])]

    Args:
        a (array-like): input array

    Returns:
        :obj:`ndarray`: An array with ``ndim >= 2``.

        Copies are avoided where possible, and views with two or more
        dimensions are returned.
    """
    a = np.asanyarray(a)
    if a.ndim == 0:
        return a.reshape(1, 1)
    if a.ndim == 1:
        return a[:, np.newaxis]
    return a


@np.vectorize
def emitter(link):
    """Return emitter spacecraft index from link index.

    >>> emitter(12)
    array(2)
    >>> emitter([12, 31, 21])
    array([2, 1, 1])
    >>> emitter(np.array([23, 12]))
    array([3, 2])

    Args:
        link (int): link index

    Returns:
        int: Emitter spacecraft index.
    """
    if link not in [12, 23, 31, 13, 32, 21]:
        raise ValueError(f"invalid link index '{link}'")
    return int(str(link)[1])


@np.vectorize
def receiver(link):
    """Return receiver spacecraft index from a link index.

    >>> receiver(12)
    array(1)
    >>> receiver([12, 31, 21])
    array([1, 3, 2])
    >>> receiver(np.array([23, 12]))
    array([2, 1])

    Args:
        link (int): link index

    Returns:
        int: Emitter spacecraft index.
    """
    if link not in [12, 23, 31, 13, 32, 21]:
        raise ValueError(f"invalid link index '{link}'")
    return int(str(link)[0])
