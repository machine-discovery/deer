from typing import Tuple, Callable
from functools import partial
import jax
import jax.numpy as jnp


def scan_binop(element_i: Tuple[jnp.ndarray, jnp.ndarray],
               element_j: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # associative operator for the scan
    gti, hti = element_i
    gtj, htj = element_j
    a = gtj @ gti
    b = jnp.einsum("...ij,...j->...i", gtj, hti) + htj
    # clip = 1e9
    # a = jnp.clip(a, a_min=-clip, a_max=clip)
    # b = jnp.clip(b, a_min=-clip, a_max=clip)
    return a, b

def matmul_recursive(mats: jnp.ndarray, vecs: jnp.ndarray, y0: jnp.ndarray) -> jnp.ndarray:
    """
    Perform the matrix multiplication recursively, y[i + 1] = mats[i] @ y[i] + vec[i].

    Arguments
    ---------
    mats: jnp.ndarray
        The matrices to be multiplied, shape ``(nsamples - 1, ny, ny)``
    vecs: jnp.ndarray
        The vector to be multiplied, shape ``(nsamples - 1, ny)``
    y0: jnp.ndarray
        The initial condition, shape ``(ny,)``

    Returns
    -------
    result: jnp.ndarray
        The result of the matrix multiplication, shape ``(nsamples, ny)``, including ``y0`` at the beginning.
    """
    # shift the elements by one index
    eye = jnp.eye(mats.shape[-1], dtype=mats.dtype)[None]  # (1, ny, ny)
    first_elem = jnp.concatenate((eye, mats), axis=0)  # (nsamples, ny, ny)
    second_elem = jnp.concatenate((y0[None], vecs), axis=0)  # (nsamples, ny)

    # perform the scan
    elems = (first_elem, second_elem)
    # _, yt = jax.lax.associative_scan(scan_binop, elems)
    _, yt = associative_scan(scan_binop, elems)
    # jax.debug.print("{nan} {inf}", nan=jnp.any(jnp.isnan(yt)), inf=jnp.any(jnp.isinf(yt)))
    return yt  # (nsamples, ny)

def associative_scan(fn: Callable, elems, reverse: bool = False, axis: int = 0):
    # associative_scan from jax's source code, but change the slice_in_dim to direct indexing
    # due to strange bug in slice_in_dim (see https://github.com/google/jax/issues/21637)
    from jax._src import util, core

    if not callable(fn):
        raise TypeError("lax.associative_scan: fn argument should be callable.")
    elems_flat, tree = jax.tree_util.tree_flatten(elems)

    if reverse:
        elems_flat = [jax.lax.rev(elem, [axis]) for elem in elems_flat]

    def combine(a_flat, b_flat):
        # Lower `fn` to operate on flattened sequences of elems.
        a = jax.tree_util.tree_unflatten(tree, a_flat)
        b = jax.tree_util.tree_unflatten(tree, b_flat)
        c = fn(a, b)
        c_flat, _ = jax.tree_util.tree_flatten(c)
        return c_flat

    # Check that all inputs have a consistent leading dimension `num_elems`.
    axis = util.canonicalize_axis(axis, elems_flat[0].ndim)

    if not core.is_constant_dim(elems_flat[0].shape[axis]):
        raise NotImplementedError("associative scan over axis "
            f"of non-constant size: {elems_flat[0].shape[axis]}. You may be "
            "able to avoid this on TPU. See b/274176030.")
    num_elems = int(elems_flat[0].shape[axis])
    if not all(int(elem.shape[axis]) == num_elems for elem in elems_flat[1:]):
        raise ValueError('Array inputs to associative_scan must have the same '
                         'first dimension. (saw: {})'
                         .format([elem.shape for elem in elems_flat]))

    def get_idxs(elem, slc):
        lst = [slice(None, None, None) for _ in range(len(elem.shape))]
        lst[axis] = slc
        return tuple(lst)

    def _scan(elems):
        """Perform scan on `elems`."""

        num_elems = elems[0].shape[axis]

        if num_elems < 2:
            return elems

        # Combine adjacent pairs of elements.
        reduced_elems = combine(
            [elem[get_idxs(elem, slice(0, -1, 2))] for elem in elems],
            [elem[get_idxs(elem, slice(1, None, 2))] for elem in elems])
        # # original JAX code, suffer from the bug in slice_in_dim (see https://github.com/google/jax/issues/21637)
        # reduced_elems = combine(
        #     [jax.lax.slice_in_dim(elem, 0, -1, stride=2, axis=axis) for elem in elems],
        #     [jax.lax.slice_in_dim(elem, 1, None, stride=2, axis=axis)
        #     for elem in elems])

        # Recursively compute scan for partially reduced tensors.
        odd_elems = _scan(reduced_elems)

        if num_elems % 2 == 0:
            even_elems = combine(
                [e[get_idxs(e, slice(0, -1, None))] for e in odd_elems],
                [e[get_idxs(e, slice(2, None, 2))] for e in elems])
            # even_elems = combine(
            #     [jax.lax.slice_in_dim(e, 0, -1, axis=axis) for e in odd_elems],
            #     [jax.lax.slice_in_dim(e, 2, None, stride=2, axis=axis) for e in elems])
        else:
            even_elems = combine(
                odd_elems,
                [e[get_idxs(e, slice(2, None, 2))] for e in elems])
            # even_elems = combine(
            #     odd_elems,
            #     [jax.lax.slice_in_dim(e, 2, None, stride=2, axis=axis) for e in elems])

        # The first element of a scan is the same as the first element
        # of the original `elems`.
        even_elems = [
            jax.lax.concatenate([elem[get_idxs(elem, slice(0, 1, None))], result],
                                dimension=axis)
            # jax.lax.concatenate([jax.lax.slice_in_dim(elem, 0, 1, axis=axis), result],
            #                     dimension=axis)
            for (elem, result) in zip(elems, even_elems)]
        return list(util.safe_map(partial(_interleave, axis=axis), even_elems, odd_elems))

    scans = _scan(elems_flat)

    if reverse:
        scans = [jax.lax.rev(scanned, [axis]) for scanned in scans]

    return jax.tree_util.tree_unflatten(tree, scans)

def _interleave(a, b, axis):
    from jax._src.lax import lax
    """Given two Tensors of static shape, interleave them along the first axis."""
    assert a.shape[axis] == b.shape[axis] or a.shape[axis] == b.shape[axis] + 1
    a_pad = [(0, 0, 0)] * a.ndim
    b_pad = [(0, 0, 0)] * b.ndim
    a_pad[axis] = (0, 1 if a.shape[axis] == b.shape[axis] else 0, 1)
    b_pad[axis] = (1, 0 if a.shape[axis] == b.shape[axis] else 1, 1)
    op = jax.lax.bitwise_or if a.dtype == jnp.bool_ else jax.lax.add
    return op(jax.lax.pad(a, lax._const(a, 0), a_pad),
              jax.lax.pad(b, lax._const(b, 0), b_pad))
