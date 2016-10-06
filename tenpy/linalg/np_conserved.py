r"""A module to handle charge conservation in tensor networks.

A detailed introduction (including notations) can be found in :doc:`../IntroNpc`.

This module `np_conserved` implements an class :class:`Array`
designed to make use of charge conservation in tensor networks.
The idea is that the `Array` class is used in a fashion very similar to
the `numpy.ndarray`, e.g you can call the functions :func:`tensordot` or :func:`svd`
(of this module) on them.
The structure of the algorithms (as DMRG) is thus the same as with basic numpy ndarrays.

Internally, an :class:`Array` saves charge meta data to keep track of blocks which are nonzero.
All possible operations (e.g. tensordot, svd, ...) on such arrays preserve the total charge
structure. In addition, these operations make use of the charges to figure out which of the blocks
it hase to use/combine - this is the basis for the speed-up.


See also
--------
tenpy.linalg.charges : Implementation of :class:`~tenpy.linalg.charge.ChargeInfo`
    and :class:`~tenpy.linalg.charge.LegCharge` with additional documentation.


.. todo ::
   usage and examples, Routine listing
   update ``from charges import``
"""
# Examples
# --------
# >>> import numpy as np
# >>> import tenpy.linalg.np_conserved as npc
# >>> Sz = np.array([[0., 1.], [1., 0.]])
# >>> Sz_c = npc.Array.from_ndarray_trivial(Sz)  # convert to npc array with trivial charge
# >>> Sz_c
# <npc.Array shape=(2, 2)>
# >>> sx = npc.ndarray.from_ndarray([[0., 1.], [1., 0.]])  # trivial charge conservation
# >>> b = npc.ndarray.from_ndarray([[0., 1.], [1., 0.]])  # trivial charge conservation
# >>>
# >>> print a[0, -1]
# >>> c = npc.tensordot(a, b, axes=([1], [0]))

from __future__ import division

import numpy as np
import copy as copy_
import warnings
import itertools

# import public API from charges
from .charges import (QDTYPE, ChargeInfo, LegCharge, LegPipe,
                      reverse_sort_perm)
from . import charges   # for private functions

"""A cutoff to ignore machine precision rounding errors when determining charges"""
QCUTOFF = np.finfo(np.float64).eps * 10


class Array(object):
    r"""A multidimensional array (=tensor) for using charge conservation.

    An `Array` represents a multi-dimensional tensor,
    together with the charge structure of its legs (for abelian charges).
    Further information can be found in :doc:`../IntroNpc`.

    ``Array(...)`` does not insert any data and thus yields an Array 'full' of zeros.
    New array can be created with one of :meth:`from_ndarray_trivial`,
    :meth:`from_ndarray`, or :meth:`from_npfunc`.
    as :meth:`__init__` does not initialize any data.

    Parameters
    ----------
    chargeinfo : :class:`~tenpy.linalg.charges.ChargeInfo`
        the nature of the charge, used as self.chinfo.
    legs : list of :class:`~tenpy.linalg.charges.LegCharge`
        the leg charges for each of the legs.
    dtype : type or string
        the data type of the array entries. Defaults to np.float64.


    Attributes
    ----------
    rank
    shape : tuple(int)
        the number of indices for each of the legs
    dtype : np.dtype
        the data type of the entries
    chinfo : :class:`~tenpy.linalg.charges.ChargeInfo`
        the nature of the charge
    qtotal : charge values
        the total charge of the tensor.
    legs : list of :class:`~tenpy.linalg.charges.LegCharge`
        the leg charges for each of the legs.
    labels : dict (string -> int)
        labels for the different legs
    _data : list of arrays
        the actual entries of the tensor
    _qdata : 2D array (len(_data), rank)
        for each of the _data entries the qind of the different legs.
    _qdata_sorted : Bool
        whether self._qdata is lexsorted. Defaults to `True`,
        but *must* be set to `False` by algorithms changing _qdata.

    .. todo ::

        test everything
    """
    def __init__(self, chargeinfo, legcharges, dtype=np.float64, qtotal=None):
        """see help(self)"""
        self.chinfo = chargeinfo
        self.legs = list(legcharges)
        self._set_shape()
        self.dtype = np.dtype(dtype)
        self.qtotal = self.chinfo.make_valid(qtotal)
        self.labels = {}
        self._data = []
        self._qdata = np.empty((0, self.rank), QDTYPE)
        self._qdata_sorted = True
        self.test_sanity()

    def copy(self, deep=False):
        """Return a (deep or shallow) copy of self.

        **Both** deep and shallow copies will share ``chinfo`` and the `LegCharges` in ``legs``.
        In contrast to a deep copy, the shallow copy will also share the tensor entries.
        """
        if deep:
            cp = copy_.deepcopy(self)
        else:
            cp = copy_.copy(self)
            # some things should be copied even for shallow copies
            cp._set_shape()
            cp.qtotal = cp.qtotal.copy()
        # even deep copies can share chargeinfo and legs, but they get
        cp.chinfo = self.chinfo
        cp.legs = list(self.legs)
        return cp

    @classmethod
    def from_ndarray_trivial(cls, data_flat, dtype=np.float64):
        """convert a flat numpy ndarray to an Array with trivial charge conservation.

        Parameters
        ----------
        data_flat : array_like
            the data to be converted to a Array
        dtype : type | string
            the data type of the array entries. Defaults to ``np.float64``.

        Returns
        -------
        res : :class:`Array`
            an Array with data of data_flat
        """
        data_flat = np.array(data_flat, dtype)
        chinfo = ChargeInfo()
        legs = [LegCharge.from_trivial(s, chinfo) for s in data_flat.shape]
        res = cls(chinfo, legs, dtype)
        res._data = [data_flat]
        res._qdata = np.zeros((1, res.rank), QDTYPE)
        res._qdata_sorted = True
        res.test_sanity()
        return res

    @classmethod
    def from_ndarray(cls, data_flat, chargeinfo, legcharges, dtype=np.float64, qtotal=None,
                     cutoff=0.):
        """convert a flat (numpy) ndarray to an Array.

        Parameters
        ----------
        data_flat : array_like
            the flat ndarray which should be converted to a npc `Array`.
            The shape has to be compatible with legcharges.
        chargeinfo : ChargeInfo
            the nature of the charge
        legcharges : list of LegCharge
            a LegCharge for each of the legs.
        dtype : type | string
            the data type of the array entries. Defaults to np.float64.
        qtotal : None | charges
            the total charge of the new array.
        cutoff : float
            A cutoff to exclude rounding errors of machine precision. Defaults to QCUTOFF.

        Returns
        -------
        res : :class:`Array`
            an Array with data of `data_flat`.

        See also
        --------
        detect_ndarray_qtotal : used to detect the total charge of the flat array.
        """
        if cutoff is None:
            cutoff = QCUTOFF
        res = cls(chargeinfo, legcharges, dtype, qtotal)  # without any data
        if res.shape != data_flat.shape:
            raise ValueError("Incompatible shapes: legcharges {0!s} vs flat {1!s} ".format(
                res.shape, data_flat.shape))
        if qtotal is None:
            res.qtotal = qtotal = res.detect_ndarray_qtotal(data_flat, cutoff)
        data = []
        qdata = []
        for qindices in res._iter_all_blocks():
            sl = res._get_block_slices(qindices)
            if np.all(res._get_block_charge(qindices) == qtotal):
                data.append(np.array(data_flat[sl], dtype=res.dtype))   # copy data
                qdata.append(qindices)
            elif np.any(np.abs(data_flat[sl]) > cutoff):
                warnings.warn("flat array has non-zero entries in blocks incompatible with charge")
        res._data = data
        if len(qdata) == 0:
            res._qdata = np.empty((0, res.rank), QDTYPE)
        else:
            res._qdata = np.array(qdata, dtype=QDTYPE)
        res._qdata_sorted = True
        res.test_sanity()
        return res

    def zeros_like(self):
        """return a shallow copy of self with only zeros as entries, containing no `_data`"""
        res = self.copy(deep=False)
        res._data = []
        res._qdata = np.empty((0, res.rank), QDTYPE)
        res._qdata_sorted = True
        return res

    @property
    def rank(self):
        """the number of legs"""
        return len(self.shape)

    def test_sanity(self):
        """Sanity check. Raises ValueErrors, if something is wrong."""
        if self.shape != tuple([lc.ind_len for lc in self.legs]):
            raise ValueError("shape mismatch with LegCharges\n self.shape={0!s} != {1!s}".format(
                self.shape, tuple([lc.ind_len for lc in self.legs])))
        if any([self.dtype != d.dtype for d in self._data]):
            raise ValueError("wrong dtype: {0!s} vs\n {1!s}".format(
                self.dtype, [self.dtype != d.dtype for d in self._data]))
        for l in self.legs:
            l.test_sanity()
            if l.chinfo != self.chinfo:
                raise ValueError("leg has different ChargeInfo:\n{0!s}\n vs {1!s}".format(
                    l.chinfo, self.chinfo))
        if self._qdata.shape != (len(self._data), self.rank):
            raise ValueError("_qdata shape wrong")
        if np.any(self._qdata < 0) or np.any(self._qdata >= [l.block_number for l in self.legs]):
            raise ValueError("invalid qind in _qdata")
        if self._qdata_sorted:
            perm = np.lexsort(self._qdata.T)
            if np.any(perm != np.arange(len(perm))):
                raise ValueError("_qdata_sorted == True, but _qdata is not sorted")

    def to_ndarray(self):
        """convert self to a dense numpy ndarray."""
        res = np.zeros(self.shape, self.dtype)
        for block, slices, _, _ in self:  # that's elegant! :)
            res[slices] = block
        return res

    def astype(self, dtype):
        """Return (deep) copy with new dtype, upcasting all blocks in ``_data``.

        Parameters
        ----------
        dtype : convertible to a np.dtype
            the new data type.
            If None, deduce the new dtype as common type of ``self._data``.

        Returns
        -------
        copy : :class:`Array`
            deep copy of self with new dtype
        """
        cp = self.copy(deep=False)  # manual deep copy: don't copy every block twice
        cp._qdata = cp._qdata.copy()
        if dtype is None:
            dtype = np.common_dtype(*self._data)
        cp.dtype = np.dtype(dtype)
        cp._data = [d.astype(self.dtype, copy=True) for d in self._data]
        return cp

    def get_leg_index(self, label):
        """translate a leg-index or leg-label to a leg-index.

        Parameters
        ----------
        label : int | string
            eather the leg-index directly or a label (string) set before.

        Returns
        -------
        leg_index : int
            the index of the label

        See also
        --------
        get_leg_indices : calls get_leg_index for a list of labels
        set_leg_labels : set the labels of different legs.
        """
        return self.labels.get(label, label)

    def get_leg_indices(self, labels):
        """Translate a list of leg-indices or leg-labels to leg indices.

        Parameters
        ----------
        labels : iterable of string/int
            The leg-labels (or directly indices) to be translated in leg-indices

        Returns
        -------
        leg_indices : list of int
            the translated labels.

        See also
        --------
        get_leg_index : used to translate each of the single entries.
        set_leg_labels : set the labels of different legs.
        """
        return [self.get_leg_index(l) for l in labels]

    def set_leg_labels(self, labels):
        """Return labels for the legs.

        Introduction to leg labeling can be found in :doc:`../IntroNpc`.

        Parameters
        ----------
        labels : iterable (strings | None), len=self.rank
            One label for each of the legs.
            An entry can be None for an anonymous leg.

        See also
        --------
        get_leg: translate the labels to indices
        get_legs: calls get_legs for an iterable of labels
        """
        if len(labels) != self.rank:
            raise ValueError("Need one leg label for each of the legs.")
        self.labels = {}
        for i, l in enumerate(labels):
            if l is not None:
                self.labels[l] = i

    def get_leg_labels(self):
        """Return tuple of the leg labels, with `None` for anonymous legs."""
        lb = [None] * self.rank
        for k, v in self.labels.iteritems():
            lb[v] = k
        return tuple(lb)

    def detect_ndarray_qtotal(self, flat_array, cutoff=None):
        """ Returns the total charge of first non-zero sector found in `a`.

        Charge information is taken from self.
        If you have only the charge data, create an empty Array(chinf, legcharges).

        Parameters
        ----------
        flat_array : array
            the flat numpy array from which you want to detect the charges
        chinfo : ChargeInfo
            the nature of the charge
        legcharges : list of LegCharge
            for each leg the LegCharge
        cutoff : float
            defaults to QCUTOFF

        Returns
        -------
        qtotal : charge
            the total charge fo the first non-zero (i.e. > cutoff) charge block
        """
        if cutoff is None:
            cutoff = QCUTOFF
        for qindices in self._iter_all_blocks():
            sl = self._get_block_slices(qindices)
            if np.any(np.abs(flat_array[sl]) > cutoff):
                return self._get_block_charge(qindices)
        warnings.warn("can't detect total charge: no entry larger than cutoff. Return 0 charge.")
        return self.chinfo.make_valid()

    def gauge_total_charge(self, leg, newqtotal=None):
        """changes the total charge of an Array `A` inplace by adjusting the charge on a certain leg.

        The total charge is given by finding a nonzero entry [i1, i2, ...] and calculating::

            qtotal = sum([l.qconj[i] * l.qind[il] for il in zip([i1,i2,...], self.legs)])

        Thus, the total charge can be changed by redefining the leg charge of a given leg.
        This is exaclty what this function does.

        Parameters
        ----------
        leg : int or string
            the new leg (index or label), for which the charge is changed
        newqtotal : charge values, defaults to 0
            the new total charge
        """
        leg = self.get_leg_index(leg)
        newqtotal = self.chinfo.make_valid(newqtotal)  # converts to array, default zero
        chdiff = newqtotal - self.qtotal
        newleg = copy_.copy(self.legs[leg])  # shallow copy of the LegCharge
        newleg.qind = newleg.qind.copy()
        newleg.qind[:, 2:] = self.chinfo.make_valid(newleg.qind[:, 2:] + newleg.qconj * chdiff)
        self.legs[leg] = newleg
        self.qtotal = newqtotal

    def sort_legcharge(self, sort=True, bunch=True):
        """Return a copy with one ore all legs sorted by charges.

        Sort/bunch one or multiple of the LegCharges.
        Note that legs which are sorted and bunched
        are guaranteed to be completely blocked by charge.

        Parameters
        ----------
        sort : True | False | list of {True, False, perm}
            A single bool holds for all legs, default=True.
            Else, `sort` should contain one entry for each leg, with a bool for sort/don't sort,
            or a 1D array perm for a given permuation to apply to a leg.
        bunch : True | False | list of {True, False}
            A single bool holds for all legs, default=True.
            whether or not to bunch at each leg, i.e. combine contiguous blocks with equal charges

        Returns
        -------
        perm : tuple of 1D arrays
            the permutation applied to each of the legs.
            cp.to_ndarray() = self.to_ndarray(perm)
        result : Array
            a shallow copy of self, with legs sorted/bunched
        """
        if sort is False or sort is True:  # ``sort in [False, True]`` doesn't work...
            sort = [sort]*self.rank
        if bunch is False or bunch is True:
            bunch = [bunch]*self.rank
        if not len(sort) == len(bunch) == self.rank:
            raise ValueError("Wrong len for bunch or sort")
        cp = self.copy(deep=False)
        cp._qdata = cp._qdata.copy(self)
        for li in xrange(self.rank):
            if sort[li] is not False:
                if sort[li] is True:
                    p_flat, p_qind, newleg = cp.legs[li].sort(bunch=False)
                    sort[li] = p_flat
                    cp.legs[li] = newleg
                else:
                    # TODO: catch exception when something like self.permute(axis=...)
                    # is implemented for general permutations
                    # try:
                    p_qind = charges._perm_qind_from_perm_flat(sort[li], self.legs[li])
                    # except ValueError:
                    #     cp = cp.permute(sort[li], axes=[li]) # implement...
                    #     continue
                # entry ``b`` of of cp._qdata refers to old ``self.legs[li][b]``.
                # since new ``cp.legs[li][i] == self.legs[li][p_qind[i]]``,
                # we have new ``cp.legs[li][reverse_sort_perm(p_qind)[b]]``
                p_qind = reverse_sort_perm(p_qind)
                cp._qdata[:, li] = [p_qind[i] for i in cp._qdata[:, li]]
                cp._qdata_sorted = False
            else:
                sort[li] = np.arange(cp.shape[li])
        if any(bunch):
            cp = cp._bunch(bunch)  # bunch does not permute...
        return tuple(sort), cp

    def sort_qdata(self):
        """(lex)sort ``self._qdata``. In place.

        Lexsort ``self._qdata`` and ``self._data`` and set ``self._qdata_sorted = True``.
        """
        if self._qdata_sorted:
            return
        if len(self._qdata) < 2:
            self._qdata_sorted = True
            return
        perm = np.lexsort(self._qdata.T)
        self._qdata = self._qdata[perm, :]
        self._data = [self._data[p] for p in perm]
        self._qdata_sorted = True

    def __repr__(self):
        return "<npc.array shape={0!s} charge={1!s} labels={2!s}>".format(
            self.shape, self.chinfo, self.get_leg_labels())

    def __str__(self):
        res = "\n".join([repr(self)[:-1], str(self.to_ndarray()), ">"])
        return res

    def __iter__(self):
        """Allow to iterate over the non-zero blocks, giving all `_data`.

        Yields
        ------
        block : ndarray
            the actual entries of a charge block
        blockslices : tuple of slices
            a slice giving the range of the block in the original tensor for each of the legs
        charges : list of charges
            the charge value(s) for each of the legs
        qdat : ndarray
            the qind for each of the legs
        """
        for block, qdat in itertools.izip(self._data, self._qdata):
            qind = [l.qind[qi] for (qi, l) in itertools.izip(qdat, self.legs)]
            blockslices = tuple([slice(qi[0], qi[1]) for qi in qind])
            qs = [qi[2:] for qi in qind]
            yield block, blockslices, qs, qdat

    # private functions ========================================================

    def _set_shape(self):
        """deduce self.shape from self.legs"""
        self.shape = tuple([lc.ind_len for lc in self.legs])

    def _iter_all_blocks(self):
        """generator to iterate over all combinations of qindices in lexiographic order.

        Yields
        ------
        qindices : tuple of int
            a qindex for each of the legs
        """
        for block_inds in itertools.product(*[xrange(l.block_number)
                                              for l in reversed(self.legs)]):
            # loop over all charge sectors in lex order (last leg most siginificant)
            yield tuple(block_inds[::-1])   # back to legs in correct order

    def _get_block_charge(self, qindices):
        """returns the charge of a block selected by `qindices`

        The charge of a single block is defined as ::

            qtotal = sum_{legs l} legs[l].qind[qindices[l], 2:] * legs[l].qconj() modulo qmod
        """
        q = np.sum([l.qind[qi, 2:]*l.qconj for l, qi in itertools.izip(self.legs, qindices)],
                   axis=0)
        return self.chinfo.make_valid(q)

    def _get_block_slices(self, qindices):
        """returns tuple of slices for a block selected by `qindices`"""
        return tuple([slice(l.qind[j, 0], l.qind[j, 1])
                      for l, j in itertools.izip(self.legs, qindices)])

    def _bunch(self, bunch_legs):
        """Return copy and bunch the qind for one or multiple legs

        Parameters
        ----------
        bunch : list of {True, False}
            one entry for each leg, whether the leg should be bunched.

        See also
        --------
        sort_legcharge: public API calling this function.
        """
        cp = self.copy(deep=False)
        # lists for each leg:
        new_to_old_idx = [None]*cp.rank     # the `idx` returned by cp.legs[li].bunch()
        map_qindex = [None]*cp.rank         # array mapping old qindex to new qindex, such that
        # new_leg.qind[m_qindex[i]] == old_leg.qind[i]  # (except the second column entry)
        bunch_qindex = [None]*cp.rank       # bool array wheter the *new* qind was bunched
        for li, bunch in enumerate(bunch_legs):
            idx, new_leg = cp.legs[li].bunch()
            cp.legs[li] = new_leg
            new_to_old_idx[li] = idx
            # generate entries in map_qindex and bunch_qdindex
            idx = np.append(idx, [self.shape[li]])
            m_qindex = []
            bunch_qindex[li] = b_qindex = np.empty(idx.shape, dtype=np.bool_)
            for inew in xrange(len(idx)-1):
                old_blocks = idx[inew+1] - idx[inew]
                m_qindex.append([inew]*old_blocks)
                b_qindex[inew] = (old_blocks > 1)
            map_qindex[li] = np.concatenate(m_qindex, axis=0)

        # now map _data and _qdata
        bunched_blocks = {}     # new qindices -> index in new _data
        new_data = []
        new_qdata = []
        for old_block, old_qindices in itertools.izip(self._data, self._qdata):
            new_qindices = tuple([m[qi] for m, qi in itertools.izip(map_qindex, old_qindices)])
            bunch = any([b[qi] for b, qi in itertools.izip(bunch_qindex, new_qindices)])
            if bunch:
                if new_qindices not in bunched_blocks:
                    # create enlarged block
                    bunched_blocks[new_qindices] = len(new_data)
                    # cp has new legs and thus gives the new shape
                    new_block = np.zeros(cp._block_shape(new_qindices), dtype=cp.dtype)
                    new_data.append(new_block)
                    new_qdata.append(new_qindices)
                else:
                    new_block = new_data[bunched_blocks[new_qindices]]
                # figure out where to insert the in the new bunched_blocks
                old_slbeg = [l.qind[qi, 0] for l, qi in itertools.izip(self.legs, old_qindices)]
                new_slbeg = [l.qind[qi, 0] for l, qi in itertools.izip(cp.legs, new_qindices)]
                slbeg = [(o-n) for o, n in itertools.izip(old_slbeg, new_slbeg)]
                sl = [slice(beg, beg+l) for beg, l in itertools.izip(slbeg, old_block.shape)]
                # insert the old block into larger new block
                new_block[tuple(sl)] = old_block
            else:
                # just copy the old block
                new_data.append(old_block.copy())
                new_qdata.append(new_qindices)
        cp._data = new_data
        cp._qdata = np.array(new_qdata, dtype=QDTYPE)
        cp._qsorted = False
        return cp

    def _block_slices(self, qindices):
        """return slices for the block specified by qindices"""
        return tuple([slice(l.qind[qi, 0], l.qind[qi, 1]) for l, qi in
                      itertools.izip(self.legs, qindices)])

    def _block_shape(self, qindices):
        """return shape for the block given by qindices"""
        return tuple([(l.qind[qi, 1] - l.qind[qi, 0]) for l, qi in
                      itertools.izip(self.legs, qindices)])


# functions ====================================================================

def zeros(chargeinfo, legcharges, qtotal=None):
    """create a npc array full of zeros (with no _data).

    This is just a wrapper around ``Array(...)``,
    detailed documentation can be found in the class doc-string of :class:`Array`."""
    return Array(chargeinfo, legcharges, qtotal)