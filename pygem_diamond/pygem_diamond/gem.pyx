
from gem cimport *

import cython
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from cpython cimport PyObject, Py_INCREF

import numpy as np
cimport numpy as np
np.import_array()

import os
import sys

# a set of currently valid GEM object, i.e. the have been allocated and not deleted
_gemObjects = set()

_def_xform = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.]

_err_codes = {
    'GEM_OUTSIDE': 1,
    'GEM_SUCCESS': 0,
    'GEM_BADDREP': -301,
    'GEM_BADFACEID': -302,
    'GEM_BADBOUNDINDEX': -303,
    'GEM_BADVSETINDEX': -304,
    'GEM_BADRANK': -305,
    'GEM_BADDSETNAME': -306,
    'GEM_MISSDISPLACE': -307,
    'GEM_NOTFOUND': -308,
    'GEM_BADMODEL': -309,
    'GEM_BADCONTEXT': -310,
    'GEM_BADBREP': -311,
    'GEM_BADINDEX': -312,
    'GEM_NOTCHANGED': -313,
    'GEM_ALLOC': -314,
    'GEM_BADTYPE': -315,
    'GEM_NULLVALUE': -316,
    'GEM_NULLNAME': -317,
    'GEM_NULLOBJ': -318,
    'GEM_BADOBJECT': -319,
    'GEM_WIREBODY': -320,
    'GEM_SOLIDBODY': -321,
    'GEM_NOTESSEL': -322,
    'GEM_BADVALUE': -323,
    'GEM_DUPLICATE': -324,
    'GEM_BADINVERSE': -325,
    'GEM_NOTPARAMBND': -326,
    'GEM_NOTCONNECT': -327,
    'GEM_NOTPARMTRIC': -328,
    'GEM_READONLYERR': -329,
    'GEM_FIXEDLEN': -330,
    'GEM_ASSEMBLY': -331,
    'GEM_BADNAME': -332,
    'GEM_UNSUPPORTED': -333,
    'GEM_BADMETHOD': -334,
}

# OOPS: I created the def_names dict backwards and I'm too lazy to fix it, so...
_err_codes = dict([(tup[1],tup[0]) for tup in _err_codes.items()])
del tup

_bool_type_dict = {
    'SUBTRACT': 0,
    'INTERSECT': 1,
    'UNION': 2,
}

_etype_dict = {
    'FACE': GEM_FACE,
    'SHELL': GEM_SHELL,
    'BREP': GEM_BREP,
    'CONTEXT': GEM_MCONTEXT,
    'MODEL': GEM_MODEL,
    'BRANCH': GEM_BRANCH,
    'PARAM': GEM_PARAM,
    'NODE': GEM_NODE,
    'EDGE': GEM_EDGE,
    'LOOP': GEM_LOOP,
}


def raise_exception(msg='', status=None, fname=None):
    status = _err_codes.get(status, status)
    message = "Error"
    if fname:
        message += " in %s" % fname
    if status:
        message += " (code=%s)" % status
    if msg:
        message += ": %s" % msg
    raise RuntimeError(message)

cdef int _add_gemobj(void* obj) except -1:
    _gemObjects.add(<long>obj)
    return 0

cdef int _remove_gemobj(void* obj) except -1:
    _gemObjects.discard(<long>obj)
    return 0

cdef int _check_gemobj(GemObj obj) except -1:
    """Raises an exception if the given object isn't a valid (allocated and not deleted)
    GEM object.
    """
    entity = obj.get_entity()
    if <long>entity not in _gemObjects:
        raise RuntimeError("object at %s does not point to a valid gem object" % hex(<long>entity))
    return 0

cdef int _remove_breps(gemModel *model) except -1:
    """remove any BReps from _gemObjects for this model"""

    cdef char *server, *filename, *modeler
    cdef int status, uptodate, nbrep, nparam, nbranch, nattr
    cdef gemBRep **breps

    if model != NULL and <long>model in _gemObjects:
        # remove any BReps from _gemObjects for this model
        status = gem_getModel(model, &server, &filename, &modeler,
                              &uptodate, &nbrep, &breps,
                              &nparam, &nbranch, &nattr);
        if status != GEM_SUCCESS:
            raise RuntimeError('failed to get info from model (code=%s)' % _err_codes.get(status, status))

        for i in range(nbrep):
            _remove_gemobj(breps[i])

    return 0

cdef int _releaseModel(gemModel *model) except -1:
    cdef int status

    if model != NULL:
        _remove_breps(model)
        if <long>(model) in _gemObjects:
            status = gem_releaseModel(model)
            if status != GEM_SUCCESS:
                raise RuntimeError('failed to release model (code=%s)' % _err_codes.get(status, status))
            _remove_gemobj(model)


# from http://gael-varoquaux.info/blog/?p=157 with small mods
cdef class ArrayWrapper:
    cdef void* data_ptr
    cdef int typenum, ndims, own
    cdef np.npy_intp *dims
 
    cdef int set_data(self, int ndims, np.npy_intp *dims, int typenum, int own, void* data_ptr) except -1:
        """ Set the data of the array (data is assumed to be contiguous (C))
 
        This cannot be done in the constructor as it must recieve C-level
        arguments.
 
        Parameters:
        -----------
        ndims: int 
            Number of dimensions
        dims: np.npy_intp
            Dimensions
        typenum: int
            type of array entries, e.g., NPY_DOUBLE
        own: int
            if 0, memory is not owned
        data_ptr: void*
            Pointer to the data            
 
        """
        self.ndims = ndims
        self.dims = dims
        self.typenum = typenum
        self.own = own
        self.data_ptr = data_ptr
        return 0
 
    def __array__(self):
        """ Here we use the __array__ method, that is called when numpy
            tries to get an array from the object."""
        return np.PyArray_SimpleNewFromData(self.ndims, self.dims,
                                            self.typenum, self.data_ptr)
 
    def __dealloc__(self):
        """ Frees the array (if self.own is true). This is called by Python when all the
        references to the object are gone. """
        if self.own:
            free(<void*>self.data_ptr)

cdef object npy_arr_from_data(void* data_ptr, int ndims, np.npy_intp *dims, 
                              int typenum, int copy):
    """Returns a numpy array based on the given data. If copy is False, the data
    will be owned by the numpy array (and deleted when the numpy array is deallocated).
    """
    cdef int i
    cdef np.npy_intp cdims[10]
    if copy:
        own = False
    else:
        own = True
    for i in range(ndims):
        cdims[i] = dims[i]
    wrapper = ArrayWrapper()
    wrapper.set_data(ndims, &cdims[0], typenum, own, data_ptr)
    return np.array(wrapper, copy=bool(copy))

cdef int _cvtXform(xform, double *transform) except -1:
    if xform is None:
        xform = _def_xform
    elif isinstance(xform, np.ndarray):
        xform = xform.ravel()
    elif not isinstance(xform, basestring):
        try:
            xform = [val for val in iter(xform)]
        except:
            raise RuntimeError('xform should be a length 12 array of reals')
    else:
        raise RuntimeError("xform should be a length 12 array of reals")

    if len(xform) != 12:
        raise RuntimeError("xform should contain 12 real entries")

    for i,val in enumerate(xform):
        transform[i] = val
    
    return 0


cdef class GemObj:

    def __str__(self):
        return "%s object at %s" % (self.__class__.__name__, hex(id(self)))

    cdef void* get_entity(self):
        raise_exception('get_entity not implemented')


cdef class HasAttrs(GemObj):
    def getAttribute(self, char *otype, eindex, ident):
        """
        Returns a tuple of the form (id, attribute). The attribute is specified by 
        ident, which can be an index or a name.  The id in the returned tuple will
        be an index if ident was a name, or a name if ident was an index.
        """
        cdef int status, atype, alen, *integers, aindex
        cdef double *reals
        cdef char *aname, *stringval
        _check_gemobj(self)

        objtype = _etype_dict.get(otype)
        if objtype is None:
            raise_exception('specified attribute type "%s" is invalid' % otype)

        if isinstance(ident, basestring):
            aname = ident
            status = gem_retAttribute(self.get_entity(), objtype, 
                                      eindex, aname, &aindex, &atype, &alen,
                                      &integers, &reals, &stringval)
            id_ret = aindex
        else:
            status = gem_getAttribute(self.get_entity(), objtype, 
                                      eindex, ident,
                                      &aname, &atype, &alen,
                                      &integers, &reals, &stringval)
            id_ret = aname
        if (status != GEM_SUCCESS):
            raise_exception('failed to get attribute using identifier %s' % ident,
                                 status, 'gem.getAttribute')

        if atype == GEM_INTEGER:
            return (id_ret, [integers[i] for i in range(alen)])
        elif atype == GEM_REAL:
            return (id_ret, [reals[i] for i in range(alen)])
        elif atype == GEM_STRING:
            s = stringval
            return (id_ret, s)
        else:
            raise_exception("couldn't convert attribute value into a python object",
                            GEM_BADTYPE, 'gem.getAttribute')


    def setAttribute(self, char* otype, int eindex, aname, values):
        cdef int status, alen, itype, *integers
        cdef double *reals
        _check_gemobj(self)

        objtype = _etype_dict.get(otype)
        if objtype is None:
            raise_exception('specified attribute type "%s" is invalid' % otype)

        if isinstance(values, basestring):
            itype = GEM_STRING
            alen = 1
        else:
            try:
                itype = GEM_INTEGER
                alen = len(values)
                for i,val in enumerate(values):
                    if isinstance(val, int):
                        pass
                    elif isinstance(val, float):
                        itype = GEM_REAL
                    else:
                        raise TypeError()
            except TypeError:
                raise TypeError("attribute value must be a string or tuple/list/array of int or float values")

        if itype == GEM_INTEGER:
            integers = <int*> malloc(len(values)*sizeof(int))
            if integers == NULL:
                raise_exception('failed to allocate integer array', 
                                      GEM_ALLOC, 'gem.setAttribute')

            for i in range(len(values)):
                integers[i] = <int>values[i]
            status = gem_setAttribute(self.get_entity(), objtype,
                                      eindex, aname, itype, alen, integers, NULL, NULL)
            if integers != NULL:
                free(integers)
        elif itype == GEM_REAL:
            reals = <double*> malloc(len(values)*sizeof(double))
            if reals == NULL:
                raise_exception('failed to allocate double array', GEM_ALLOC,
                                     'gem.setAttribute')
            for i in range(len(values)):
                reals[i] = <double>values[i]
            status = gem_setAttribute(self.get_entity(), objtype,
                                      eindex, aname, itype, alen,
                                      NULL, reals, NULL)
            if reals != NULL:
                free(reals)
        else: # GEM_STRING
            status = gem_setAttribute(self.get_entity(), objtype,
                                      eindex, aname, itype, alen,
                                      NULL, NULL, values)
        if (status != GEM_SUCCESS):
            raise_exception('failed to set attribute %s' % aname,
                            status, 'gem.setAttribute')


cdef class Attr(GemObj):
    cdef gemAttr* attr

    cdef void* get_entity(self):
        return <void*>self.attr


cdef class DRep(HasAttrs):
    cdef gemDRep* drep

    cdef void* get_entity(self):
        return <void*>self.drep

    def tessellate(self, int ibrep, double maxang, double maxlen, double maxsag):
        cdef:
            int status
            int typ, nnode, nedge, nloop, nface, nshell, nattr
            int uptodate, nbrep, nparam, nbranch, nbound, i
            double box[6], bbox[6], size
            char *filename, *server, *modeler
            gemModel *model
            gemBRep **breps, *brep
            gemDRep *drep
            int nbreps

        _check_gemobj(self)

        if maxang <=0. or maxlen <=0. or maxsag <= 0.:
            status = gem_getDRepInfo(self.drep, &model, &nbound, &nattr)
            if status != GEM_SUCCESS:
                raise_exception('failed to get DRep info', status, 'GemDRep.tessellate')

            box[0] = box[1] = box[2] = 1e20
            box[3] = box[4] = box[5] = -1e20

            status = gem_getModel(model, &server, &filename, &modeler,
                            &uptodate, &nbrep, &breps, &nparam, &nbranch, &nattr)
            if status != GEM_SUCCESS:
                raise_exception('failed to get Model info', status, 'GemDRep.tessellate')

            for i in range(1,nbrep+1):
                if ibrep == 0 or i == ibrep:
                    status = gem_getBRepInfo(breps[i-1], bbox, &typ, &nnode,
                                             &nedge, &nloop, &nface, &nshell, &nattr)
                    if status != GEM_SUCCESS:
                        raise_exception('failed to get BRep info', status, 'GemDRep.tessellate')

                    if bbox[0] < box[0]:
                        box[0] = bbox[0]
                    if bbox[1] < box[1]:
                        box[1] = bbox[1]
                    if bbox[2] < box[2]:
                        box[2] = bbox[2]
                    if bbox[3] > box[3]:
                        box[3] = bbox[3]
                    if bbox[4] > box[4]:
                        box[4] = bbox[4]
                    if bbox[5] > box[5]:
                        box[5] = bbox[5]

            size = sqrt( (box[0]-box[3])*(box[0]-box[3])
                       + (box[1]-box[4])*(box[1]-box[4])
                       + (box[2]-box[5])*(box[2]-box[5]))

            if maxang <= 0:
                maxang = 15
            if maxlen == 0:
                maxlen =  0.020  * size
            elif maxlen <  0:
                maxlen = -maxlen * size
            if maxsag == 0:
                maxsag =  0.001  * size
            elif maxsag <  0:
                maxsag = -maxsag * size

        drep = self.drep
        nbreps = drep.nBReps
        if <long>self.drep.model not in _gemObjects:
            raise_exception('failed to tessellate DRep: bad model')
        status = gem_tesselDRep(self.drep, ibrep, maxang, maxlen, maxsag)

        if status != GEM_SUCCESS:
            raise_exception('failed to tessellate DRep', status, 'GemDRep.tessellate')

    def getTessel(self, int ibrep, int iface):
        cdef:
            int status, npts, ntris, *tris, *tris_copy, ndims
            double *xyz, *xyz_copy
            gemPair bface
            np.npy_intp dims[2]

        _check_gemobj(self)

        bface.BRep = ibrep
        bface.index = iface
        status = gem_getTessel(self.drep, bface, &ntris, &npts, &tris, &xyz)
        if status != GEM_SUCCESS:
            raise_exception('gem_getTessel failed', status)

        ndims = 2;
        dims[0] = npts;
        dims[1] = 3;
        xyz_copy = <double*>malloc(3*npts*sizeof(double))
        memcpy(xyz_copy, <void*>xyz, 3*npts*sizeof(double))
        xyz_nd = np.PyArray_SimpleNewFromData(ndims, dims, np.NPY_DOUBLE, xyz_copy)
        np.PyArray_UpdateFlags(xyz_nd, np.NPY_OWNDATA)   

        dims[0] = ntris;
        dims[1] = 3;
        tris_copy = <int*>malloc(3*ntris*sizeof(int))
        memcpy(tris_copy, <void*>tris, 3*ntris*sizeof(int))
        tri_nd = np.PyArray_SimpleNewFromData(ndims, dims, np.NPY_INT, tris_copy)
        np.PyArray_UpdateFlags(tri_nd, np.NPY_OWNDATA)
        return (tri_nd, xyz_nd)

    def getDiscrete(self, int ibrep, int iedge):
        cdef:
            int status, npts, ndims
            double *xyz, *xyz_copy
            gemPair bedge
            np.npy_intp dims[2]

        _check_gemobj(self)

        bedge.BRep = ibrep
        bedge.index = iedge
        status = gem_getDiscrete(self.drep, bedge, &npts, &xyz)
        if status != GEM_SUCCESS:
            raise_exception('gem_getDiscrete failed', status)

        ndims = 2;
        dims[0] = npts;
        dims[1] = 3;
        xyz_copy = <double*>malloc(3*npts*sizeof(double))
        memcpy(xyz_copy, <void*>xyz, 3*npts*sizeof(double))
        xyz_nd = np.PyArray_SimpleNewFromData(ndims, dims, np.NPY_DOUBLE, xyz_copy)
        np.PyArray_UpdateFlags(xyz_nd, np.NPY_OWNDATA)

        return xyz_nd 


cdef object createPyDRep(gemDRep* drep):
    pydrep = DRep()
    pydrep.drep = drep
    _add_gemobj(drep)
    return pydrep


cdef class BRep(HasAttrs):
    cdef gemBRep* brep

    cdef void* get_entity(self):
        return <void*>self.brep

    def getOwner(self):
        cdef gemModel* model
        cdef int instance, branch, status

        _check_gemobj(self)
        status = gem_getBRepOwner(self.brep, &model, &instance, &branch);
        if (status != GEM_SUCCESS):
            raise_exception(status=status, fname='BRep.getOwner')

        return (createPyModel(model), instance, branch)

    def get_gemtype(self):
        return GEM_BREP

    def getMassProps(self, etype, eindex):
        cdef int status
        cdef double props[14]

        _check_gemobj(self)
        itype = _etype_dict.get(etype)
        if itype is None:
            raise_exception('getMassProps: etype of "%s" is invalid' % etype)

        status = gem_getMassProps(self.brep, itype, eindex, props)
        if status != GEM_SUCCESS:
            raise_exception('failed to get mass props', status,
                                 'gem_getMassProps')

        return tuple([props[i] for i in range(14)])

    def getInfo(self):
        cdef int status, typ, nnode, nedge, nloop, nface, nshell, nattr, i
        cdef double box[6]

        _check_gemobj(self)

        status = gem_getBRepInfo(self.brep, box, &typ, &nnode, &nedge, &nloop,
                                 &nface, &nshell, &nattr)
        if status != GEM_SUCCESS:
            raise_exception('failed to get brep info', status, 'gem.BRep.getInfo')

        return (tuple([box[i] for i in range(6)]),
                typ, nnode, nedge, nloop, nface, nshell, nattr)

    def getNode(self, int inode):
        cdef int status, nattr
        cdef double xyz[3]

        _check_gemobj(self)

        status = gem_getNode(self.brep, inode, xyz, &nattr)
        if status != GEM_SUCCESS:
            raise_exception('getNode(%s) failed' % inode, status, 'gem.BRep.getNode')

        return ((xyz[0],xyz[1],xyz[2]), nattr)

    def getLoop(self, int iloop):
        cdef int status, face, typ, nedge, *edges, nattr, i

        _check_gemobj(self)

        status = gem_getLoop(self.brep, iloop, &face, &typ, &nedge, &edges, &nattr)
        if status != GEM_SUCCESS:
            raise_exception('getLoop(%s) failed' % iloop, status, 'gem.BRep.getLoop')

        return (face, typ, tuple([edges[i] for i in range(nedge)]), nattr)

    def getEdge(self, int iedge):
        cdef int status, nodes[2], faces[2], nattr, i
        cdef double tlimit[2]

        _check_gemobj(self)

        status = gem_getEdge(self.brep, iedge, tlimit, nodes, faces, &nattr)
        if status != GEM_SUCCESS:
            raise_exception('getEdge(%s) failed' % iedge, status, 'gem.BRep.getEdge')

        return ((tlimit[0], tlimit[1]),
                (nodes[0], nodes[1]),
                (faces[0], faces[1]), nattr)

    def getShell(self, int ishell):
        cdef int status, typ, nface, *faces, nattr, i

        _check_gemobj(self)

        status = gem_getShell(self.brep, ishell, &typ, &nface, &faces, &nattr)
        if status != GEM_SUCCESS:
            raise_exception('getShell(%s) failed' % ishell, status, 'gem.BRep.getShell')

        return (typ, tuple([faces[i] for i in range(nface)]), nattr)

    def getFace(self, int iface):
        cdef int status, norm, nloop, *loops, nattr, i
        cdef double uvbox[4]
        cdef char *ID

        _check_gemobj(self)

        status = gem_getFace(self.brep, iface, &ID, uvbox, &norm, &nloop, &loops, &nattr)
        if status != GEM_SUCCESS:
            raise_exception('getFace(%s) failed' % iface, status, 'gem.BRep.getFace')

        return (ID, (uvbox[0],uvbox[1],uvbox[2],uvbox[3]),
                norm, tuple([loops[i] for i in range(nloop)]), nattr)


cdef object createPyBRep(gemBRep* brep):
    pybrep = BRep()
    pybrep.brep = brep
    _add_gemobj(brep)
    return pybrep


cdef class Model(HasAttrs):
    cdef gemModel* model
    cdef object _param_map

    def __init__(self):
        self._reset()

    cdef void* get_entity(self):
        return <void*>self.model

    cdef int _reset(self) except -1:
        self.model = NULL
        self._param_map = {}
        return 0

    cdef int _set_model(self, gemModel *model) except -1:
        if self.model != NULL:
            raise_exception('gem model of Model object has already been set')

        if model != NULL:
            _add_gemobj(model)

        self.model = model
        self._populate()

    cdef int _populate(self) except -1:
        """fill in name to id mappings, etc."""
        cdef char *server, *filename, *modeler, *pname, *string
        cdef int status, uptodate, nbrep, nparam, nbranch, bflag, i, j
        cdef int order, ptype, plen, *integers, nattr
        cdef gemBRep **breps
        cdef double *reals
        cdef gemSpl *spline

        _check_gemobj(self)
        status = gem_getModel(self.model, &server, &filename, &modeler,
                              &uptodate, &nbrep, &breps,
                              &nparam, &nbranch, &nattr);
        if status != GEM_SUCCESS:
            raise_exception('failed to get info from model', status, 'gem_getModel')

        # do parameter mapping

        for i in range(1,nparam+1):
            status = gem_getParam(self.model, i, &pname, &bflag, &order,
                                  &ptype, &plen, &integers, &reals, &string,
                                  &spline, &nattr)
            if status != GEM_SUCCESS:
                raise_exception('failed to get parameter %s' % i, status, 'gem_getParam')

            name = pname
            if name.startswith('@'):
                name = name[1:]

            if ptype not in [GEM_BOOL, GEM_INTEGER, GEM_REAL, GEM_STRING]:
                raise_exception('unsupported type returned from gem_getParam. (name=%s)' % name)

            self._param_map[name] = i

        return 0

    def listParams(self, iotype=None):
        params = []
        if iotype not in [None, 'in', 'out']:
            raise_exception('list_params: iotype must be one of [None, "in", "out"]')
        for name, ident in self._param_map.items():
            meta = self.getParam(ident)
            if meta['iotype'] == iotype or iotype is None:
                params.append((name, meta))
        return params
        
    def getLimits(self, int iparam):
        cdef int status, integers[2], i
        cdef double reals[2]

        _check_gemobj(self)

        integers[0] = 0
        integers[1] = 0
        reals[0] = 0.
        reals[1] = 0.

        status = gem_getLimits(self.model, iparam, integers, reals)
        if status != GEM_SUCCESS:
            raise_exception('failed to get limits for param %s' % iparam,
                                 status, 'gem.Model.getLimits')

        if reals[0] != 0. or reals[1] != 0.:
            return (reals[0], reals[1])
        else:
            return (integers[0], integers[1])

    def get_gemtype(self):
        return GEM_MODEL

    def save(Model self, filename):
        cdef int status
        _check_gemobj(self)
        status = gem_saveModel(self.model, filename)
        if status != GEM_SUCCESS:
            raise_exception('model save to %s failed' % filename, status, '')

    def add2Model(Model model, BRep brep, xform=None):
        _check_gemobj(model)
        _check_gemobj(brep)

        cdef double transform[12]
        cdef int status

        _cvtXform(xform, transform)
        status = gem_add2Model(<gemModel*>model.get_entity(), 
                               <gemBRep*>brep.get_entity(), transform);
        if (status != GEM_SUCCESS):
            raise_exception('failed to add BRep to model', status, 'gem.add2Model')

    def newDRep(self):
        cdef int status
        cdef gemDRep *drep

        _check_gemobj(self)

        status = gem_newDRep(self.model, &drep)
        if status != GEM_SUCCESS:
            raise_exception("can't create new DRep", status, 'gem.newDRep')

        return createPyDRep(drep)

    def getInfo(self):
        cdef char *server, *filename, *modeler
        cdef int status, uptodate, nbrep, nparam, nbranch, nattr
        cdef gemBRep **breps

        _check_gemobj(self)
        status = gem_getModel(self.model, &server, &filename, &modeler,
                              &uptodate, &nbrep, &breps,
                              &nparam, &nbranch, &nattr);
        if status != GEM_SUCCESS:
            raise_exception('failed to get info from model', status, 'Model.getInfo')

        pybreps = [createPyBRep(breps[i]) for i in range(nbrep)]

        tup = (server if server != NULL else '', 
               filename if filename != NULL else '', 
               modeler if modeler != NULL else '', 
               uptodate,
               pybreps, nparam, nbranch, nattr)
        return tup

    def release(self):
        self._param_map = {}
        if self.model != NULL:
            _releaseModel(self.model)
            self._reset()

    def copy(self):
        cdef int status, uptodate, nbrep, nparam, nbranch, nattr
        cdef char *server, *filename, *modeler
        cdef gemModel *model2

        _check_gemobj(self)

        status = gem_copyModel(self.model, &model2)
        if status != GEM_SUCCESS:
            raise_exception('failed to copy model', status, 'gem.copyModel')

        return createPyModel(model2)

    def regenerate(self):
        """Rebuilds the model"""
        _check_gemobj(self)
        _remove_breps(self.model)
        status = gem_regenModel(self.model)
        if status != GEM_SUCCESS:
            raise_exception('failed to regenerate model', status, 'gem_regenModel')

    def getBranch(self, int ibranch):
        cdef int status, suppress, nparent, *parents, nchild, *childs, nattr
        cdef int iparent, ichild
        cdef char *bname, *btype

        _check_gemobj(self)
        status = gem_getBranch(self.model, ibranch, &bname, &btype, &suppress,
                               &nparent, &parents, &nchild, &childs, &nattr)
        if status != GEM_SUCCESS:
            raise_exception('failed to get branch %s' % ibranch, status, 'gem_getBranch')

        return (bname, btype, suppress,
                tuple([parents[i] for i in range(nparent)]),
                tuple([childs[i] for i in range(nchild)]),
                nattr)

    def setSuppress(self, int ibranch, int istate):
        cdef int status

        _check_gemobj(self)
        status = gem_setSuppress(self.model, ibranch, istate)
        if status != GEM_SUCCESS:
            raise_exception('failed to set branch %s suppression state to %s' % (ibranch, istate), 
                            status, 'gem_setSuppress')

    def getParam(self, param_id):
        cdef int status, bflag, order, ptype, plen, *integers, nattr
        cdef double *reals
        cdef char *pname, *string
        cdef gemSpl *spline

        _check_gemobj(self)

        if isinstance(param_id, basestring): # it's the parameter name
            try:
                iparam = self._param_map[param_id]
            except KeyError:
                raise_exception('parameter %s not found' % param_id)
        elif isinstance(param_id, (int, long)):
            iparam = param_id
        else:
            raise_exception('param_id must be an int or string')

        status = gem_getParam(self.model, iparam, &pname, &bflag, &order,
                              &ptype, &plen, &integers, &reals, &string,
                              &spline, &nattr)
        if status != GEM_SUCCESS:
            raise_exception('failed to get parameter %s' % iparam, status, 
                            'gem_getParam')

        if ptype == GEM_BOOL or ptype == GEM_INTEGER:
            if plen == 1:
                value = integers[0]
            else:
                value = tuple([integers[i] for i in range(plen)])
        elif ptype == GEM_REAL:
            if plen == 1:
                value = reals[0]
            else:
                value = tuple([reals[i] for i in range(plen)])
        elif ptype == GEM_STRING:
            value = string
        else:
            raise_exception('type of value returned from gem_getParam (%s) is invalid' % ptype)

        meta = {}
        if bflag & 8:  # check if param has limits
            high, low = self.getLimits(iparam)
            meta['high'] = high
            meta['low'] = low
        else:
            high = low = None
        meta['value'] = value
        if (bflag & 2):
            meta['iotype'] = 'out'
        else:
            meta['iotype'] = 'in'
        meta['nattr'] = nattr
        meta['order'] = order
        meta['id'] = iparam
        name = pname
        if name.startswith('@'):
            name = name[1:]
        meta['name'] = name
        return meta

    def setParam(self, param_id, values):
        cdef int status, iparam, itype, plen, *integers=NULL
        cdef double *reals=NULL
        cdef char *string
        cdef np.ndarray[int, ndim=1] ivals
        cdef np.ndarray[double, ndim=1] rvals

        iparam = 0  # silence compiler warning
        _check_gemobj(self)

        if isinstance(param_id, basestring): # it's the parameter name
            iparam = self._param_map[param_id]
        elif isinstance(param_id, (int, long)):
            iparam = param_id
        else:
            raise_exception('param_id must be an int or string')

        if isinstance(values, basestring):
            itype = GEM_STRING
            plen = 1
        else:
            try:
                plen = len(values)
            except TypeError:
                values = (values,)

            itype = GEM_INTEGER
            plen = len(values)
            for i,val in enumerate(values):
                if isinstance(val, int):
                    pass
                elif isinstance(val, float):
                    itype = GEM_REAL
                else:
                    raise TypeError("parameter value must be a string or tuple/list/array of int or float values")

        if itype == GEM_INTEGER:
            ivals = np.array(values, dtype=np.int, order='c')
            status = gem_setParam(self.model, iparam, plen, &(ivals[0]), NULL, NULL, NULL)
        elif itype == GEM_REAL:
            rvals = np.array(values, dtype=np.double, order='c')
            status = gem_setParam(self.model, iparam, plen, NULL, &(rvals[0]), NULL, NULL)
        else: # GEM_STRING
            status = gem_setParam(self.model, iparam, plen, NULL, NULL, values, NULL)
        if (status != GEM_SUCCESS):
            raise_exception('failed to set parameter %s' % param_id,
                            status, 'gem.setParam')

    def get_bounding_box(self, iBRep=None):
        server, filename, modeler, uptodate, BReps, nparam, \
            nbranch, nattr = self.getInfo() 
        
        if iBRep:
            BReps = [BReps[iBRep]]

        if BReps:
            box = [1e99, 1e99, 1e99, -1e99, -1e99, -1e99]
            for BRep in BReps:
                tup = BRep.getInfo()
                bx = tup[0]
                if (bx[0] < box[0]):
                    box[0] = bx[0]
                if (bx[1] < box[1]):
                    box[1] = bx[1]
                if (bx[2] < box[2]):
                    box[2] = bx[2]

                if (bx[3] > box[3]):
                    box[3] = bx[3]
                if (bx[4] > box[4]):
                    box[4] = bx[4]
                if (bx[5] > box[5]):
                    box[5] = bx[5]
        else:
            box = [0]*6

        return box

    def make_tess(self, wv, iBRep=None, angle=0., relSide=0., relSag=0.):
        box = self.get_bounding_box(iBRep)

        size = box[3] - box[0]
        if (size < box[4]-box[1]):
            size = box[4] - box[1]
        if (size < box[5]-box[2]):
            size = box[5] - box[2]

        focus = [0.]*4
        focus[0] = 0.5*(box[0] + box[3])
        focus[1] = 0.5*(box[1] + box[4])
        focus[2] = 0.5*(box[2] + box[5])
        focus[3] = size

        server, filename, modeler, uptodate, breps, nparam, \
            nbranch, nattr = self.getInfo() 
        
        drep = self.newDRep()
        
        if iBRep is not None:
            breps = [breps[iBRep]]
            iBRep += 1
        else:
            iBRep = 0

        drep.tessellate(iBRep, angle, relSide*focus[3], relSag*focus[3])

        for i, brep in enumerate(breps):
            bx, typ, nnode, nedge, nloop, nface, nshell, nattr = brep.getInfo()

            for j in range(1, nface+1):
                tris, xyzs = drep.getTessel(i+1, j)
                wv.set_face_data(xyzs.astype(np.float32).flatten(), 
                                 tris.astype(np.int32).flatten(), bbox=box,
                                 name="Body %d Face %d"%(i+1,j), fp=outfp)

            for j in range(1, nedge+1):
                points = drep.getDiscrete(i+1, j)
                if len(points) < 2:
                    continue
                wv.set_edge_data(points.astype(np.float32).flatten(),
                                bbox=box,
                                name="Body %d Edge %d"%(i+1,j), fp=outfp)


    def __dealloc__(self):
        self.release()


cdef object createPyModel(gemModel* model):
    pymod = Model()
    pymod._set_model(model)
    return pymod


cdef class Context(HasAttrs):
    cdef gemCntxt* context

    def __cinit__(self):
        cdef int status
        cdef gemCntxt* context

        status = gem_initialize(&context)
        if status != GEM_SUCCESS:
            self.context = NULL
            raise_exception("unable to initialize GEM context", status)
        self.context = context
        _add_gemobj(context)

    def __dealloc__(self):
        cdef gemModel *mod
        if self.context != NULL:
            _remove_gemobj(self.context)
            mod = self.context.model
            while mod != NULL:
                next = mod.next
                _releaseModel(mod)
                mod = next

            gem_terminate(self.context)

    def get_gemtype(self):
        return GEM_MCONTEXT

    def loadModel(self, fname):
        cdef gemModel *model
        cdef int status
        _check_gemobj(self)
        if not os.path.isfile(fname):
            raise IOError("Can't find model file '%s'" % fname)
        status = gem_loadModel(self.context, NULL, fname, &model)
        if status != GEM_SUCCESS:
            raise_exception("model failed to load", status, 'Context.loadModel')
        return createPyModel(model)

    def staticModel(self):
        """
        Returns a new static model object.
        """
        cdef gemModel *model
        cdef int status
        status = gem_staticModel(self.context, &model)
        if status != GEM_SUCCESS:
            raise_exception('failed to create a static model', status, 
                            'Context.staticModel')
        return createPyModel(model)

    cdef void* get_entity(self):
        return <void*>self.context


def solidBoolean(BRep brep1, BRep brep2, optype, xform=None):
    """Perform a boolean operation on the given BReps."""
    cdef gemModel* model
    cdef double transform[12]
    cdef int status

    if optype not in _bool_type_dict:
        raise RuntimeError("'optype' must be one of %s" % _bool_type_dict.keys())
    if not isinstance(brep1, BRep):
        raise RuntimeError("brep1 is not a GEMBrep object")
    if not isinstance(brep2, BRep):
        raise RuntimeError("brep2 is not a GEMBrep object")

    _check_gemobj(brep1)
    _check_gemobj(brep2)

    _cvtXform(xform, transform)

    status = gem_solidBoolean(brep1.brep, brep2.brep, transform, 
                              _bool_type_dict[optype], &model)
    if (status != GEM_SUCCESS):
        raise_exception(status, 'gemSolidBoolean')

    return createPyModel(model)


