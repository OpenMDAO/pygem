
import os

import numpy as np

from openmdao.main.api import implements
from openmdao.main.interfaces import IParametricGeometry

from pygem_diamond import gem


class GEMParametricGeometry(object):
    """A wrapper for a GEM object with modifiable parameters.  This object
    implements the IParametricGeometry interface.
    """

    implements(IParametricGeometry)

    def __init__(self):
        super(GEMParametricGeometry, self).__init__()
        self._model = None
        self._callbacks = []
        self._context = gem.Context()
        self._model_file = None

    @property
    def model_file(self):
        return self._model_file

    @model_file.setter
    def model_file(self, fname):
        self.load_model(os.path.expanduser(fname))

    def load_model(self, filename):
        """Load a model from a file."""

        if self._model is not None:
            self._model.release()

        self._model = None

        try:
            if filename:
                if os.path.isfile(filename):
                    self._model = self._context.loadModel(filename)
                else:
                    raise IOError("file '%s' not found." % filename)
        finally:
            self._model_file = filename
            for cb in self._callbacks:
                cb()

        return self._model

    def regenModel(self):
        if self._model is not None:
            try:
                return self._model.regenerate()
            except Exception as err:
                raise RuntimeError("Error regenerating model: %s" % str(err))

    def listParameters(self):
        """Return a list of parameters (inputs and outputs) for this model.
        """
        if self._model is not None:
            return self._model.listParams()
        else:
            return []

    def setParameter(self, name, val):
        """Set new value for a driving parameter.

        """
        if self._model is not None:
            try:
                self._model.setParam(name, val)
            except Exception as err:
                raise RuntimeError("Error setting parameter '%s': %s" % (name, str(err)))
        else:
            raise RuntimeError("Error setting parameter: no model")

    def getParameter(self, name):
        """Get info about a Parameter in a Model"""
        if self._model is not None:
            return self._model.getParam(name)
        else:
            raise RuntimeError("Error getting parameter: no model")

    def register_param_list_changedCB(self, callback):
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def terminate(self):
        """Terminate GEM context."""
        self._context = None

    def get_attributes(self, io_only=True):
        """Return an attribute dict for use by the openmdao GUI.
        """
        
        return {
            'type': type(self).__name__,
            'Inputs': [
                {
                    'name': 'model_file',
                    'id': 'model_file',
                    'type': type(self._model_file).__name__,
                    'value': self._model_file,
                    'connected': '',
                }
            ]
        }

    def get_geometry(self):
        if self._model is not None:
            geom = GEMGeometry()
            geom._model = self._model
            return geom
        return None


class GEMGeometry(object):
    '''A wrapper for a GEM object that respresents a specific instance of a
    geometry at a designated set of parameters. Parameters are not modifiable.
    This object is able to provide data for visualization.
    
    TODO: Add feature suppression.
    TODO: Add query of the BRep.
    TODO: Add DRep manipulation.
    '''
    
    def __init__(self):
        
        # GEM stores the geometry model in a context which originates in the
        # GEMParametricGeometry object.
        self._model  = None
        
        
    def return_visualization_data(self, iBRep=None):
        '''Returns a list of tuples containing the ndarrays of triangle
        coordinates and connectivities for the chosen brep.
        
        iBRep: int
            Index (starting at 0) of BRep to visualize. If not supplied, all BReps
            will be visualized.  
        '''
                  
        server, filename, modeler, uptodate, myBReps, nparam, \
            nbranch, nattr = self._model.getInfo() 
        
        myDRep = self._model.newDRep()
        
        # Tesselate our DRep
        # BRep, maxang, maxlen, maxasg
        if iBRep is None:
            myDRep.tessellate(0, 0, 0, 0)  # do all BReps
            all_BReps = range(0, len(myBReps))
        else:
            myDRep.tessellate(iBRep+1, 0, 0, 0) # do only specified BRep
            all_BReps = [iBRep]
           
        data = [] 
        for iBRep in all_BReps:
            
            # Need number of faces
            box, typ, nnode, nedge, nloop, nface, nshell, \
                        nattr = myBReps[iBRep].getInfo()
                        
            # Get the tessellation for each face
            data_triArray = []
            data_xyzArray = []
            for iface in range(1, nface+1):
                triArray, xyzArray = myDRep.getTessel(iBRep+1, iface)
                
                # pyV3D wants 1D arrays of single precision
                xyzArray = xyzArray.astype(np.float32).flatten()
                triArray = triArray.astype(np.int32).flatten()
                
                data_xyzArray.append(xyzArray)
                data_triArray.append(triArray)
                
                
            # Return single arrays containing all faces
            full_xyzArray = np.hstack(data_xyzArray)
            full_triArray = np.hstack(data_triArray)
            
            data.append( (full_xyzArray, full_triArray) )
        
        return data