
import os

from numpy import array, float32, float64, int32, uint8
import numpy as np

from openmdao.main.api import implements
from openmdao.main.interfaces import IParametricGeometry
from openmdao.main.geom import ParametricGeometry

from pygem_diamond import gem


class GEMParametricGeometry(ParametricGeometry):
    """A wrapper for a GEM object with modifiable parameters.  This object
    implements the IParametricGeometry interface.
    """

    def __init__(self):
        super(GEMParametricGeometry, self).__init__()
        self._model = None
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
            self.invoke_callbacks()

        return self._model

    def regen_model(self):
        if self._model is not None:
            try:
                return self._model.regenerate()
            except Exception as err:
                raise RuntimeError("Error regenerating model: %s" % str(err))

    def list_parameters(self):
        """Return a list of parameters (inputs and outputs) for this model.
        """
        if self._model is not None:
            return self._model.listParams()
        else:
            return []

    def set_parameter(self, name, val):
        """Set new value for a driving parameter.

        """
        if self._model is not None:
            try:
                self._model.setParam(name, val)
            except Exception as err:
                raise RuntimeError("Error setting parameter '%s': %s" % (name, str(err)))
        else:
            raise RuntimeError("Error setting parameter: no model")

    def get_parameters(self, names):
        """Get parameter values"""
        if self._model is not None:
            return [self._model.getParam(n) for n in names]
        else:
            raise RuntimeError("Error getting parameters: no model")

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

    def get_static_geometry(self):
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
        
    def get_visualization_data(self, wv, iBRep=None, angle=0., relSide=0., relSag=0.):
        '''Fills the given WV_Wrapper object with data for faces,
        edges, colors, etc.
        
        wv: WV_Wrapper object

        iBRep: int
            Index (starting at 0) of BRep to visualize. If not supplied, all BReps
            will be visualized.  
        '''
        if self._model is None:
            return []

        self._model.make_tess(wv, iBRep, angle, relSide, relSag)


try:
    from pyV3D.handlers import WV_ViewHandler
except ImportError:
    pass
else:
    class GEMViewHandler(WV_ViewHandler):

        @staticmethod
        def get_file_extensions():
            """Returns a list of file extensions that this handler knows how to view."""
            return ['csm']

        @staticmethod
        def can_view(obj):
            """Returns True if the given object type is viewable using this handler."""
            return isinstance(GEMParametricGeometry) or isinstance(GEMGeometry)

        def create_geom(self):
            eye    = array([0.0, 0.0, 7.0], dtype=float32)
            center = array([0.0, 0.0, 0.0], dtype=float32)
            up     = array([0.0, 1.0, 0.0], dtype=float32)
            fov   = 30.0
            zNear = 1.0
            zFar  = 10.0

            bias  = 1
            self.wv.createContext(bias, fov, zNear, zFar, eye, center, up)

            self.my_param_geom = GEMParametricGeometry()
            self.my_param_geom.model_file = os.path.expanduser(os.path.abspath(self.geometry_file))
            geom = self.my_param_geom.get_static_geometry()
            if geom is None:
                raise RuntimeError("can't get Geometry object")
            geom.get_visualization_data(self.wv, angle=15., relSide=.02, relSag=.001)

