
import os

from openmdao.main.api import implements
from openmdao.main.interfaces import IParametricGeometry

from pygem_diamond import gem


# class GEMStaticGeometry(object):
#     """A wrapper for a GEM object without parameters.  This object implements the
#     IStaticGeometry interface.
#     """
#     def __init__(self):
#         pass

#     def get_tris(self):
#         pass


class GEMParametricGeometry(object):
    """A wrapper for a GEM object with modifiable parameters.  This object
    implements the IParametricGeometry interface.
    """

    implements(IParametricGeometry)

    def __init__(self, mfile=''):
        super(GEMParametricGeometry, self).__init__()
        self._model = None
        self._callbacks = []
        self._context = gem.Context()
        self._model_file = mfile

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

