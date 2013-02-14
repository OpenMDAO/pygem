# this is a simple python file that demonstrates optimization using the "gem" module

import numpy
from pygem_quartz import gem

# allow printing of entire arrays
numpy.set_printoptions(threshold=100)

# initialize the context, load the model, and get the first BRep
myContext = gem.initialize()
myModel   = gem.loadModel(myContext, "sample.csm")
foo       = gem.getModel(myModel)
myBRep    = foo[4][0]

# plot initial configuration
print "plotting initial configuration..."
myDRep = gem.newDRep(myModel)
gem.tesselDRep(myDRep, 0, 0, 0, 0)
gem.plotDRep(myDRep);
gem.destroyDRep(myDRep)

# "optimization" loop to vary "ymax" to drive the "volume" to a specified value
volume_target = 15
dvoldymax     =  8

for iter in range(100):
    print "---------------------"
    print "iter  =", iter

    # get current design variable
    foo  = gem.getParam(myModel, 4)
    ymax = foo[3][0]
    print "ymax  =", ymax

    # get the objective function
    foo    = gem.getModel(myModel)
    myBRep = foo[4][0]
    foo    = gem.getMassProps(myBRep, "BREP", 0)
    volume = foo[0]
    print "volume=", volume

    # if we have converged, stop the iterations
    if (abs(volume-volume_target) < 0.001):
        break

    # change the box height and regenerate the model
    ymax = ymax + (volume_target - volume) / dvoldymax
    print "ymax  =", ymax
    gem.setParam(myModel, 4, (ymax,))
    gem.regenModel(myModel)

# print and plot "final" configuration
print "*******************************"
print "final  ymax   =", ymax
print "target volume =", volume_target
print "final  volume =", volume
print "differ volume =", volume-volume_target

print "plotting final configuration..."
myDRep = gem.newDRep(myModel)
gem.tesselDRep(myDRep, 0, 0, 0, 0)
gem.plotDRep(myDRep);
gem.destroyDRep(myDRep)

# release the Model and terminate
gem.releaseModel(myModel)
gem.terminate(myContext)

print "SUCCESSFUL completion of 'opt_gem.py'"
