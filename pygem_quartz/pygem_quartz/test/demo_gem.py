# this is a simple python file that demonstrates the use of the "gem" module
import os
import numpy
from pygem_quartz import gem
help(gem)

# allow printing of only parts of ndarrays
numpy.set_printoptions(threshold=100)

# initialize GEM
myContext = gem.initialize()                      ;print "myContext      ->", myContext

gem.setAttribute(myContext, "CONTEXT", 0, "s_attr", "context attribute")
gem.setAttribute(myContext, "CONTEXT", 0, "i_attrs", (1111111, 2222222, 3333333, 4444444))
gem.setAttribute(myContext, "CONTEXT", 0, "r_attrs", (0.1234567890123456, 0.2345678901234567))

dum    = 0                                        ;print "s_attr(myContext)"
foo    = gem.retAttribute(myContext, "CONTEXT", 0, "s_attr")
aindex = foo[0]                                   ;print ".  aindex      ->", aindex
values = foo[1]                                   ;print ".  values      ->", values

ydum    = 0                                        ;print "i_attrs(myContext)"
foo    = gem.retAttribute(myContext, "CONTEXT", 0, "i_attrs")
aindex = foo[0]                                   ;print ".  aindex      ->", aindex
values = foo[1]                                   ;print ".  values      ->", values

dum    = 0                                        ;print "r_attrs(myContext)"
foo    = gem.retAttribute(myContext, "CONTEXT", 0, "r_attrs")
aindex = foo[0]                                   ;print ".  aindex      ->", aindex
values = foo[1]                                   ;print ".  values      ->", values

# load the sample OpenCSM Model
myModel = gem.loadModel(myContext, 
                        os.path.join(os.path.dirname(__file__),"sample.csm"))
print "myModel        ->", myModel

gem.setAttribute(myModel, "MODEL",  0, "s_attr", "model attribute")
gem.setAttribute(myModel, "MODEL",  0, "i_attrs", (2222222, 3333333, 4444444, 5555555))
gem.setAttribute(myModel, "MODEL",  0, "r_attrs", (0.2345678901234567, 0.3456789012345678))
gem.setAttribute(myModel, "BRANCH", 2, "s_attr", "branch attribute")
gem.setAttribute(myModel, "PARAM",  1, "s_attr", "param attribute")
gem.setAttribute(myModel, "PARAM",  1, "i_attrs", (3333333, 4444444, 5555555, 6666666))
gem.setAttribute(myModel, "PARAM",  1, "r_attrs", (0.3456789012345678, 0.4567890123456789))

# get information about the Model
foo      = gem.getModel(myModel)

server   = foo[0]                                 ;print "server         ->", server
filename = foo[1]                                 ;print "filename       ->", filename
modeler  = foo[2]                                 ;print "modeler        ->", modeler
uptodate = foo[3]                                 ;print "uptodate       ->", uptodate
myBReps  = foo[4]                                 ;print "myBReps        ->", myBReps
nparam   = foo[5]                                 ;print "nparam         ->", nparam
nbranch  = foo[6]                                 ;print "nbranch        ->", nbranch
nattr    = foo[7]

gem.setAttribute(myBReps[0], "BREP", 0, "s_attr", "brep attribute")
gem.setAttribute(myBReps[0], "BREP", 0, "i_attrs", (0, 1, 2, 3, 4, 5))
gem.setAttribute(myBReps[0], "BREP", 0, "r_attrs", (0., 0.1, 0.2, 0.3, 0.4, 0.5))

# get information about each of the Attributes
for iattr in range(1, nattr+1):
    dum    = 0                                    ;print "iattr          ->", iattr
    foo    = gem.getAttribute(myModel, "MODEL", 0, iattr)
    aname  = foo[0]                               ;print ".  aname       ->", aname
    values = foo[1]                               ;print ".  values      ->", values

# get information about each of the Parameters
for iparam in range(1, nparam+1):
    dum    = 0                                    ;print "iparam         ->", iparam
    foo    = gem.getParam(myModel, iparam)
    pname  = foo[0]                               ;print ".  pname       ->", pname
    bflag  = foo[1]                               ;print ".  bflag       ->", bflag
    order  = foo[2]                               ;print ".  order       ->", order
    values = foo[3]                               ;print ".  values      ->", values
    nattr  = foo[4]

    for iattr in range(1, nattr+1):
        dum    = 0                                ;print ".  iattr       ->", iattr
        foo    = gem.getAttribute(myModel, "PARAM", iparam, iattr)
        aname  = foo[0]                           ;print ".  .  aname    ->", aname
        values = foo[1]                           ;print ".  .  values   ->", values

# get information about each of the Branches
for ibranch in range(1, nbranch+1):
    dum      = 0                                  ;print "ibranch        ->", ibranch
    foo      = gem.getBranch(myModel, ibranch)
    bname    = foo[0]                             ;print ".  bname       ->", bname
    btype    = foo[1]                             ;print ".  btype       ->", btype
    suppress = foo[2]                             ;print ".  suppress    ->", suppress
    parents  = foo[3]                             ;print ".  parents     ->", parents
    childs   = foo[4]                             ;print ".  childs      ->", childs
    nattr    = foo[5]

    for iattr in range(1, nattr+1):
        dum    = 0                                ;print ".  iattr       ->", iattr
        foo    = gem.getAttribute(myModel, "BRANCH", ibranch, iattr)
        aname  = foo[0]                           ;print ".  .  aname    ->", aname
        values = foo[1]                           ;print ".  .  values   ->", values

# get information about each of the BReps
for myBRep in myBReps:
    dum      = 0                                  ;print "myBRep         ->", myBRep
    foo      = gem.getBRepOwner(myBRep)
    model    = foo[0]                             ;print ".  model       ->", model
    instance = foo[1]                             ;print ".  instance    ->", instance
    branch   = foo[2]                             ;print ".  branch      ->", branch

    foo    = gem.getMassProps(myBRep, "BREP", 0)
    volume = foo[ 0]                              ;print ".  volume      ->", volume
    area   = foo[ 1]                              ;print ".  area        ->", area
    xcg    = foo[ 2]                              ;print ".  xcg         ->", xcg
    ycg    = foo[ 3]                              ;print ".  ycg         ->", ycg
    zcg    = foo[ 4]                              ;print ".  zcg         ->", zcg
    Ixx    = foo[ 5]                              ;print ".  Ixx         ->", Ixx
    Ixy    = foo[ 6]                              ;print ".  Ixy         ->", Ixy
    Ixz    = foo[ 7]                              ;print ".  Ixz         ->", Ixz
    Iyx    = foo[ 8]                              ;print ".  Iyx         ->", Iyx
    Iyy    = foo[ 9]                              ;print ".  Iyy         ->", Iyy
    Iyz    = foo[10]                              ;print ".  Iyz         ->", Iyz
    Izx    = foo[11]                              ;print ".  Izx         ->", Izx
    Izy    = foo[12]                              ;print ".  Izy         ->", Izy
    Izz    = foo[13]                              ;print ".  Izz         ->", Izz

    foo    = gem.getBRepInfo(myBRep)
    box    = foo[0]                               ;print ".  box         ->", box
    type   = foo[1]                               ;print ".  type        ->", type
    nnode  = foo[2]                               ;print ".  nnode       ->", nnode
    nedge  = foo[3]                               ;print ".  nedge       ->", nedge
    nloop  = foo[4]                               ;print ".  nloop       ->", nloop
    nface  = foo[5]                               ;print ".  nface       ->", nface
    nshell = foo[6]                               ;print ".  nshell      ->", nshell
    nattr  = foo[7]

    # get information about each of the Attrs
    for iattr in range(1, nattr+1):
        dum    = 0                                ;print ".  iattr       ->", iattr
        foo    = gem.getAttribute(myBRep, "BREP", 0, iattr)
        aname  = foo[0]                           ;print ".  .  aname    ->", aname
        values = foo[1]                           ;print ".  .  values   ->", values

    # get information about each of the Nodes
    for inode in range(1, nnode+1):
        dum    = 0                                ;print ".  inode       ->", inode
        foo    = gem.getNode(myBRep, inode)
        xyz    = foo[0]                           ;print ".  .  xyz      ->", xyz
        nattr  = foo[1]

        for iattr in range(1, nattr+1):
            dum    = 0                            ;print ".  .  iattr    ->", iattr
            foo    = gem.getAttribute(myBRep, "NODE", inode, iattr)
            aname  = foo[0]                       ;print ".  .  .  aname ->", aname
            values = foo[1]                       ;print ".  .  .  values->", values

    # get information about each of the Edges
    for iedge in range(1, nedge+1):
        dum    = 0                                ;print ".  iedge       ->", iedge
        foo    = gem.getEdge(myBRep, iedge)
        tlimit = foo[0]                           ;print ".  .  tlimit   ->", tlimit
        nodes  = foo[1]                           ;print ".  .  nodes    ->", nodes
        faces  = foo[2]                           ;print ".  .  faces    ->", faces
        nattr  = foo[3]

        for iattr in range(1, nattr+1):
            dum    = 0                            ;print ".  .  iattr    ->", iattr
            foo    = gem.getAttribute(myBRep, "EDGE", iedge, iattr)
            aname  = foo[0]                       ;print ".  .  .  aname ->", aname
            values = foo[1]                       ;print ".  .  .  values->", values

    # get information about each of the Loops
    for iloop in range(1, nloop+1):
        dum    = 0                                ;print ".  iloop       ->", iloop
        foo    = gem.getLoop(myBRep, iloop)
        face   = foo[0]                           ;print ".  .  face     ->", face
        type   = foo[1]                           ;print ".  .  type     ->", type
        edges  = foo[2]                           ;print ".  .  edges    ->", edges
        nattr  = foo[3]

        for iattr in range(1, nattr+1):
            dum    = 0                            ;print ".  .  iattr    ->", iattr
            foo    = gem.getAttribute(myBRep, "LOOP", iloop, iattr)
            aname  = foo[0]                       ;print ".  .  .  aname ->", aname
            values = foo[1]                       ;print ".  .  .  values->", values

    # get information about each of the Faces
    for iface in range(1, nface+1):
        dum    = 0                                ;print ".  iface       ->", iface
        foo    = gem.getFace(myBRep, iface)
        ID     = foo[0]                           ;print ".  .  ID       ->", ID
        uvbox  = foo[1]                           ;print ".  .  uvbox    ->", uvbox
        norm   = foo[2]                           ;print ".  .  norm     ->", norm
        loops  = foo[3]                           ;print ".  .  loops    ->", loops
        nattr  = foo[4]

        for iattr in range(1, nattr+1):
            dum    = 0                            ;print ".  .  iattr    ->", iattr
            foo    = gem.getAttribute(myBRep, "FACE", iface, iattr)
            aname  = foo[0]                       ;print ".  .  .  aname ->", aname
            values = foo[1]                       ;print ".  .  .  values->", values

        foo    = gem.getMassProps(myBRep, "FACE", iface)
        area   = foo[ 1]                          ;print ".  .  area     ->", area

    # get information about each of the Shells
    for ishell in range(1, nshell+1):
        dum    = 0                                ;print ".  ishell      ->", ishell
        foo    = gem.getShell(myBRep, ishell)
        type   = foo[0]                           ;print ".  .  type     ->", type
        faces  = foo[1]                           ;print ".  .  faces    ->", faces
        nattr  = foo[2]

        for iattr in range(1, nattr+1):
            dum    = 0                            ;print ".  .  iattr    ->", iattr
            foo    = gem.getAttribute(myBRep, "SHELL", ishell, iattr)
            aname  = foo[0]                       ;print ".  .  .  aname ->", aname
            values = foo[1]                       ;print ".  .  .  values->", values

        foo    = gem.getMassProps(myBRep, "SHELL", ishell)
        area   = foo[ 1]                          ;print ".  .  area     ->", area

# set up a DRep for myModel
myDRep = gem.newDRep(myModel)                     ;print "myDRep         ->", myDRep

maxang = 0;
maxlen = 0;
maxsag = 0;
gem.tesselDRep(myDRep, 0, maxang, maxlen, maxsag);

# get tessellation associated with the first Face of each DRep
for ibrep in range(1, len(myBReps)+1):
    dum   = 0                                     ;print ".  ibrep       ->", ibrep
    iface = 1                                     ;print ".  .  iface    ->", iface

    foo  = gem.getTessel(myDRep, ibrep, iface)
    xyz  = foo[0]                                 ;print ".  .  xyz    --->\n", xyz
    uv   = foo[1]                                 ;print ".  .  uv     --->\n", uv
    conn = foo[2]                                 ;print ".  .  conn   --->\n", conn

# plot the DRep
print "plotting myDRep..."
gem.plotDRep(myDRep);

# destroy the DRep
print "destroying myDRep"
gem.destroyDRep(myDRep)

# create a static model and add myBrep into it twice
newModel = gem.staticModel(myContext)             ;print "newModel       ->", newModel

print "adding myBRep to newModel twice"
gem.add2Model(newModel, myBRep)
gem.add2Model(newModel, myBRep, (0.5, 0, 0, 2,   0, 0.5, 0, 0,  0, 0, 0.5, 0))

foo      = gem.getModel(newModel)
server   = foo[0]                                 ;print "server         ->", server
filename = foo[1]                                 ;print "filename       ->", filename
modeler  = foo[2]                                 ;print "modeler        ->", modeler
uptodate = foo[3]                                 ;print "uptodate       ->", uptodate
myBReps  = foo[4]                                 ;print "myBReps        ->", myBReps
nparam   = foo[5]                                 ;print "nparam         ->", nparam
nbranch  = foo[6]                                 ;print "nbranch        ->", nbranch
nattr    = foo[7]

for iattr in range(1, nattr+1):
    dum    = 0                                    ;print "iattr          ->", iattr
    foo    = gem.getAttribute(newModel, "MODEL", 0, iattr)
    aname  = foo[0]                               ;print ".  aname       ->", aname
    values = foo[1]                               ;print ".  values      ->", values

print "plotting newDRep..."
newDRep = gem.newDRep(newModel)
gem.tesselDRep(newDRep, 0, 0, 0, 0)
gem.plotDRep(newDRep);
gem.destroyDRep(newDRep)

print "releasing newModel"
gem.releaseModel(newModel);

# release the Model and terminate
print "releasing myModel"
gem.releaseModel(myModel)

# terminate GEM
print "terminating myContext"
gem.terminate(myContext)

print "SUCCESSFUL completion of 'test_gem.py'"
