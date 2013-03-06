
cdef extern from "math.h":
    double sqrt( double x)


cdef extern from "gem.h":

    int GEM_MCONTEXT
    int GEM_MMODEL
    int GEM_MBREP
    int GEM_MDREP

    int GEM_OUTSIDE
    int GEM_SUCCESS
    int GEM_BADDREP
    int GEM_BADFACEID
    int GEM_BADBOUNDINDEX
    int GEM_BADVSETINDEX
    int GEM_BADRANK
    int GEM_BADDSETNAME
    int GEM_MISSDISPLACE
    int GEM_NOTFOUND
    int GEM_BADMODEL
    int GEM_BADCONTEXT
    int GEM_BADBREP
    int GEM_BADINDEX
    int GEM_NOTCHANGED 
    int GEM_ALLOC
    int GEM_BADTYPE
    int GEM_NULLVALUE
    int GEM_NULLNAME
    int GEM_NULLOBJ
    int GEM_BADOBJECT
    int GEM_WIREBODY
    int GEM_SOLIDBODY
    int GEM_NOTESSEL
    int GEM_BADVALUE
    int GEM_DUPLICATE
    int GEM_BADINVERSE
    int GEM_NOTPARAMBND
    int GEM_NOTCONNECT
    int GEM_NOTPARMTRIC
    int GEM_READONLYERR
    int GEM_FIXEDLEN
    int GEM_ASSEMBLY
    int GEM_BADNAME
    int GEM_UNSUPPORTED
    int GEM_BADMETHOD

    int GEM_NODE
    int GEM_EDGE
    int GEM_LOOP
    int GEM_FACE
    int GEM_SHELL
    int GEM_BREP

    int GEM_PARAM
    int GEM_BRANCH
    int GEM_MODEL

    int GEM_CONTEXT
    int GEM_DREP

    int GEM_BOOL
    int GEM_INTEGER
    int GEM_REAL
    int GEM_STRING
    int GEM_SPLINE

    int GEM_SUBTRACT
    int GEM_INTERSECT
    int GEM_UNION

    int GEM_SOLID
    int GEM_SHEET
    int GEM_WIRE

    cdef union my_gem_ident:
        void* ptr
        int tag

    ctypedef struct gemID:
        int index
        my_gem_ident ident

    ctypedef struct gemAttr:
        char* name
        int length
        int type
        int* integers
        double* reals
        char* string

    ctypedef struct gemAttrs:
        int nattrs
        int matters
        gemAttr* attrs

    ctypedef struct gemSpl:
        int    pntype
        double pos[3]
        double tan[3]
        double radc[3]

    cdef union param_vals_union:
        int    bool1
        int    *bools
        int    integer
        int    *integers
        double real
        double *reals
        char   *string
        gemSpl *splpnts

    cdef union bounds_union:
        int    ilims[2]
        double rlims[2]

    ctypedef struct gemParam:
        char    *name
        int      type
        int      order
        gemID    handle
        short    changed
        short    bitflag
        int      len
        param_vals_union vals
        bounds_union bnds
        gemAttrs *attr

    cdef union feat_par_union:
        int pnode
        int *pnodes

    cdef union feat_child_union:
        int node
        int *nodes

    ctypedef struct gemFeat:
        char     *name
        gemID     handle
        short     sflag
        short     changed
        char     *branchType
        int       nParents
        feat_par_union parents
        int       nChildren
        feat_child_union children
        gemAttrs *attr

    ctypedef struct gemModel:
        int      magic
        gemID    handle
        int      nonparam
        char     *server
        char     *location
        char     *modeler
        int      nBRep
        gemBRep  **BReps
        int      nParams
        gemParam *Params
        int      nBranches
        gemFeat  *Branches
        gemAttrs *attr
        gemModel *prev
        gemModel *next

    ctypedef struct gemBRep:
        pass

    ctypedef struct gemTri:
        pass

    ctypedef struct gemTConn:
        pass

    ctypedef struct gemDEdge:
        pass

    ctypedef struct gemTRep:
        int      nEdges
        gemDEdge *Edges
        int      nFaces
        gemTri   *Faces
        gemTConn *conns

    ctypedef struct gemBound:
        pass

    ctypedef struct gemDRep:
        int magic
        gemModel *model
        int nIDs
        char **IDs
        int nBReps
        gemTRep *TReps
        int nBound
        gemBound *bound
        gemAttrs *attr
        gemDRep *prev
        gemDRep *next

    ctypedef struct gemNode:
        pass

    ctypedef struct gemEdge:
        pass

    ctypedef struct gemLoop:
        pass

    ctypedef struct gemFace:
        pass

    ctypedef struct gemShell:
        pass

    ctypedef struct gemBody:
        pass


    ctypedef struct gemCntxt:
        int magic
        gemModel *model
        gemDRep  *drep
        gemAttrs *attr

    ctypedef struct gemPair:
        int BRep                 # BRep index 
        int index                # entity index 


    int gem_initialize(gemCntxt **context)

    int gem_terminate(gemCntxt *context)

    int gem_staticModel(gemCntxt *cntxt, gemModel **model)

    int gem_loadModel(gemCntxt *cntxt, char server[], char location[], gemModel **model)

    int gem_getBRepOwner(gemBRep *brep, gemModel **model, int *instance, int *branch)

    int gem_solidBoolean(gemBRep *breps, gemBRep *brept, double xform[], 
                         int sboType, gemModel **model)              

    int gem_getAttribute(void *gemObj, int etype, int eindex, int aindex,
                         char *name[], int *atype, int *alen, 
                         int *integers[], double *reals[], char *string[])

    int gem_retAttribute(void *gemObj, int etype, int eindex, char name[],
                         int *aindex, int *atype, int *alen, 
                         int *integers[], double *reals[], char *string[])


    int gem_setAttribute(void *gemObj, int etype, int eindex, char name[], int atype, 
                         int alen, int integers[], double reals[], char string[])

    int gem_getObject(void *gemObj, int  *otype, int  *nattr)

    char* gem_errorString(int code)

    void gem_free(void *ptr)

    int gem_add2Model(gemModel *model, gemBRep *BRep, double xform[])
               
    int gem_saveModel(gemModel *model, char location[])

    int gem_releaseModel(gemModel *model)

    int gem_copyModel(gemModel *model, gemModel **newmdl)

    int gem_regenModel(gemModel *model)

    int gem_getModel(gemModel *model, char *server[], char *filnam[], char *modeler[],
                     int *uptodate, int *nBRep, gemBRep **BReps[], int *nParams,
                     int *nBranch, int *nattr)

    int gem_getBranch(gemModel *model, int branch, char *bname[], char *btype[], 
                      int *suppress, int *nparent, int *parents[], int *nchild, 
                      int *children[], int *nattr)

    int gem_setSuppress(gemModel *model, int branch, int suppress)

    int gem_getParam(gemModel *model, int param, char *pname[], int *bflag,
                     int *order, int *ptype, int *plen, int *integers[], 
                     double *reals[], char *string[], gemSpl *spline[], int *nattr)

    int gem_getLimits(gemModel *model, int param, int intlims[2], double realims[2])

    int gem_setParam(gemModel *model, int param, int plen, 
                     int integers[], double reals[], char string[], gemSpl spline[])



    int gem_getBRepInfo(gemBRep *brep,      # (in)  BRep pointer 
                    double  box[],          # (out) xyz bounding box (6) 
                    int     *type,          # (out) body type 
                    int     *nnode,         # (out) number of nodes 
                    int     *nedge,         # (out) number of edges 
                    int     *nloop,         # (out) number of loops 
                    int     *nface,         # (out) number of faces 
                    int     *nshell,        # (out) number of shells 
                    int     *nattr)         # (out) number of attributes 


    int gem_getShell(gemBRep *brep,  # (in)  BRep pointer 
                 int     shell,  # (in)  shell index 
                 int     *type,  # (out) 0 outer, 1 inner 
                 int     *nface,  # (out) number of faces 
                 int     *faces[],  # (out) pointer to list of faces 
                 int     *nattr)  # (out) number of attributes 


    int gem_getFace(gemBRep *brep,              # (in)  BRep pointer 
                int     face,               # (in)  face index 
                char    *ID[],              # (out) pointer to persistent ID 
                double  uvbox[],            # (out) uv bounding box (4) 
                int     *norm,      # (out) flip normal flag (-1 or 1) 
                int     *nloops,        # (out) number of loops 
                int     *loops[],           # (out) loop indices 
                int     *nattr)            # (out) number of attributes 


    int gem_getWire(gemBRep *brep,              # (in)  BRep pointer 
                int     *nloops,        # (out) number of loops 
                int     *loops[])          # (out) loop indices 


    int gem_getLoop(gemBRep *brep,              # (in)  BRep pointer 
                int     loop,               # (in)  loop index 
                int     *face,              # (out) owning face index 
                int     *type,              # (out) 0 outer, 1 inner 
                int     *nedge,     # (out) number of edges in loop 
                int     *edges[],       # (out) pointer to edges/senses 
                int     *nattr)            # (out) number of attributes 


    int gem_getEdge(gemBRep *brep,              # (in)  BRep pointer 
                int edge,  # (in)  edge index 
                double tlimit[],  # (out) t range (2) 
                int nodes[],  # (out) bounding node indices (2) 
                int faces[],  # (out) trimmed faces (2) 
                int *nattr)  # (out) number of attributes 


    int gem_getNode(gemBRep *brep,  # (in)  BRep pointer 
                int node,  # (in)  node index 
                double xyz[],  # (out) coordinates (3) 
                int *nattr)  # (out) number of attributes 


    int gem_getMassProps(gemBRep *brep,  # (in)  BRep pointer 
                     int etype,  # (in)  Topo: Face, Shell or Body 
                     int eindex,  # (in)  Topological entity index 
                     double props[])  # (out) the data returned (must be 
                                                     #declared to at least #14):
                                                     #  volume, surface area
                                                     #  center of gravity (3)
                                                     #  inertia matrix at #CoG (9) 


    int gem_isEquivalent(int etype,  # (in)  Topological entity type 
                     gemBRep *brep1,  # (in)  BRep pointer 
                     int eindex1,  # (in)  Topological entity index 
                     gemBRep *brep2,  # (in)  BRep pointer 
                     int eindex2)  # (in)  Topological entity index 

    int gem_newDRep(gemModel *model,            #  (in)  pointer to model #
                gemDRep  **drep)           #  (out) pointer to new DRep #


    #  (re)tessellate the BReps in the DRep
    
    # If all input triangulation parameters are set to zero, then a default 
    # tessellation is performed. If this data already exists, it is recomputed 
    # and overwritten. Use gem_getTessel or gem_getDiscrete to get the data 
    # generated on the appropriate entity.
    
    int gem_tesselDRep(gemDRep *drep,           #  (in)  pointer to DRep #
                   int     BRep,            #  (in)  BRep index in DRep - 0 all #
                   double  angle,           #  (in)  tessellation parameters: #
                   double  maxside,         #        all zero -- default #
                   double  sag)

                   
    #  get Edge Discretization
    
    # # Returns the polyline associated with the Edge.
    
    int gem_getDiscrete(gemDRep *drep,          #  (in)  pointer to DRep #
                        gemPair bedge,          #  (in)  BRep/Edge index in DRep #
                        int     *npts,          #  (out) number of vertices #
                        double  *xyzs[])       #  (out) pointer to the coordinates #


    #  get Face Tessellation
    
    # Returns the triangulation associated with the Face.
    
    int gem_getTessel(gemDRep *drep,            #  (in)  pointer to DRep #
                  gemPair bface,            #  (in)  BRep/Face index in DRep #
                  int     *ntris,           #  (out) number of triangles #
                  int     *npts,            #  (out) number of vertices #
                  int     *tris[],          #  (out) pointer to triangles defns #
                  double  *xyzs[])         #  (out) pointer to the coordinates #


    #  create a DRep that has Bound definitions copied from another DRep
    
    # Creates a new DRep by populating empty Bounds based on an existing DRep. 
    # The unused persistent Face IDs are reported and pointer to the list should 
    # be freed by the programmer after the data is examined by gem_free. Though 
    # unused, these IDs are maintained in the appropriate Bound definitions 
    # (for possible later use).
    
    int gem_copyDRep(gemDRep  *src,             #  (in)  pointer to DRep to be copied #
                 gemModel *model,           #  (in)  pointer to new master model #
                 gemDRep  **drep,           #  (out) pointer to the new DRep #
                 int      *nIDs,            #  (out) # of IDs in model not in src #
                 char     **IDs[])         #  (out) array  of IDs (freeable) #


    # #  delete a DRep and all its storage
    
    # # Cleans up all data contained within the DRep. This also frees up the 
    # # DRep structure.
    
    # int gem_destroyDRep(gemDRep *drep)         #  (in)  ptr to DRep to be destroyed #


    # #  create Bound in a DRep
    
    # # Create a new Bound within the DRep. The IDs are the unique and persistent 
    # # strings returned via gem_getFace (that is, this ID persists across Model 
    # # regenerations where the number of Faces and the Face order does not). 
    # # After a Model regen all VertexSets (and tessellations) found in the DRep 
    # # are removed but the Bound remains.
    #  #
    # int gem_createBound(gemDRep *drep,          #  (in)  pointer to DRep #
    #                 int     nIDs,           #  (in)  number of Face IDs #
    #                 char    *IDs[],         #  (in)  array  of Face IDs #
    #                 int     *ibound)       #  (out) index of new Bound #


    # #  add other IDs to a Bound
    
    # # Extends an existing Bound within the DRep by including more ID strings. 
    # # This may be necessary if after a Model regen new Faces are found that 
    # # would belong to this Bound. This may also be useful in a multidisciplinary 
    # # setting for adding places where data transfer is performed. This call will
    # # invalidate any existing data in the Bound and therefore acts like the 
    # # existing connected Vertex Sets are newly created.
    #  #
    # int gem_extendBound(gemDRep *drep,          #  (in)  pointer to DRep #
    #                 int     ibound,         #  (in)  Bound to be extended #
    #                 int     nIDs,           #  (in)  number of new Face IDs #
    #                 char    *IDs[])        #  (in)  array  of new Face IDs #


    # #  create a connected VertexSet
    
    # # Generates a new VertexSet (for connected Sets) and returns the index. This
    # # invokes the function "gemDefineQuilt" in the disMethod so/DLL (specified for 
    # # the connected VertexSet) based on the collection of Faces (Quilt) currently
    # # defined in the Bound. Note that VertexSets can overlap in space but the 
    # # internals of a Quilt may not.
    #  #
    # int gem_createVset(gemDRep *drep,           #  (in)  pointer to DRep #
    #                int     ibound,          #  (in)  index of Bound #
    #                char    disMethod[],     #  (in)  the name of the shared object
    #                                                 # or DLL used for describing
    #                                                 # interploation & integration #
    #                int     *ivs)           #  (out) index of Vset in Bound #


    # #  create an unconnected VertexSet
    
    # # Makes an unconnected VertexSet. This can be used for receiving interpolation 
    # # data and performing evaluations. An unconnected VertexSet cannot specify a
    # # (re)parameterization (by invoking gem_paramBound), be the source of data
    # # transfers or get sensitivities.
    #  #
    # int gem_makeVset(gemDRep *drep,             #  (in)  pointer to DRep #
    #              int     ibound,            #  (in)  index of Bound #
    #              int     npts,              #  (in)  number of points #
    #              double  xyz[],             #  (in)  3*npts XYZ positions #
    #              int     *ivs)             #  (out) index of Vset in Bound #


    # #  (re)parameterize a Bound
     
    # # Creates a global parameterization (u,v) based on all connected VertexSets
    # # in the bound. If the parameterization exists, making this call does nothing.
     
    # # This function must be performed before any data is put into a VertexSet or
    # # a transfer is requested. If there is only 1 Face specified in the Bound then 
    # # the native parameterization for that Face is used for the DRep functions. 
    # # DataSets named "xyz" and "uv" are produced for all of the VertexSets found 
    # # in the Bound (where "uv" is the single parameterization for the Bound).
    # # DataSets "xyzd" and "uvd" may also be generated if the data positions differ
    # # from the geometric reference positions.
     
    # int gem_paramBound(gemDRep *drep,           #  (in)  DRep pointer #
    #                int     ibound)         #  (in)  Bound index (1-bias) #


    # #  put data into a Vset
     
    # # Stores the named data in a connected VertexSet. This allows for the 
    # # assignment of field data, for example "pressure" (rank = 1) can be associated 
    # # with the vertex ref positions. Only one VertexSet within a Bound can have a 
    # # unique name, therefore it is an error to put a DataSet on a VertexSet if the 
    # # name already exists (or the name is reserved such as "xyz" or "uv", see 
    # # below). If the name exists in this VertexSet the data will be overwritten.
     
    # int gem_putData(gemDRep *drep,              #  (in)  pointer to DRep #
    #             int     ibound,             #  (in)  index of Bound #
    #             int     ivs,                #  (in)  index of Vset in Bound #
    #             char    name[],             #  (in)  dataset name:
    #                                 # if name exists in ibound
    #                                  #   if name associated with ivs and rank agrees
    #                                  #      overwrite dataset
    #                                  #   else
    #                                  #      GEM_BADSETNAME
    #                                  # else
    #                                     # create new dataset #
    #             int     nverts,             #  (in)  number of verts -- must match
    #                                                 # VSet total from getVsetInfo #
    #             int     rank,               #  (in)  # of elements per vertex #
    #             double  data[])            #  (in)  rank*nverts data values or NULL
    #                                                #  deletes the Dset if exists #


    # #  get data from a Vset
     
    # # Returns (or computes and returns) the data found in the VertexSet. If another 
    # # VertexSet in the Bound has the name, then data transfer is performed. 
    # # Transfers are performed via the "xferMethod".
    # # 
    # # The following reserved names automatically generate data (with listed rank) 
    # # and are all geometry based (nGpts in length not nVerts of getVsetInfo):
    # # coordinates:         xyz  (3)        parametric coords:      uv       (2)
    # # d(xyz)/d(uv):        d1   (6)        d2(xyz)/d(uv)2:         d2       (9)
    # # GeomCurvature:       curv (8)        Inside/Outside:         inside   (1)
    # #                                      Sensitivity:            d/dPARAM (3)
    # #                  Where PARAM is the full parameter name in the Model
     
    # # The following reserved names automatically generate data (with listed rank)
    # # and are all data based (nVerts in length not nGpts of getVsetInfo):
    # # data coordinates:    xyzd (3)        data parametrics:       uvd      (2)
     
    # # Data that is explicity placed via gem_putData is based on the number of 
    # # data reference positions (nVerts) not the geometry positions (nGpts).
     
    # int gem_getData(gemDRep *drep,              #  (in)  pointer to DRep #
    #             int     ibound,             #  (in)  index of Bound #
    #             int     ivs,                #  (in)  index of Vset in Bound #
    #             char    name[],             #  (in)  dataset name #
    #             int     xferMethod,         #  (in)  GEM_INTERP, GEM_CONSERVE, or
    #                                                 # GEM_MOMENTS. Note: GEM_CONSERVE 
    #                                                 # and GEM_MOMENTS can be added #
    #             int     *npts,              #  (out) number of points/verts #
    #             int     *rank,              #  (out) # of elements per vertex #
    #             double  *data[])           #  (out) pointer to data values #


    # #  get Vset-based Triangulation
    # # 
    # # Returns the triangulation associated with the Bound for this Vset. This
    # # internally forces a call to the function "gemTriangulate" in the disMethod 
    # # so/DLL to fill in the appropriate data (which is done in a way that using
    # # normal triangle color rendering techniques produces correct imagery). The 
    # # triangle indices returned can be used as reference into the DataSet (bias 1) 
    # # in order to view the results.
     
    # int gem_triVset(gemDRep *drep,              #  (in)  pointer to DRep #
    #             int     ibound,             #  (in)  index of Bound #
    #             int     ivs,                #  (in)  index of Vset in Bound #
    #             char    name[],             #  (in)  dataset name (used to trigger
    #                                                  #geom vs data ref) #
    #             int     *ntris,             #  (out) number of triangles #
    #             int     *npts,              #  (out) number of vertices #
    #             int     *tris[],            #  (out) pointer to triangles defns #
    #             double  *xyzs[])           #  (out) pointer to the coordinates #


    #  get info about a DRep
    # 
    # Returns current information about a DRep. Many DRep functions refer to a 
    # BRep by index (bias 1), you must use the Model to get the information about
    # the included BReps where the listed order is the index. Use gem_getModel to 
    # get the number of BReps and the actual BRep pointers. 
     #
    int gem_getDRepInfo(gemDRep  *drep,         #  (in)  pointer to DRep #
                    gemModel **model,       #  (out) owning Model pointer #
                    int      *nbound,       #  (out) number of Bounds #
                    int      *nattr)       #  (out) number of attributes #


    # #  get info about a Bound

    #  # Returns current information about a Bound in a DRep. Any BRep/Face pair 
    #  # which has the value {0,0} reflects an inactive ID. The uvbox returns 
    #  # {0.0,0.0,0.0,0.0} if the Bound has yet to be parameterized (see 
    #  # gem_paramBound). Note that the uvbox returned is different than that of 
    #  # the Face's uvbox if there is more than a single Face. In this case the 
    #  # aggregate Faces are fit to a single surface and the uvbox reflects this 
    #  # new "skin".
     
    # int gem_getBoundInfo(gemDRep *drep,         #  (in)  pointer to DRep #
    #                  int     ibound,        #  (in)  index of Bound #
    #                  int     *nIDs,         #  (out) number of Face IDs #
    #                  int     *iIDs[],       #  (out) array  of Face ID indices #
    #                  gemPair *indices[],    #  (out) BRep/Face active pairs #
    #                  double  uvlimits[],    #  (out) UV box (umin,umax,vmin,vmax) #
    #                  int     *nvs)         #  (out) number of Vsets #


    # #  get info about a Vset
    # # 
    # # Returns information about a VertexSet. This includes the number of 
    # # geometric and data reference vertices and the number of DataSets. Pointers 
    # # to collections of names and ranks should be freed by the programmer after 
    # # they are used by invoking gem_free.
    # #
    # int gem_getVsetInfo(gemDRep *drep,          #  (in)  pointer to DRep #
    #                 int     ibound,         #  (in)  index of Bound #
    #                 int     ivs,            #  (in)  index of Vset in Bound #
    #                 int     *vstype,        #  (out) 0-connected, 1-unconnected #
    #                 int     *nGpts,         #  (out) total number of Geom points #
    #                 int     *nVerts,        #  (out) total number of Data verts #
    #                 int     *nset,          #  (out) number of datasets #
    #                 char    **names[],      #  (out) ptr to Dset names (freeable) #
    #                 int     *ivsrc[],       #  (out) ptr to Dset owner 
    #                                                 # 0 or index of src (freeable) #
    #                 int     *ranks[])      #  (out) ptr to Dset ranks (freeable) #
                    


    # #  ********************* transfer method (loadable) code ******************** #

    # #  all dynamically loadable transfer methods are defined by the name of the
    # #   shared object/DLL and each must contain the following 6 functions:  #


    # int gemDefineQuilt(
    #                gemDRep  *drep,          #  (in)  the DRep pointer -- read-only #
    #                int      nFaces,         #  (in)  number of BRep/Face pairs
    #                                                  #to be filled #
    #                gemPair  bface[],        #  (in)  index to the pairs in DRep #
    #                gemQuilt *quilt)        #  (out) ptr to Quilt to be filled #


    # void gemFreeQuilt(gemQuilt *quilt)          #  (in)  ptr to the Quilt to free #


    # int gemTriangulate(gemQuilt *quilt,         #  (in)  the quilt description #
    #                int      geomFlag,       #  (in)  0 - data ref, 1 - geom based #
    #                gemTri   *tessel)       #  (out) ptr to the triangulation #


    # void gemFreeTriangles(gemTri *tessel)       #  (in)  ptr to triangulation to free #


    # int gemInterpolation(gemQuilt *quilt,       #  (in)  the quilt description #
    #                  int      geomFlag,     #  (in)  0 - data ref, 1 - geom based #
    #                  int      eIndex,       #  (in)  element index (bias 1) #
    #                  double   st[],         #  (in)  the element ref coordinates #
    #                  int      rank,         #  (in)  data depth #
    #                  double   data[],       #  (in)  values (rank*npts in length) #
    #                  double   result[])    #  (out) interpolated result - (rank) #


    # int gemIntegration(gemQuilt   *quilt,       #  (in)  the quilt to integrate upon #
    #                int        whichQ,       #  (in)  1 or 2 (cut quilt index) #
    #                int        nFrags,       #  (in)  # of fragments to integrate #
    #                gemCutFrag frags[],      #  (in)  cut element fragments #
    #                int        rank,         #  (in)  data depth #
    #                double     data[],       #  (in)  values (rank*npts in length) #
    #                double     xyzs[],       #  (in)  3D coordinates #
    #                double     result[])    #  (out) integrated result - (rank) #
