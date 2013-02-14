/*
 ************************************************************************
 *                                                                      *
 * pygem -- Python interface to the Geometry Environment for MDAO (GEM)*
 *                                                                      *
 *            Written by John Dannenhoffer @ Syracuse University        *
 *                                                                      *
 ************************************************************************
 */

#include <Python.h>
#include <structmember.h>
#include "numpy/arrayobject.h"

#include "gem.h"

#ifdef GEM_GRAPHICS
    #include "gv.h"
    static gemDRep    *plotDRep = NULL;
    static gemBRep    **plotBReps;
    static int        plotNg, plotNbrep;
    static GvGraphic **plotList;           /* list of graphic objects */
    static FILE       *script  = NULL;     /* pointer to script file (if it exists) */
    static int        new_data = 1;        /* =1 means image needs to be updated */
    static int        numarg   = 0;        /* numeric argument */
    static double     plotBox[6];          /* bounding box of config */

    extern float      gv_xform[4][4];

    /* window defines */
    #define           DataBase        1
    #define           TwoD            2
    #define           ThreeD          3
    #define           Dials           4
    #define           Key             5

    /* event types */
    #define           KeyPress        2
    #define           KeyRelease      3
    #define           ButtonPress     4
    #define           ButtonRelease   5
    #define           Expose          12
    #define           NoExpose        14
    #define           ClientMessage   33
#endif


/*
 ************************************************************************
 *                                                                      *
 *   macro for throwing an exception                                    *
 *                                                                      *
 ************************************************************************
 */

static char exc_message[80];

#define THROW_EXCEPTION(STATUS, ROUTINE)                                \
    if        (STATUS ==             GEM_BADDREP                 ) {    \
        sprintf(exc_message, "%s failed: BadDRep",       #ROUTINE);     \
    } else if (STATUS ==             GEM_BADFACEID               ) {    \
        sprintf(exc_message, "%s failed: BadFaceID",     #ROUTINE);     \
    } else if (STATUS ==             GEM_BADBOUNDINDEX           ) {    \
        sprintf(exc_message, "%s failed: BadBoundIndex", #ROUTINE);     \
    } else if (STATUS ==             GEM_BADVSETINDEX            ) {    \
        sprintf(exc_message, "%s failed: BadVSetIndex",  #ROUTINE);     \
    } else if (STATUS ==             GEM_BADRANK                 ) {    \
        sprintf(exc_message, "%s failed: BadRank",       #ROUTINE);     \
    } else if (STATUS ==             GEM_BADDSETNAME             ) {    \
        sprintf(exc_message, "%s failed: BadSetName",    #ROUTINE);     \
    } else if (STATUS ==             GEM_MISSDISPLACE            ) {    \
        sprintf(exc_message, "%s failed: MissDisplace",  #ROUTINE);     \
    } else if (STATUS ==             GEM_NOTFOUND                ) {    \
        sprintf(exc_message, "%s failed: NotFound",      #ROUTINE);     \
    } else if (STATUS ==             GEM_BADMODEL                ) {    \
        sprintf(exc_message, "%s failed: BadModel",      #ROUTINE);     \
    } else if (STATUS ==             GEM_BADCONTEXT              ) {    \
        sprintf(exc_message, "%s failed: BadContext",    #ROUTINE);     \
    } else if (STATUS ==             GEM_BADBREP                 ) {    \
        sprintf(exc_message, "%s failed: BadBRep",       #ROUTINE);     \
    } else if (STATUS ==             GEM_BADINDEX                ) {    \
        sprintf(exc_message, "%s failed: BadIndex",      #ROUTINE);     \
    } else if (STATUS ==             GEM_NOTCHANGED              ) {    \
        sprintf(exc_message, "%s failed: NotChanged",    #ROUTINE);     \
    } else if (STATUS ==             GEM_ALLOC                   ) {    \
        sprintf(exc_message, "%s failed: Alloc",         #ROUTINE);     \
    } else if (STATUS ==             GEM_BADTYPE                 ) {    \
        sprintf(exc_message, "%s failed: BadType",       #ROUTINE);     \
    } else if (STATUS ==             GEM_NULLVALUE               ) {    \
        sprintf(exc_message, "%s failed: NullValue",     #ROUTINE);     \
    } else if (STATUS ==             GEM_NULLNAME                ) {    \
        sprintf(exc_message, "%s failed: NullName",      #ROUTINE);     \
    } else if (STATUS ==             GEM_NULLOBJ                 ) {    \
        sprintf(exc_message, "%s failed: NullObj",       #ROUTINE);     \
    } else if (STATUS ==             GEM_BADOBJECT               ) {    \
        sprintf(exc_message, "%s failed: BadObject",     #ROUTINE);     \
    } else if (STATUS ==             GEM_WIREBODY                ) {    \
        sprintf(exc_message, "%s failed: WireBody",      #ROUTINE);     \
    } else if (STATUS ==             GEM_SOLIDBODY               ) {    \
        sprintf(exc_message, "%s failed: SolidBody",     #ROUTINE);     \
    } else if (STATUS ==             GEM_NOTESSEL                ) {    \
        sprintf(exc_message, "%s failed: NotTessel",     #ROUTINE);     \
    } else if (STATUS ==             GEM_BADVALUE                ) {    \
        sprintf(exc_message, "%s failed: BadValue",      #ROUTINE);     \
    } else if (STATUS ==             GEM_DUPLICATE               ) {    \
        sprintf(exc_message, "%s failed: Duplicate",     #ROUTINE);     \
    } else if (STATUS ==             GEM_BADINVERSE              ) {    \
        sprintf(exc_message, "%s failed: BadInverse",    #ROUTINE);     \
    } else if (STATUS ==             GEM_NOTPARAMBND             ) {    \
        sprintf(exc_message, "%s failed: NotParamBnd",   #ROUTINE);     \
    } else if (STATUS ==             GEM_NOTCONNECT              ) {    \
        sprintf(exc_message, "%s failed: NotConnect",    #ROUTINE);     \
    } else if (STATUS ==             GEM_NOTPARMTRIC             ) {    \
        sprintf(exc_message, "%s failed: NotParmtric",   #ROUTINE);     \
    } else if (STATUS ==             GEM_READONLYERR             ) {    \
        sprintf(exc_message, "%s failed: ReadOnlyErr",   #ROUTINE);     \
    } else if (STATUS ==             GEM_FIXEDLEN                ) {    \
        sprintf(exc_message, "%s failed: FixedLen",      #ROUTINE);     \
    } else if (STATUS ==             GEM_ASSEMBLY                ) {    \
        sprintf(exc_message, "%s failed: Assembly",      #ROUTINE);     \
    } else if (STATUS ==             GEM_BADNAME                 ) {    \
        sprintf(exc_message, "%s failed: BadName",       #ROUTINE);     \
    } else if (STATUS ==             GEM_UNSUPPORTED             ) {    \
        sprintf(exc_message, "%s failed: Unsupported",   #ROUTINE);     \
    } else {                                                            \
        sprintf(exc_message, "%s failed: unknown error", #ROUTINE);     \
    }                                                                   \
    PyErr_SetString(PyExc_RuntimeError, exc_message);                   \
    return NULL;


/*
 ************************************************************************
 *                                                                      *
 *   routines used to validate gemObjects                               *
 *                                                                      *
 *   NOTE: we should probably use real Python objects, but this         *
 *         works for now (assuming that a long is at least as big       *
 *         as a void* --- which is checked in gem.initialize)           *
 *                                                                      *
 ************************************************************************
 */

#ifdef LONGLONG
#define LONG long long
#else
#define LONG long
#endif

static LONG   *gemObjects = NULL;
static int  numGemObjects = 0;
static int  maxGemObjects = 0;

static int addGemObject(LONG pointer)
{
    int i;

    if (pointer == 0) return 1;

    for (i = 0; i < numGemObjects; i++) {
        if (pointer == gemObjects[i]) return 1;
    }

    if (maxGemObjects == 0) {
        maxGemObjects = 25;
        gemObjects = (LONG*)malloc(maxGemObjects*sizeof(LONG));
        if (gemObjects == NULL) return 0;
    }

    if (numGemObjects >= maxGemObjects) {
        maxGemObjects += 25;
        gemObjects = (LONG*)realloc(gemObjects, maxGemObjects*sizeof(LONG));
        if (gemObjects == NULL) return 0;
    }

    gemObjects[numGemObjects] = pointer;
    numGemObjects++;
    return 1;
}

static void removeGemObject(LONG pointer)
{
    int i;

    if (pointer == 0) return;

    for (i = 0; i < numGemObjects; i++) {
        if (pointer == gemObjects[i]) {
            gemObjects[i] = 0;
        }
    }

    while (numGemObjects > 0) {
        if (gemObjects[numGemObjects-1] != 0) {
            return;
        } else {
            numGemObjects--;
        }
    }
}

static int checkGemObject(LONG pointer)
{
    int i;

    if (pointer == 0) return 0;

    for (i = 0; i < numGemObjects; i++) {
        if (pointer == gemObjects[i]) {
            return 1;
        }
    }

    return 0;
}


/*
 ************************************************************************
 *                                                                      *
 *   gemInitialize -- implement gem.initialize()                        *
 *                                                                      *
 ************************************************************************
 */

static PyObject *
gemInitialize(PyObject *self, PyObject *args)
{
    PyObject  *result;
    int       status;
    LONG      longContext;
    gemCntxt  *context;

    /* make sure that a LONG is at least as big as a void* */
    if (sizeof(LONG) < sizeof(void*)) {
        THROW_EXCEPTION(9999, gem.initialize);
    }

    /* validate the inputs */
    if (!PyArg_ParseTuple(args, "")) {
        PyErr_SetString(PyExc_TypeError, "bad args: has no arguments");
        return NULL;
    }

    /* execute the GEM call */
    status = gem_initialize(&context);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.initialize);
    }

    /* add the context to the gemObject list */
    longContext = (LONG)context;
    if (!addGemObject(longContext)) {
        THROW_EXCEPTION(GEM_ALLOC, gem.initialize);
    }

    /* return the result */
    result = Py_BuildValue("l", longContext);
    return result;
}


/*
 ************************************************************************
 *                                                                      *
 *   gemTerminate -- implement gem.terminate()                          *
 *                                                                      *
 ************************************************************************
 */

static PyObject *
gemTerminate(PyObject *self, PyObject *args)
{
    int       status, i;
    LONG      longContext;
    gemCntxt  *context;

    /* validate the inputs */
    if (!PyArg_ParseTuple(args, "l", &longContext)) {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"contextObj\"");
        return NULL;
    }

    if (checkGemObject(longContext)) {
        context = (gemCntxt*)longContext;
    } else {
        THROW_EXCEPTION(GEM_BADCONTEXT, gem.terminate);
    }

    /* execute the GEM call */
    status = gem_terminate(context);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.terminate);
    }

    /* remove the context from the gemObject list */
    removeGemObject(longContext);

    /* print out the remaining gemObjects */
    printf("numGemObjects = %d\n", numGemObjects);
    printf("maxGemObjects = %d\n", maxGemObjects);
    for (i = 0; i < numGemObjects; i++) {
        if (gemObjects[i] != 0) {
            printf("gemObjects[%3d] = %08lx remains\n", i, gemObjects[i]);
        }
    }

    /* return the result */
    Py_RETURN_NONE;
}


/*
 ************************************************************************
 *                                                                      *
 *   gemStaticModel -- implement gem.staticModel()                      *
 *                                                                      *
 ************************************************************************
 */

static PyObject *
gemStaticModel(PyObject *self, PyObject *args)
{
    PyObject  *result;
    int       status;
    LONG      longContext, longModel;
    gemCntxt  *context;
    gemModel  *model;

    /* validate the inputs */
    if (!PyArg_ParseTuple(args, "l", &longContext)) {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"contextObj\"");
        return NULL;
    }

    if (checkGemObject(longContext)) {
        context = (gemCntxt*)longContext;
    } else {
        THROW_EXCEPTION(GEM_BADCONTEXT, gem.staticModel);
    }

    /* execute the GEM call */
    status = gem_staticModel(context, &model);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.staticModel);
    }

    /* add the model to the gemObject list */
    longModel = (LONG)model;
    if (!addGemObject(longModel)) {
        THROW_EXCEPTION(GEM_ALLOC, gem.staticModel);
    }

    /* return the result */
    result = Py_BuildValue("l", longModel);
    return result;
}


/*
 ************************************************************************
 *                                                                      *
 *   gemLoadModel -- implement gem.loadModel()                          *
 *                                                                      *
 ************************************************************************
 */

static PyObject *
gemLoadModel(PyObject *self, PyObject *args)
{
    PyObject  *result;
    int       status, uptodate, nbrep, nparam, nbranch, nattr, i;
    LONG      longContext, longModel;
    char      *filename, *server, *filename2, *modeler;
    gemCntxt  *context;
    gemModel  *model;
    gemBRep   **breps;

    /* validate the inputs */
    if (!PyArg_ParseTuple(args, "ls", &longContext, &filename)) {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"contextObj filename\"");
        return NULL;
    }

    if (checkGemObject(longContext)) {
        context = (gemCntxt*)longContext;
    } else {
        THROW_EXCEPTION(GEM_BADCONTEXT, gem.loadModel);
    }

    /* execute the GEM call */
    status = gem_loadModel(context, NULL, filename, &model);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.loadModel);
    }

    /* add the model to the gemObject list */
    longModel = (LONG)model;
    if (!addGemObject(longModel)) {
        THROW_EXCEPTION(GEM_ALLOC, gem.loadModel);
    }

    /* add the BReps to the gemObject list */
    status = gem_getModel(model, &server, &filename2, &modeler,
                          &uptodate, &nbrep, &breps,
                          &nparam, &nbranch, &nattr);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.loadModel);
    }

    for (i = 0; i < nbrep; i++) {
        if (!addGemObject((LONG)(breps[i]))) {
            THROW_EXCEPTION(GEM_ALLOC, gem.loadModel);
        }
    }

    /* return the result */
    result = Py_BuildValue("l", longModel);
    return result;
}


/*
 ************************************************************************
 *                                                                      *
 *   gemGetBRepOwner -- implement gem.getBRepOwner()                    *
 *                                                                      *
 ************************************************************************
 */

static PyObject *
gemGetBRepOwner(PyObject *self, PyObject *args)
{
    PyObject  *result;
    int       status, instance, branch;
    LONG      longBRep;
    gemModel  *model;
    gemBRep   *brep;

    /* validate the inputs */
    if (!PyArg_ParseTuple(args, "l", &longBRep)) {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"brepObj\"");
        return NULL;
    }

    if (checkGemObject(longBRep)) {
        brep = (gemBRep*)longBRep;
    } else {
        THROW_EXCEPTION(GEM_BADBREP, gem.getBRepOwner);
    }

    /* execute the GEM call */
    status = gem_getBRepOwner(brep, &model, &instance, &branch);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.getBRepOwner);
    }

    /* return the result */
    result = PyTuple_New(3);
    PyTuple_SetItem(result, 0, Py_BuildValue("l", (LONG)model));
    PyTuple_SetItem(result, 1, Py_BuildValue("i", instance));
    PyTuple_SetItem(result, 2, Py_BuildValue("i", branch));
    return result;
}


/*
 ************************************************************************
 *                                                                      *
 *   gemSolidBoolean -- implement gem.solidBoolean()                    *
 *                                                                      *
 ************************************************************************
 */

static PyObject *
gemSolidBoolean(PyObject *self, PyObject *args)
{
    PyObject  *result;
    int       status, itype, uptodate, nbrep, nparam, nbranch, nattr, i;
    LONG      longBRep1, longBRep2, longModel;
    double    xform[12];
    char      *type, *server, *filename, *modeler;
    gemModel  *model;
    gemBRep   *brep1, *brep2, **breps;

    /* validate the inputs */
    xform[ 0] = 1; xform[ 1] = 0; xform[ 2] = 0; xform[ 3] = 0;
    xform[ 4] = 0; xform[ 5] = 1; xform[ 6] = 0; xform[ 7] = 0;
    xform[ 8] = 0; xform[ 9] = 0; xform[10] = 1; xform[11] = 0;

    if (!PyArg_ParseTuple(args, "lls|(dddddddddddd)", &longBRep1, &longBRep2, &type,
                          &(xform[0]), &(xform[ 1]), &(xform[ 2]), &(xform[ 3]),
                          &(xform[4]), &(xform[ 5]), &(xform[ 6]), &(xform[ 7]),
                          &(xform[8]), &(xform[ 9]), &(xform[10]), &(xform[11]))) {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"brepObj1 bepObj2 (xform)\"");
        return NULL;
    }

    if (checkGemObject(longBRep1)) {
        brep1 = (gemBRep*)longBRep1;
    } else {
        THROW_EXCEPTION(GEM_BADBREP, gem.solidBoolean);
    }

    if (checkGemObject(longBRep2)) {
        brep2 = (gemBRep*)longBRep2;
    } else {
        THROW_EXCEPTION(GEM_BADBREP, gem.solidBoolean);
    }

    if        (strcmp(type, "SUBTRACT") == 0) {
        itype = 0;
    } else if (strcmp(type, "INTERSECT") == 0) {
        itype = 1;
    } else if (strcmp(type, "UNION") == 0) {
        itype = 2;
    } else {
        THROW_EXCEPTION(GEM_BADVALUE, gem.solidBoolean);
    }

    /* execute the GEM call */
    status = gem_solidBoolean(brep1, brep2, xform, itype, &model);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.solidBoolean);
    }

    /* add the model to the gemObject list */
    longModel = (LONG)model;
    if (!addGemObject(longModel)) {
        THROW_EXCEPTION(GEM_ALLOC, gem.solidBoolean);
    }

    /* add the BReps to the gemObject list */
    status = gem_getModel(model, &server, &filename, &modeler,
                          &uptodate, &nbrep, &breps,
                          &nparam, &nbranch, &nattr);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.solidBoolean);
    }

    for (i = 0; i < nbrep; i++) {
        if (!addGemObject((LONG)(breps[i]))) {
            THROW_EXCEPTION(GEM_ALLOC, gem.solidBoolean);
        }
    }

    /* return the result */
    result = Py_BuildValue("l", longModel);
    return result;
}


/*
 ************************************************************************
 *                                                                      *
 *   gemGetAttribute -- implement gem.getAttribute()                    *
 *                                                                      *
 ************************************************************************
 */

static PyObject*
gemGetAttribute(PyObject* module, PyObject* args)
{
    PyObject  *result, *values_tuple;
    int       status, iotype, eindex, aindex, atype, alen, *integers, i;
    LONG      longObject;
    double    *reals;
    char      *otype, *aname, *string;
    void      *object;

    /* validate the inputs */
    if (!PyArg_ParseTuple(args, "lsii", &longObject, &otype, &eindex, &aindex)) {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"gemObj otype eindex aindex\"");
        return NULL;
    }

    if (checkGemObject(longObject)) {
        object = (void*)longObject;
    } else {
        THROW_EXCEPTION(GEM_BADOBJECT, gem.getAttribute);
    }

    if        (strcmp(otype, "CONTEXT") == 0) {
        iotype = GEM_MCONTEXT;
    } else if (strcmp(otype, "MODEL") == 0) {
        iotype = GEM_MODEL;
    } else if (strcmp(otype, "BRANCH") == 0) {
        iotype = GEM_BRANCH;
    } else if (strcmp(otype, "PARAM") == 0) {
        iotype = GEM_PARAM;
    } else if (strcmp(otype, "BREP") == 0) {
        iotype = GEM_BREP;
    } else if (strcmp(otype, "NODE") == 0) {
        iotype = GEM_NODE;
    } else if (strcmp(otype, "EDGE") == 0) {
        iotype = GEM_EDGE;
    } else if (strcmp(otype, "LOOP") == 0) {
        iotype = GEM_LOOP;
    } else if (strcmp(otype, "FACE") == 0) {
        iotype = GEM_FACE;
    } else if (strcmp(otype, "SHELL") == 0) {
        iotype = GEM_SHELL;
    } else {
        iotype = 0;
        THROW_EXCEPTION(GEM_BADVALUE, gem.getAttribute);
    }

    /* execute the GEM call */
    status = gem_getAttribute(object, iotype, eindex, aindex,
                              &aname, &atype, &alen,
                              &integers, &reals, &string);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.getAttribute);
    }

    /* build a tuple of the values */
    if (atype == GEM_INTEGER) {
        values_tuple = PyTuple_New(alen);
        for (i = 0; i < alen; i++) {
            PyTuple_SetItem(values_tuple, i, Py_BuildValue("i", integers[i]));
        }
    } else if (atype == GEM_REAL) {
        values_tuple = PyTuple_New(alen);
        for (i = 0; i < alen; i++) {
            PyTuple_SetItem(values_tuple, i, Py_BuildValue("d", reals[i]));
        }
    } else if (atype == GEM_STRING) {
        values_tuple = Py_BuildValue("s", string);
    } else {
        THROW_EXCEPTION(GEM_BADTYPE, gem.getAttribute);
    }

    /* return the result */
    result = PyTuple_New(2);
    PyTuple_SetItem(result, 0, Py_BuildValue("s", aname));
    PyTuple_SetItem(result, 1,                    values_tuple);
    return result;
}


/*
 ************************************************************************
 *                                                                      *
 *   gemRetAttribute -- implement gem.retAttribute()                    *
 *                                                                      *
 ************************************************************************
 */

static PyObject *
gemRetAttribute(PyObject *self, PyObject *args)
{
    PyObject  *result, *values_tuple;
    int       status, iotype, eindex, aindex, atype, alen, *integers, i;
    LONG      longObject;
    double    *reals;
    char      *otype, *aname, *string;
    void      *object;

    /* validate the inputs */
    if (!PyArg_ParseTuple(args, "lsis", &longObject, &otype, &eindex, &aname)) {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"gemObj atype eindex aname\"");
        return NULL;
    }

    if (checkGemObject(longObject)) {
        object = (void*)longObject;
    } else {
        THROW_EXCEPTION(GEM_BADOBJECT, gem.retAttribute);
    }

    if        (strcmp(otype, "CONTEXT") == 0) {
        iotype = GEM_MCONTEXT;
    } else if (strcmp(otype, "MODEL") == 0) {
        iotype = GEM_MODEL;
    } else if (strcmp(otype, "BRANCH") == 0) {
        iotype = GEM_BRANCH;
    } else if (strcmp(otype, "PARAM") == 0) {
        iotype = GEM_PARAM;
    } else if (strcmp(otype, "BREP") == 0) {
        iotype = GEM_BREP;
    } else if (strcmp(otype, "NODE") == 0) {
        iotype = GEM_NODE;
    } else if (strcmp(otype, "EDGE") == 0) {
        iotype = GEM_EDGE;
    } else if (strcmp(otype, "LOOP") == 0) {
        iotype = GEM_LOOP;
    } else if (strcmp(otype, "FACE") == 0) {
        iotype = GEM_FACE;
    } else if (strcmp(otype, "SHELL") == 0) {
        iotype = GEM_SHELL;
    } else {
        iotype = 0;
        THROW_EXCEPTION(GEM_BADVALUE, gem.retAttribute);
    }

    /* execute the GEM call */
    status = gem_retAttribute(object, iotype, eindex, aname,
                              &aindex, &atype, &alen,
                              &integers, &reals, &string);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.retAttribute);
    }

    /* build a tuple of the values */
    if (atype == GEM_INTEGER) {
        values_tuple = PyTuple_New(alen);
        for (i = 0; i < alen; i++) {
            PyTuple_SetItem(values_tuple, i, Py_BuildValue("i", integers[i]));
        }
    } else if (atype == GEM_REAL) {
        values_tuple = PyTuple_New(alen);
        for (i = 0; i < alen; i++) {
            PyTuple_SetItem(values_tuple, i, Py_BuildValue("d", reals[i]));
        }
    } else if (atype == GEM_STRING) {
        values_tuple = Py_BuildValue("s", string);
    } else {
        THROW_EXCEPTION(GEM_BADTYPE, gem.retAttribute);
    }

    /* return the result */
    result = PyTuple_New(2);
    PyTuple_SetItem(result, 0, Py_BuildValue("i", aindex));
    PyTuple_SetItem(result, 1,                    values_tuple);
    return result;
}


/*
 ************************************************************************
 *                                                                      *
 *   gemSetAttribute -- implement gem.setAttribute()                    *
 *                                                                      *
 ************************************************************************
 */

static PyObject*
gemSetAttribute(PyObject* module, PyObject* args)
{
    PyObject  *values, *item;
    int       status, iotype, eindex, alen, itype, *integers=NULL, i;
    long      ltemp;
    LONG      longObject;
    double    *reals=NULL;
    char      *otype, *aname, *string;
    void      *object;

    /* validate the inputs */
    if (!PyArg_ParseTuple(args, "lsisO", &longObject, &otype, &eindex, &aname, &values)) {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"gemObj otype eindex aname (values)\"");
        return NULL;
    }

    if (checkGemObject(longObject)) {
        object = (void*)longObject;
    } else {
        THROW_EXCEPTION(GEM_BADOBJECT, gem.setAttribute);
    }

    if        (strcmp(otype, "CONTEXT") == 0) {
        iotype = GEM_MCONTEXT;
    } else if (strcmp(otype, "MODEL") == 0) {
        iotype = GEM_MODEL;
    } else if (strcmp(otype, "BRANCH") == 0) {
        iotype = GEM_BRANCH;
    } else if (strcmp(otype, "PARAM") == 0) {
        iotype = GEM_PARAM;
    } else if (strcmp(otype, "BREP") == 0) {
        iotype = GEM_BREP;
    } else if (strcmp(otype, "NODE") == 0) {
        iotype = GEM_NODE;
    } else if (strcmp(otype, "EDGE") == 0) {
        iotype = GEM_EDGE;
    } else if (strcmp(otype, "LOOP") == 0) {
        iotype = GEM_LOOP;
    } else if (strcmp(otype, "FACE") == 0) {
        iotype = GEM_FACE;
    } else if (strcmp(otype, "SHELL") == 0) {
        iotype = GEM_SHELL;
    } else {
        iotype = 0;
        THROW_EXCEPTION(GEM_BADVALUE, gem.setAttribute);
    }

    /* determine the attribute type */
    if (PyTuple_Check(values)) {
        alen  = (int)PySequence_Length(values);
        itype = GEM_INTEGER;

        for (i = 0; i < alen; i++) {
            item = PyTuple_GetItem(values, i);
            if (!PyInt_Check(item)) {
                itype = GEM_REAL;
                break;
            }
        }
    } else if (PyString_Check(values)) {
        alen   = 1;
        itype  = GEM_STRING;
        string = PyString_AsString(values);
    } else {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"gemObj otype eindex aname (values)\"");
        return NULL;
    }

    /* execute the GEM call */
    if (itype == GEM_INTEGER) {
        integers = (int*) malloc(alen*sizeof(int));
        if (integers == NULL) {
            THROW_EXCEPTION(GEM_ALLOC, gem.setAttribute);
        }

        for (i = 0; i < alen; i++) {
            item = PyTuple_GetItem(values, i);
            if (PyInt_Check(item)) {
                integers[i] = (int)PyInt_AsLong(item);
            } else {
                if (integers != NULL) free(integers);
                THROW_EXCEPTION(GEM_BADVALUE, gem.setAttribute);
            }
        }

        status = gem_setAttribute(object, iotype, eindex, aname,
                                  GEM_INTEGER, alen, integers, NULL, NULL);
        if (status != GEM_SUCCESS) {
            if (integers != NULL) free(integers);
            THROW_EXCEPTION(status, gem.setAttribute);
        }

        if (integers != NULL) free(integers);
    } else if (itype == GEM_REAL) {
        reals = (double*) malloc(alen*sizeof(double));
        if (reals == NULL) {
            THROW_EXCEPTION(GEM_ALLOC, gem.setAttribute);
        }

        for (i = 0; i < alen; i++) {
            item = PyTuple_GetItem(values, i);
            if (PyFloat_Check(item)) {
                reals[i] = PyFloat_AsDouble(item);
            } else if (PyInt_Check(item)) {
                ltemp = PyInt_AsLong(item);
                reals[i] = (double)ltemp;
            } else {
                if (reals != NULL) free(reals);
                THROW_EXCEPTION(GEM_BADVALUE, gem.setAttribute);
            }
        }

        status = gem_setAttribute(object, iotype, eindex, aname,
                                  GEM_REAL, alen, NULL, reals, NULL);
        if (status != GEM_SUCCESS) {
            if (reals != NULL) free(reals);
            THROW_EXCEPTION(status, gem.setAttribute);
        }

        if (reals != NULL) free(reals);
    } else {
        string = PyString_AsString(values);

        status = gem_setAttribute(object, iotype, eindex, aname,
                                  GEM_STRING, alen, NULL, NULL, string);
        if (status != GEM_SUCCESS) {
            THROW_EXCEPTION(status, gem.setAttribute);
        }
    }

    /* return the result */
    Py_RETURN_NONE;
}


/*
 ************************************************************************
 *                                                                      *
 *   gemAdd2Model -- implement gem.add2Model()                          *
 *                                                                      *
 ************************************************************************
 */

static PyObject *
gemAdd2Model(PyObject *self, PyObject *args)
{
    int       status, uptodate, nbrep, nparam, nbranch, nattr;
    LONG      longModel, longBRep, longBRep2;
    double    xform[12];
    char      *type, *server, *filename, *modeler;
    gemModel  *model;
    gemBRep   *brep, **breps;

    /* validate the inputs */
    xform[ 0] = 1; xform[ 1] = 0; xform[ 2] = 0; xform[ 3] = 0;
    xform[ 4] = 0; xform[ 5] = 1; xform[ 6] = 0; xform[ 7] = 0;
    xform[ 8] = 0; xform[ 9] = 0; xform[10] = 1; xform[11] = 0;

    if (!PyArg_ParseTuple(args, "ll|(dddddddddddd)s", &longModel, &longBRep,
                          &(xform[0]), &(xform[ 1]), &(xform[ 2]), &(xform[ 3]),
                          &(xform[4]), &(xform[ 5]), &(xform[ 6]), &(xform[ 7]),
                          &(xform[8]), &(xform[ 9]), &(xform[10]), &(xform[11]),
                          &type)) {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"modelObj brepObj (xform)\"");
        return NULL;
    }

    if (checkGemObject(longModel)) {
        model = (gemModel*)longModel;
    } else {
        THROW_EXCEPTION(GEM_BADMODEL, gem.add2Model);
    }

    if (checkGemObject(longBRep)) {
        brep = (gemBRep*)longBRep;
    } else {
        THROW_EXCEPTION(GEM_BADBREP, gem.add2Model);
    }

    /* execute the GEM call */
    status = gem_add2Model(model, brep, xform);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.add2Model);
    }

    /* add the BReps just added to the gemObject list */
    status = gem_getModel(model, &server, &filename, &modeler,
                          &uptodate, &nbrep, &breps,
                          &nparam, &nbranch, &nattr);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.loadModel);
    }

    longBRep2 = (LONG)breps[nbrep-1];
    if (!addGemObject(longBRep2)) {
        THROW_EXCEPTION(GEM_ALLOC, gem.add2Model);
    }

    /* return the result */
    Py_RETURN_NONE;
}


/*
 ************************************************************************
 *                                                                      *
 *   gemSaveModel -- implement gem.saveModel()                          *
 *                                                                      *
 ************************************************************************
 */

static PyObject *
gemSaveModel(PyObject *self, PyObject *args)
{
    int       status;
    LONG      longModel;
    char      *filename;
    gemModel  *model;

    /* validate the inputs */
    if (!PyArg_ParseTuple(args, "ls", &longModel, &filename)) {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"modelObj filename\"");
        return NULL;
    }

    if (checkGemObject(longModel)) {
        model = (gemModel*)longModel;
    } else {
        THROW_EXCEPTION(GEM_BADMODEL, gem.saveModel);
    }

    /* execute the GEM call */
    status = gem_saveModel(model, filename);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.saveModel);
    }

    /* return the result */
    Py_RETURN_NONE;
}


/*
 ************************************************************************
 *                                                                      *
 *   gemReleaseModel -- implement gem.releaseModel()                    *
 *                                                                      *
 ************************************************************************
 */

static PyObject *
gemReleaseModel(PyObject *self, PyObject *args)
{
    int       status, uptodate, nbrep, nparam, nbranch, nattr, i;
    LONG      longModel;
    char      *server, *filename, *modeler;
    gemModel  *model;
    gemBRep   **breps;

    /* validate the inputs */
    if (!PyArg_ParseTuple(args, "l", &longModel)) {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"modelObj\"");
        return NULL;
    }

    if (checkGemObject(longModel)) {
        model = (gemModel*)longModel;
    } else {
        THROW_EXCEPTION(GEM_BADMODEL, gem.releaseModel);
    }

    /* remove the BReps from the gemObject list */
    status = gem_getModel(model, &server, &filename, &modeler,
                          &uptodate, &nbrep, &breps,
                          &nparam, &nbranch, &nattr);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.releaseModel);
    }

    for (i = 0; i < nbrep; i++) {
        removeGemObject((LONG)(breps[i]));
    }

    /* execute the GEM call */
    status = gem_releaseModel(model);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.releaseModel);
    }

    /* remove the model from the gemObject list */
    removeGemObject(longModel);

    /* return the result */
    Py_RETURN_NONE;
}


/*
 ************************************************************************
 *                                                                      *
 *   gemCopyModel -- implement gem.copyModel()                          *
 *                                                                      *
 ************************************************************************
 */

static PyObject *
gemCopyModel(PyObject *self, PyObject *args)
{
    PyObject  *result;
    int       status, uptodate, nbrep, nparam, nbranch, nattr, i;
    LONG      longModel1, longModel2;
    char      *server, *filename, *modeler;
    gemModel  *model1, *model2;
    gemBRep   **breps;

    /* validate the inputs */
    if (!PyArg_ParseTuple(args, "l", &longModel1)) {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"modelObj\"");
        return NULL;
    }

    if (checkGemObject(longModel1)) {
        model1 = (gemModel*)longModel1;
    } else {
        THROW_EXCEPTION(GEM_BADMODEL, gem.copyModel);
    }

    /* execute the GEM call */
    status = gem_copyModel(model1, &model2);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.copyModel);
    }

    /* add the model2 to the gemObject list */
    longModel2 = (LONG)model2;
    if (!addGemObject(longModel2)) {
        THROW_EXCEPTION(GEM_ALLOC, gem.copyModel);
    }

    /* add the BReps to the gemObject list */
    status = gem_getModel(model2, &server, &filename, &modeler,
                          &uptodate, &nbrep, &breps,
                          &nparam, &nbranch, &nattr);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.copyModel);
    }

    for (i = 0; i < nbrep; i++) {
        if (!addGemObject((LONG)(breps[i]))) {
            THROW_EXCEPTION(GEM_ALLOC, gem.copyModel);
        }
    }

    /* return the result */
    result = Py_BuildValue("l", longModel2);
    return result;
}


/*
 ************************************************************************
 *                                                                      *
 *   gemRegenModel -- implement gem.regenModel()                        *
 *                                                                      *
 ************************************************************************
 */

static PyObject *
gemRegenModel(PyObject *self, PyObject *args)
{
    int       status, uptodate, nbrep, nparam, nbranch, nattr, i;
    LONG      longModel;
    char      *server, *filename, *modeler;
    gemModel  *model;
    gemBRep   **breps;

    /* validate the inputs */
    if (!PyArg_ParseTuple(args, "l", &longModel)) {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"modelObj\"");
        return NULL;
    }

    if (checkGemObject(longModel)) {
        model = (gemModel*)longModel;
    } else {
        THROW_EXCEPTION(GEM_BADMODEL, gem.regenModel);
    }

    /* remove the old BReps from the gemObject list */
    status = gem_getModel(model, &server, &filename, &modeler,
                          &uptodate, &nbrep, &breps,
                          &nparam, &nbranch, &nattr);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.regenModel);
    }

    for (i = 0; i < nbrep; i++) {
        removeGemObject((LONG)(breps[i]));
    }

    /* execute the GEM call */
    status = gem_regenModel(model);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.regenModel);
    }

    /* add the new BReps to the gemObject list */
    status = gem_getModel(model, &server, &filename, &modeler,
                          &uptodate, &nbrep, &breps,
                          &nparam, &nbranch, &nattr);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.regenModel);
    }

    for (i = 0; i < nbrep; i++) {
        if (!addGemObject((LONG)(breps[i]))) {
            THROW_EXCEPTION(GEM_ALLOC, gem.regenModel);
        }
    }

    /* return the result */
    Py_RETURN_NONE;
}


/*
 ************************************************************************
 *                                                                      *
 *   gemGetModel -- implement gem.getModel()                            *
 *                                                                      *
 ************************************************************************
 */

static PyObject *
gemGetModel(PyObject *self, PyObject *args)
{
    PyObject  *result, *breps_tuple;
    int       status, uptodate, nbrep, nparam, nbranch, nattr, ibrep;
    LONG      longModel;
    char      *server, *filename, *modeler;
    gemModel  *model;
    gemBRep   **breps;

    /* validate the inputs */
    if (!PyArg_ParseTuple(args, "l", &longModel)) {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"modelObj\"");
        return NULL;
    }

    if (checkGemObject(longModel)) {
        model = (gemModel*)longModel;
    } else {
        THROW_EXCEPTION(GEM_BADMODEL, gem.getModel);
    }

    /* execute the GEM call */
    status = gem_getModel(model, &server, &filename, &modeler,
                          &uptodate, &nbrep, &breps,
                          &nparam, &nbranch, &nattr);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.getModel);
    }

    /* build a tuple of the BReps */
    breps_tuple = PyTuple_New(nbrep);
    for (ibrep = 0; ibrep < nbrep; ibrep++) {
        PyTuple_SetItem(breps_tuple, ibrep, Py_BuildValue("l", (LONG)(breps[ibrep])));
    }

    /* return the result */
    result = PyTuple_New(8);
    PyTuple_SetItem(result, 0, Py_BuildValue("s", server));
    PyTuple_SetItem(result, 1, Py_BuildValue("s", filename));
    PyTuple_SetItem(result, 2, Py_BuildValue("s", modeler));
    PyTuple_SetItem(result, 3, Py_BuildValue("i", uptodate));
    PyTuple_SetItem(result, 4,                    breps_tuple);
    PyTuple_SetItem(result, 5, Py_BuildValue("i", nparam));
    PyTuple_SetItem(result, 6, Py_BuildValue("i", nbranch));
    PyTuple_SetItem(result, 7, Py_BuildValue("i", nattr));
    return result;
}


/*
 ************************************************************************
 *                                                                      *
 *   gemGetBranch -- implement gem.getBranch()                          *
 *                                                                      *
 ************************************************************************
 */

static PyObject *
gemGetBranch(PyObject *self, PyObject *args)
{
    PyObject  *result, *parents_tuple, *childs_tuple;
    int       ibranch, status, suppress, nparent, *parents, nchild, *childs, nattr;
    int       iparent, ichild;
    LONG      longModel;
    char      *bname, *btype;
    gemModel  *model;

    /* validate the inputs */
    if (!PyArg_ParseTuple(args, "li", &longModel, &ibranch)) {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"modelObj ibranch\"");
        return NULL;
    }

    if (checkGemObject(longModel)) {
        model = (gemModel*)longModel;
    } else {
        THROW_EXCEPTION(GEM_BADMODEL, gem.getBranch);
    }

    /* execute the GEM call */
    status = gem_getBranch(model, ibranch, &bname, &btype, &suppress,
                           &nparent, &parents, &nchild, &childs, &nattr);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.getBranch);
    }

    /* build a tuple of the parents */
    parents_tuple = PyTuple_New(nparent);
    for (iparent = 0; iparent < nparent; iparent++) {
        PyTuple_SetItem(parents_tuple, iparent, Py_BuildValue("i", parents[iparent]));
    }

    /* build a tuple if the children */
    childs_tuple = PyTuple_New(nchild);
    for (ichild = 0; ichild < nchild; ichild++) {
        PyTuple_SetItem(childs_tuple, ichild, Py_BuildValue("i", childs[ichild]));
    }

    /* return the result */
    result = PyTuple_New(6);
    PyTuple_SetItem(result, 0, Py_BuildValue("s", bname));
    PyTuple_SetItem(result, 1, Py_BuildValue("s", btype));
    PyTuple_SetItem(result, 2, Py_BuildValue("i", suppress));
    PyTuple_SetItem(result, 3,                    parents_tuple);
    PyTuple_SetItem(result, 4,                    childs_tuple);
    PyTuple_SetItem(result, 5, Py_BuildValue("i", nattr));
    return result;
}


/*
 ************************************************************************
 *                                                                      *
 *   gemSetSuppress -- implement gem.setSuppress()                      *
 *                                                                      *
 ************************************************************************
 */

static PyObject *
gemSetSuppress(PyObject *self, PyObject *args)
{
    int       status, ibranch, istate;
    LONG      longModel;
    gemModel  *model;

    /* validate the inputs */
    if (!PyArg_ParseTuple(args, "lii", &longModel, &ibranch, &istate)) {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"modelObj ibranch istate\"");
        return NULL;
    }

    if (checkGemObject(longModel)) {
        model = (gemModel*)longModel;
    } else {
        THROW_EXCEPTION(GEM_BADMODEL, gem.setSuppress);
    }

    /* execute the GEM call */
    status = gem_setSuppress(model, ibranch, istate);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.setSuppress);
    }

    /* return the result */
    Py_RETURN_NONE;
}


/*
 ************************************************************************
 *                                                                      *
 *   gemGetParam -- implement gem.getParam()                            *
 *                                                                      *
 ************************************************************************
 */

static PyObject *
gemGetParam(PyObject *self, PyObject *args)
{
    PyObject  *result, *values_tuple;
    int       iparam, status, bflag, order, ptype, plen, *integers, nattr, i;
    LONG      longModel;
    double    *reals;
    char      *pname, *string;
    gemModel  *model;
    gemSpl    *spline;

    /* validate the inputs */
    if (!PyArg_ParseTuple(args, "li", &longModel, &iparam)) {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"modelObj iparam\"");
        return NULL;
    }

    if (checkGemObject(longModel)) {
        model = (gemModel*)longModel;
    } else {
        THROW_EXCEPTION(GEM_BADMODEL, gem.getParam);
    }

    /* execute the GEM call */
    status = gem_getParam(model, iparam, &pname, &bflag, &order,
                          &ptype, &plen, &integers, &reals,
                          &string, &spline, &nattr);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.getParam);
    }

    /* build a tuple of the values */
    if        (ptype == GEM_BOOL || ptype == GEM_INTEGER) {
        values_tuple = PyTuple_New(plen);
        for (i = 0; i < plen; i++) {
            PyTuple_SetItem(values_tuple, i, Py_BuildValue("i", integers[i]));
        }
    } else if (ptype == GEM_REAL) {
        values_tuple = PyTuple_New(plen);
        for (i = 0; i < plen; i++) {
            PyTuple_SetItem(values_tuple, i, Py_BuildValue("d", reals[i]));
        }
    } else if (ptype == GEM_STRING) {
        values_tuple = Py_BuildValue("s", string);
    } else {
        THROW_EXCEPTION(GEM_BADVALUE, gem.getParam);
    }

    /* return the result */
    result = PyTuple_New(5);
    PyTuple_SetItem(result, 0, Py_BuildValue("s", pname));
    PyTuple_SetItem(result, 1, Py_BuildValue("i", bflag));
    PyTuple_SetItem(result, 2, Py_BuildValue("i", order));
    PyTuple_SetItem(result, 3,                    values_tuple);
    PyTuple_SetItem(result, 4, Py_BuildValue("i", nattr));
    return result;
}


/*
 ************************************************************************
 *                                                                      *
 *   gemSetParam -- implement gem.setParam()                            *
 *                                                                      *
 ************************************************************************
 */

static PyObject *
gemSetParam(PyObject *self, PyObject *args)
{
    PyObject  *values, *item;
    int       status, iparam, itype, plen, *integers=NULL, i;
    long      ltemp;
    LONG      longModel;
    double    *reals=NULL;
    char      *string;
    gemModel  *model;

    /* validate the inputs */
    if (!PyArg_ParseTuple(args, "liO", &longModel, &iparam, &values)) {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"modelObj iparam (values)\"");
        return NULL;
    }

    if (checkGemObject(longModel)) {
        model = (gemModel*)longModel;
    } else {
        THROW_EXCEPTION(GEM_BADOBJECT, gem.setParam);
    }

    /* determine the attribute type */
    if (PyTuple_Check(values)) {
        plen  = (int)PySequence_Length(values);
        itype = GEM_INTEGER;

        for (i = 0; i < plen; i++) {
            item = PyTuple_GetItem(values, i);
            if (!PyInt_Check(item)) {
                itype = GEM_REAL;
                break;
            }
        }
    } else if (PyString_Check(values)) {
        plen   = 1;
        itype  = GEM_STRING;
        string = PyString_AsString(values);
    } else {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"gemObj otype eindex aname (values)\"");
        return NULL;
    }

    /* execute the GEM call */
    if (itype == GEM_INTEGER) {
        integers = (int*) malloc(plen*sizeof(int));
        if (integers == NULL) {
            THROW_EXCEPTION(GEM_ALLOC, gem.setParam);
        }

        for (i = 0; i < plen; i++) {
            item = PyTuple_GetItem(values, i);
            if (PyInt_Check(item)) {
                integers[i] = (int)PyInt_AsLong(item);
            } else {
                if (integers != NULL) free(integers);
                THROW_EXCEPTION(GEM_BADVALUE, gem.setParam);
            }
        }

        status = gem_setParam(model, iparam, plen, integers, NULL, NULL, NULL);
        if (status != GEM_SUCCESS) {
            if (integers != NULL) free(integers);
            THROW_EXCEPTION(status, gem.setParam);
        }

        if (integers != NULL) free(integers);
    } else if (itype == GEM_REAL) {
        reals = (double*) malloc(plen*sizeof(double));
        if (reals == NULL) {
            THROW_EXCEPTION(GEM_ALLOC, gem.setParam);
        }

        for (i = 0; i < plen; i++) {
            item = PyTuple_GetItem(values, i);
            if (PyFloat_Check(item)) {
                reals[i] = PyFloat_AsDouble(item);
            } else if (PyInt_Check(item)) {
                ltemp = PyInt_AsLong(item);
                reals[i] = (double)ltemp;
            } else {
                if (reals != NULL) free(reals);
                THROW_EXCEPTION(GEM_BADVALUE, gem.setParam);
            }
        }

        status = gem_setParam(model, iparam, plen, NULL, reals, NULL, NULL);
        if (status != GEM_SUCCESS) {
            if (reals != NULL) free(reals);
            THROW_EXCEPTION(status, gem.setParam);
        }

        if (reals != NULL) free(reals);
    } else {
        string = PyString_AsString(values);

        status = gem_setParam(model, iparam, plen, NULL, NULL, string, NULL);
        if (status != GEM_SUCCESS) {
            THROW_EXCEPTION(status, gem.setParam);
        }
    }

    /* return the result */
    Py_RETURN_NONE;
}


/*
 ************************************************************************
 *                                                                      *
 *   gemGetBRepInfo -- implement gem.getBRepInfo()                      *
 *                                                                      *
 ************************************************************************
 */

static PyObject *
gemGetBRepInfo(PyObject *self, PyObject *args)
{
    PyObject  *result;
    int       status, type, nnode, nedge, nloop, nface, nshell, nattr;
    LONG      longBRep;
    double    box[6];
    gemBRep   *brep;

    /* validate the inputs */
    if (!PyArg_ParseTuple(args, "l", &longBRep)) {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"brepObj\"");
        return NULL;
    }

    if (checkGemObject(longBRep)) {
        brep = (gemBRep*)longBRep;
    } else {
        THROW_EXCEPTION(GEM_BADBREP, gem.getBRepInfo);
    }

    /* execute the GEM call */
    status = gem_getBRepInfo(brep, box, &type, &nnode, &nedge, &nloop,
                             &nface, &nshell, &nattr);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.getBRepInfo);
    }

    /* return the result */
    result = PyTuple_New(8);
    PyTuple_SetItem(result, 0, Py_BuildValue("(dddddd)", box[0], box[1], box[2], box[3], box[4], box[5]));
    PyTuple_SetItem(result, 1, Py_BuildValue("i", type));
    PyTuple_SetItem(result, 2, Py_BuildValue("i", nnode));
    PyTuple_SetItem(result, 3, Py_BuildValue("i", nedge));
    PyTuple_SetItem(result, 4, Py_BuildValue("i", nloop));
    PyTuple_SetItem(result, 5, Py_BuildValue("i", nface));
    PyTuple_SetItem(result, 6, Py_BuildValue("i", nshell));
    PyTuple_SetItem(result, 7, Py_BuildValue("i", nattr));
    return result;
}


/*
 ************************************************************************
 *                                                                      *
 *   gemShell -- implement gem.getShell()                               *
 *                                                                      *
 ************************************************************************
 */

static PyObject *
gemGetShell(PyObject *self, PyObject *args)
{
    PyObject  *result, *faces_tuple;
    int       status, ishell, type, nface, *faces, nattr, i;
    LONG      longBRep;
    gemBRep   *brep;

    /* validate the inputs */
    if (!PyArg_ParseTuple(args, "li", &longBRep, &ishell)) {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"brepObj ishell\"");
        return NULL;
    }

    if (checkGemObject(longBRep)) {
        brep = (gemBRep*)longBRep;
    } else {
        THROW_EXCEPTION(GEM_BADBREP, gem.getShell);
    }

    /* execute the GEM call */
    status = gem_getShell(brep, ishell, &type, &nface, &faces, &nattr);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.getShell);
    }

    /* build a tuple of the Faces */
    faces_tuple = PyTuple_New(nface);
    for (i = 0; i < nface; i++) {
        PyTuple_SetItem(faces_tuple, i, Py_BuildValue("i", faces[i]));
    }

    /* return the result */
    result = PyTuple_New(3);
    PyTuple_SetItem(result, 0, Py_BuildValue("i", type));
    PyTuple_SetItem(result, 1,                    faces_tuple);
    PyTuple_SetItem(result, 2, Py_BuildValue("i", nattr));
    return result;
}


/*
 ************************************************************************
 *                                                                      *
 *   gemGetFace -- implement gem.getFace()                              *
 *                                                                      *
 ************************************************************************
 */

static PyObject *
gemGetFace(PyObject *self, PyObject *args)
{
    PyObject  *result, *loops_tuple;
    int       status, iface, norm, nloop, *loops, nattr, i;
    LONG      longBRep;
    double    uvbox[4];
    char      *ID;
    gemBRep   *brep;

    /* validate the inputs */
    if (!PyArg_ParseTuple(args, "li", &longBRep, &iface)) {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"brepObj iface\"");
        return NULL;
    }

    if (checkGemObject(longBRep)) {
        brep = (gemBRep*)longBRep;
    } else {
        THROW_EXCEPTION(GEM_BADBREP, gem.getFace);
    }

    /* execute the GEM call */
    status = gem_getFace(brep, iface, &ID, uvbox, &norm, &nloop, &loops, &nattr);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.getFace);
    }

    /* build a tuple of the Loops */
    loops_tuple = PyTuple_New(nloop);
    for (i = 0; i < nloop; i++) {
        PyTuple_SetItem(loops_tuple, i, Py_BuildValue("i", loops[i]));
    }

    /* return the result */
    result = PyTuple_New(5);
    PyTuple_SetItem(result, 0, Py_BuildValue("s", ID));
    PyTuple_SetItem(result, 1, Py_BuildValue("(dddd)", uvbox[0], uvbox[1], uvbox[2], uvbox[3]));
    PyTuple_SetItem(result, 2, Py_BuildValue("i", norm));
    PyTuple_SetItem(result, 3,                    loops_tuple);
    PyTuple_SetItem(result, 4, Py_BuildValue("i", nattr));
    return result;
}


/*
 ************************************************************************
 *                                                                      *
 *   gemGetWire -- implement gem.getWire()                              *
 *                                                                      *
 ************************************************************************
 */

static PyObject *
gemGetWire(PyObject *self, PyObject *args)
{
    PyObject  *result;
    int       status, nloop, *loops, i;
    LONG      longBRep;
    gemBRep   *brep;

    /* validate the inputs */
    if (!PyArg_ParseTuple(args, "l", &longBRep)) {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"brepObj\"");
        return NULL;
    }

    if (checkGemObject(longBRep)) {
        brep = (gemBRep*)longBRep;
    } else {
        THROW_EXCEPTION(GEM_BADBREP, gem.getWire);
    }

    /* execute the GEM call */
    status = gem_getWire(brep, &nloop, &loops);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.getWire);
    }

    /* return the result */
    result = PyTuple_New(nloop);
    for (i = 0; i < nloop; i++) {
        PyTuple_SetItem(result, i, Py_BuildValue("i", loops[i]));
    }
    return result;
}


/*
 ************************************************************************
 *                                                                      *
 *   gemGetLoop -- implement gem.getLoop()                              *
 *                                                                      *
 ************************************************************************
 */

static PyObject *
gemGetLoop(PyObject *self, PyObject *args)
{
    PyObject  *result, *edges_tuple;
    int       status, iloop, face, type, nedge, *edges, nattr, i;
    LONG      longBRep;
    gemBRep   *brep;

    /* validate the inputs */
    if (!PyArg_ParseTuple(args, "li", &longBRep, &iloop)) {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"brepObj iloop\"");
        return NULL;
    }

    if (checkGemObject(longBRep)) {
        brep = (gemBRep*)longBRep;
    } else {
        THROW_EXCEPTION(GEM_BADBREP, gem.getLoop);
    }

    /* execute the GEM call */
    status = gem_getLoop(brep, iloop, &face, &type, &nedge, &edges, &nattr);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.getLoop);
    }

    /* build a tuple of the Edges */
    edges_tuple = PyTuple_New(nedge);
    for (i = 0; i < nedge; i++) {
        PyTuple_SetItem(edges_tuple, i, Py_BuildValue("i", edges[i]));
    }

    /* return the result */
    result = PyTuple_New(4);
    PyTuple_SetItem(result, 0, Py_BuildValue("i", face));
    PyTuple_SetItem(result, 1, Py_BuildValue("i", type));
    PyTuple_SetItem(result, 2,                    edges_tuple);
    PyTuple_SetItem(result, 3, Py_BuildValue("i", nattr));
    return result;
}


/*
 ************************************************************************
 *                                                                      *
 *   gemGetEdge -- implement gem.getEdge()                              *
 *                                                                      *
 ************************************************************************
 */

static PyObject*
gemGetEdge(PyObject* module, PyObject* args)
{
    PyObject  *result;
    int       status, iedge, nodes[2], faces[2], nattr;
    LONG      longBRep;
    double    tlimit[2];
    gemBRep   *brep;

    /* validate the inputs */
    if (!PyArg_ParseTuple(args, "li", &longBRep, &iedge)) {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"brepObj iedge\"");
        return NULL;
    }

    if (checkGemObject(longBRep)) {
        brep = (gemBRep*)longBRep;
    } else {
        THROW_EXCEPTION(GEM_BADBREP, gem.getEdge);
    }

    /* execute the GEM call */
    status = gem_getEdge(brep, iedge, tlimit, nodes, faces, &nattr);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.getEdge);
    }

    /* return the result */
    result = PyTuple_New(4);
    PyTuple_SetItem(result, 0, Py_BuildValue("(dd)", tlimit[0], tlimit[1]));
    PyTuple_SetItem(result, 1, Py_BuildValue("(ii)", nodes[0], nodes[1]));
    PyTuple_SetItem(result, 2, Py_BuildValue("(ii)", faces[0], faces[1]));
    PyTuple_SetItem(result, 3, Py_BuildValue("i",    nattr));
    return result;
}


/*
 ************************************************************************
 *                                                                      *
 *   gemGetNode -- implement gem.getNode()                              *
 *                                                                      *
 ************************************************************************
 */

static PyObject *
gemGetNode(PyObject *self, PyObject *args)
{
    PyObject  *result;
    int       status, inode, nattr;
    LONG      longBRep;
    double    xyz[3];
    gemBRep   *brep;

    /* validate the inputs */
    if (!PyArg_ParseTuple(args, "li", &longBRep, &inode)) {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"brepObj inode\"");
        return NULL;
    }

    if (checkGemObject(longBRep)) {
        brep = (gemBRep*)longBRep;
    } else {
        THROW_EXCEPTION(GEM_BADBREP, gem.getNode);
    }

    /* execute the GEM call */
    status = gem_getNode(brep, inode, xyz, &nattr);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.getNode);
    }

    /* return the result */
    result = PyTuple_New(2);
    PyTuple_SetItem(result, 0, Py_BuildValue("(ddd)", xyz[0], xyz[1], xyz[2]));
    PyTuple_SetItem(result, 1, Py_BuildValue("i", nattr));
    return result;
}


/*
 ************************************************************************
 *                                                                      *
 *   gemGetMassProps -- implement gem.getMassProps()                    *
 *                                                                      *
 ************************************************************************
 */

static PyObject *
gemGetMassProps(PyObject *self, PyObject *args)
{
    PyObject  *result;
    int       status, eindex, itype;
    LONG      longBRep;
    double    props[14];
    char      *etype;
    gemBRep   *brep;

    /* validate the inputs */
    if (!PyArg_ParseTuple(args, "lsi", &longBRep, &etype, &eindex)) {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"brepObj etype eindex\"");
        return NULL;
    }

    if (checkGemObject(longBRep)) {
        brep = (gemBRep*)longBRep;
    } else {
        THROW_EXCEPTION(GEM_BADBREP, gem.getNode);
    }

    if        (strcmp(etype, "FACE") == 0) {
        itype = GEM_FACE;
    } else if (strcmp(etype, "SHELL") == 0) {
        itype = GEM_SHELL;
    } else if (strcmp(etype, "BREP") == 0) {
        itype = GEM_BREP;
    } else {
        THROW_EXCEPTION(GEM_BADVALUE, gem.getMassProps);
    }

    /* execute the GEM call */
    status = gem_getMassProps(brep, itype, eindex, props);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.getMassProps);
    }

    /* return the result */
    result = PyTuple_New(14);
    PyTuple_SetItem(result,  0, Py_BuildValue("d", props[ 0]));
    PyTuple_SetItem(result,  1, Py_BuildValue("d", props[ 1]));
    PyTuple_SetItem(result,  2, Py_BuildValue("d", props[ 2]));
    PyTuple_SetItem(result,  3, Py_BuildValue("d", props[ 3]));
    PyTuple_SetItem(result,  4, Py_BuildValue("d", props[ 4]));
    PyTuple_SetItem(result,  5, Py_BuildValue("d", props[ 5]));
    PyTuple_SetItem(result,  6, Py_BuildValue("d", props[ 6]));
    PyTuple_SetItem(result,  7, Py_BuildValue("d", props[ 7]));
    PyTuple_SetItem(result,  8, Py_BuildValue("d", props[ 8]));
    PyTuple_SetItem(result,  9, Py_BuildValue("d", props[ 9]));
    PyTuple_SetItem(result, 10, Py_BuildValue("d", props[10]));
    PyTuple_SetItem(result, 11, Py_BuildValue("d", props[11]));
    PyTuple_SetItem(result, 12, Py_BuildValue("d", props[12]));
    PyTuple_SetItem(result, 13, Py_BuildValue("d", props[13]));
    return result;
}


/*
 ************************************************************************
 *                                                                      *
 *   gemIsEquivalent -- implement gem.isEquivalent()                    *
 *                                                                      *
 ************************************************************************
 */

static PyObject *
gemIsEquivalent(PyObject *self, PyObject *args)
{
    PyObject  *result;
    int       status, eindex1, eindex2, itype;
    LONG      longBRep1, longBRep2;
    char      *type;
    gemBRep   *brep1, *brep2;

    /* validate the inputs */
    if (!PyArg_ParseTuple(args, "slili", &type, &longBRep1, &eindex1, &longBRep2, &eindex2)) {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"type bepObj1 eindex1 brepObj2 eindex2\"");
        return NULL;
    }

    if (checkGemObject(longBRep1)) {
        brep1 = (gemBRep*)longBRep1;
    } else {
        THROW_EXCEPTION(GEM_BADBREP, gem.isEquivalent);
    }

    if (checkGemObject(longBRep2)) {
        brep2 = (gemBRep*)longBRep2;
    } else {
        THROW_EXCEPTION(GEM_BADBREP, gem.isEquivalent);
    }

    if        (strcmp(type, "NODE") == 0) {
        itype = GEM_NODE;
    } else if (strcmp(type, "EDGE") == 0) {
        itype = GEM_EDGE;
    } else if (strcmp(type, "FACE") == 0) {
        itype = GEM_FACE;
    } else {
        THROW_EXCEPTION(GEM_BADVALUE, gem.isEquivalent);
    }

    /* execute the GEM call */
    status = gem_isEquivalent(itype, brep1, eindex1, brep2, eindex2);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.isEquivalent);
    }

    /* return the result */
    result = Py_BuildValue("i", status);
    return result;
}


/*
 ************************************************************************
 *                                                                      *
 *   gemNewDRep -- implement gem.newDRep()                              *
 *                                                                      *
 ************************************************************************
 */

static PyObject*
gemNewDRep(PyObject* module, PyObject* args)
{
    PyObject  *result;
    LONG      longModel, longDRep;
    int       status;
    gemModel  *model;
    gemDRep   *drep;

    /* validate the inputs */
    if (!PyArg_ParseTuple(args, "l", &longModel)) {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"modelObj\"");
        return NULL;
    }

    if (checkGemObject(longModel)) {
        model = (gemModel*)longModel;
    } else {
        THROW_EXCEPTION(GEM_BADMODEL, gem.newDRep);
    }

    /* make the GEM call */
    status = gem_newDRep(model, &drep);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.newDRep);
    }

    /* add the DRep to the gemObject list */
    longDRep = (LONG)drep;
    if (!addGemObject(longDRep)) {
        THROW_EXCEPTION(GEM_ALLOC, gem.newDRep);
    }

    /* return the result */
    result = Py_BuildValue("l", longDRep);
    return result;
}


/*
 ************************************************************************
 *                                                                      *
 *   gemTesselDRep -- implement gem.tesselDRep()                        *
 *                                                                      *
 ************************************************************************
 */

static PyObject*
gemTesselDRep(PyObject* module, PyObject* args)
{
    LONG      longDRep;
    int       status, ibrep;
    int       type, nnode, nedge, nloop, nface, nshell, nattr;
    int       uptodate, nbrep, nparam, nbranch, nbound, i;
    double    maxang, maxlen, maxsag, box[6], bbox[6], size;
    char      *filename, *server, *modeler;
    gemModel  *model;
    gemBRep   **breps;
    gemDRep   *drep;

    /* validate the inputs */
    if (!PyArg_ParseTuple(args, "liddd", &longDRep, &ibrep, &maxang, &maxlen, &maxsag)) {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"drepObj ibrep maxang maxlen maxsag\"");
        return NULL;
    }

    if (checkGemObject(longDRep)) {
        drep = (gemDRep*)longDRep;
    } else {
        THROW_EXCEPTION(GEM_BADDREP, gem.tesselDRep);
    }

    /* overwrite inputs if negative values are given */
    if (maxang <= 0 || maxlen <= 0 || maxsag <= 0) {
        status = gem_getDRepInfo(drep, &model, &nbound, &nattr);
        if (status != GEM_SUCCESS) {
            THROW_EXCEPTION(status, gem.tesselDRep);
        }

        box[0] = box[1] = box[2] = +1e20;
        box[3] = box[4] = box[5] = -1e20;

        status = gem_getModel(model, &server, &filename, &modeler, &uptodate,
                              &nbrep, &breps, &nparam, &nbranch, &nattr);
        if (status != GEM_SUCCESS) {
            THROW_EXCEPTION(status, gem.tesselDRep);
        }

        for (i = 1; i <= nbrep; i++) {
            if (ibrep == 0 || i == ibrep) {
                status = gem_getBRepInfo(breps[i-1], bbox, &type, &nnode,
                                         &nedge, &nloop, &nface, &nshell, &nattr);
                if (status != GEM_SUCCESS) {
                    THROW_EXCEPTION(status, gem.tesselDRep);
                }

                if (bbox[0] < box[0]) box[0] = bbox[0];
                if (bbox[1] < box[1]) box[1] = bbox[1];
                if (bbox[2] < box[2]) box[2] = bbox[2];
                if (bbox[3] > box[3]) box[3] = bbox[3];
                if (bbox[4] > box[4]) box[4] = bbox[4];
                if (bbox[5] > box[5]) box[5] = bbox[5];
            }
        }

        size = sqrt( (box[0]-box[3])*(box[0]-box[3])
                   + (box[1]-box[4])*(box[1]-box[4])
                   + (box[2]-box[5])*(box[2]-box[5]));

        if (maxang <= 0) {
            maxang = 15;
        }
        if        (maxlen == 0) {
            maxlen =  0.020  * size;
        } else if (maxlen <  0) {
            maxlen = -maxlen * size;
        }
        if        (maxsag == 0) {
            maxsag =  0.001  * size;
        } else if (maxsag <  0) {
            maxsag = -maxsag * size;
        }
    }

    /* execute the GEM call */
    status = gem_tesselDRep(drep, ibrep, maxang, maxlen, maxsag);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.tesselDRep);
    }

    /* return the result */
    Py_RETURN_NONE;
}


/*
 ************************************************************************
 *                                                                      *
 *   gemGetTessel -- implement gem.getTessel()                          *
 *                                                                      *
 ************************************************************************
 */

static PyObject*
gemGetTessel(PyObject* module, PyObject* args)
{
    PyObject  *result, *xyz_nd, *tri_nd;
    LONG      longDRep;
    int       status, ibrep, iface, npts, ntris, *tris;
    double    *xyz;
    gemDRep   *drep;
    gemPair   bface;
    int       rank;
    npy_intp  dims[2];

    /* validate the inputs */
    if (!PyArg_ParseTuple(args, "lii", &longDRep, &ibrep, &iface)) {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"drepObj ibep iface\"");
        return NULL;
    }

    if (checkGemObject(longDRep)) {
        drep = (gemDRep*)longDRep;
    } else {
        THROW_EXCEPTION(GEM_BADDREP, gem.getTessel);
    }

    /* execute the GEM call */
    bface.BRep  = ibrep;
    bface.index = iface;   /* iface */
    status = gem_getTessel(drep, bface, &ntris, &npts, &tris, &xyz);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.getTessel);
    }

    /* build the ndarrays */
    rank    = 2;
    dims[0] = npts;
    dims[1] = 3;
    xyz_nd  = PyArray_SimpleNew(rank, dims, NPY_DOUBLE);
    memcpy(((PyArrayObject*)(xyz_nd))->data, (void*)xyz, 3*npts*sizeof(double));

    rank    = 2;
    dims[0] = ntris;
    dims[1] = 3;
    tri_nd   = PyArray_SimpleNew(rank, dims, NPY_INT);
    memcpy(((PyArrayObject*)(tri_nd))->data, (void*)tris, 3*ntris*sizeof(int));

    /* return the result */
    result = PyTuple_New(2);
    PyTuple_SetItem(result, 0, tri_nd);
    PyTuple_SetItem(result, 1, xyz_nd);
    return result;
}


/*
 ************************************************************************
 *                                                                      *
 *   gemDestoryDRep -- implement gem.destroyDRep()                      *
 *                                                                      *
 ************************************************************************
 */

static PyObject*
gemDestroyDRep(PyObject* module, PyObject* args)
{
    LONG      longDRep;
    int       status;
    gemDRep   *drep;

    /* validate the inputs */
    if (!PyArg_ParseTuple(args, "l", &longDRep)) {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"drepObj\"");
        return NULL;
    }

    if (checkGemObject(longDRep)) {
        drep = (gemDRep*)longDRep;
    } else {
        THROW_EXCEPTION(GEM_BADDREP, gem.destroyDRep);
    }

    /* execute the GEM call */
    status = gem_destroyDRep(drep);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.destroyDRep);
    }

    /* remove the DRep from the gemObject list */
    removeGemObject(longDRep);

    /* return the result */
    Py_RETURN_NONE;
}


/*
 ************************************************************************
 *                                                                      *
 *   gemPlotDRep -- implement gem.plotDRep()                            *
 *                                                                      *
 ************************************************************************
 */

static PyObject*
gemPlotDRep(PyObject* module, PyObject* args)
{
    LONG      longDRep;

#ifdef GEM_GRAPHICS
    int       status, nbound, nattr, uptodate, nparam, nbranch, i;
    int       mtflag=0, keys[2] = {117, 118}, types[2] = {GV_SURF, GV_SURF};
    int       type, nnode, nedge, nloop, nface, nshell;
    float     lims[4], focus[4];
    double    box[6];
    char      titles[32] = {"U Parameter     V Parameter     "};
    char      *server, *filename, *modeler;
    gemModel  *model;
#endif

    /* validate the inputs */
    if (!PyArg_ParseTuple(args, "l", &longDRep)) {
        PyErr_SetString(PyExc_TypeError, "bad args: should be \"drepObj\"");
        return NULL;
    }

    if (!checkGemObject(longDRep)) {
        THROW_EXCEPTION(GEM_BADMODEL, gem.plotModel);
    }

#ifdef GEM_GRAPHICS
    gemDRep   *drep;

    drep = (gemDRep*)longDRep;

    /* store the plot Model */
    plotDRep = drep;

    /* set up and start gv */
    status = gem_getDRepInfo(drep, &model, &nbound, &nattr);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.plotDRep);
    }

    status = gem_getModel(model, &server, &filename, &modeler, &uptodate, &plotNbrep, &plotBReps, &nparam, &nbranch, &nattr);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.plotDRep);
    }

    plotNg = 3;
    plotBox[0] = plotBox[1] = plotBox[2] = +1.0e20;
    plotBox[3] = plotBox[4] = plotBox[5] = -1.0e20;

    for (i = 0; i < plotNbrep; i++) {
        status = gem_getBRepInfo(plotBReps[i], box, &type, &nnode, &nedge,
                                 &nloop, &nface, &nshell, &nattr);
        if (status != GEM_SUCCESS) {
            THROW_EXCEPTION(status, gem.plotDRep);
        }

        if (box[0] < plotBox[0]) plotBox[0] = box[0];
        if (box[1] < plotBox[1]) plotBox[1] = box[1];
        if (box[2] < plotBox[2]) plotBox[2] = box[2];
        if (box[3] > plotBox[3]) plotBox[3] = box[3];
        if (box[4] > plotBox[4]) plotBox[4] = box[4];
        if (box[5] > plotBox[5]) plotBox[5] = box[5];

        plotNg += nedge + nface;
    }

    focus[0] = (float)(0.5 * (plotBox[0]+plotBox[3]));
    focus[1] = (float)(0.5 * (plotBox[1]+plotBox[4]));
    focus[2] = (float)(0.5 * (plotBox[2]+plotBox[5]));
    focus[3] = (float)sqrt( (plotBox[0]-plotBox[3])*(plotBox[0]-plotBox[3])
                           +(plotBox[1]-plotBox[4])*(plotBox[1]-plotBox[4])
                           +(plotBox[2]-plotBox[5])*(plotBox[2]-plotBox[5]) );

    lims[0] = (float)0.0;
    lims[1] = (float)2.0;
    lims[2] = (float)0.0;
    lims[3] = (float)2.0;

    status = gv_init("           gem.plotDRep()", mtflag, 2, keys, types, lims, titles, focus);
    if (status != GEM_SUCCESS) {
        THROW_EXCEPTION(status, gem.plotDRep);
    }

    /* reset plotDRep since it is not needed any more */
    plotDRep = NULL;
#endif

    /* return the result */
    Py_RETURN_NONE;
}

/*
 ************************************************************************
 *                                                                      *
 *   getEdge -- return an array of points along an edge                 *
 *                                                                      *
 ************************************************************************
 */
#ifdef GEM_GRAPHICS
int
getEdge(gemBRep *brep, int iedge, int *npts, double xyz[])
{
    int    nodes[2], faces[2], nattr, i;
    double tlimit[2], frac;

    /* this routine simply creates 11 points between the beginning
       and end of the Edge, WHICH IS WRONG.  this will have to be fixed
       when the design of the drep is revisited */
    *npts = 11;

    gem_getEdge(brep, iedge, tlimit, nodes, faces, &nattr);

    gem_getNode(brep, nodes[0], &xyz[ 0], &nattr);
    gem_getNode(brep, nodes[1], &xyz[30], &nattr);

    for (i = 1; i < *npts-1; i++) {
        frac = (double)(i) / (double)(*npts-1);

        xyz[3*i  ] = (1-frac) * xyz[0] + frac * xyz[30];
        xyz[3*i+1] = (1-frac) * xyz[1] + frac * xyz[31];
        xyz[3*i+2] = (1-frac) * xyz[2] + frac * xyz[32];
    }

    return GEM_SUCCESS;
}
#endif

/*
 ************************************************************************
 *                                                                      *
 *   gvupdate -- used for single process operation of gv to change data *
 *                                                                      *
 ************************************************************************
 */
#ifdef GEM_GRAPHICS
int
gvupdate(void )
{
    new_data = 0;

    return plotNg;
}
#endif

/*
 ************************************************************************
 *                                                                      *
 *   gvdata -- used to (re)set the graphics objects to be used in plotting *
 *                                                                      *
 ************************************************************************
 */
#ifdef GEM_GRAPHICS
void
gvdata( int ngraphics, GvGraphic* graphic[] )
{
    GvColor  color;
    GvObject *object;
    int      status, nnode, nedge, nloop, nface, nshell, nattr;
    int      i, ibrep, iedge, iface, m, type, npts, attr, mask, utype;
    double   box[6], *points,*uvf;
    double   xyz[3000];      // needed because getEdge is not written correctly
    char     title[16], brepName[16];
    gemPair  bface;
    gemConn  *conn;

    plotList = graphic;
    i        = 0;

    /* if the family does not exist, create it */
    if (gv_getfamily(  "Axes", 1, &attr) == -1) {
        gv_allocfamily("Axes");
    }

    /* x-axis */
    mask = GV_FOREGROUND | GV_FORWARD | GV_ORIENTATION;

    color.red   = (float)1.0;
    color.green = (float)0.0;
    color.blue  = (float)0.0;

    sprintf(title, "X axis");
    utype = 999;
    graphic[i] = gv_alloc(GV_NONINDEXED, GV_POLYLINES, mask, color, title, utype, 0);
    if (graphic[i] != NULL) {
        graphic[i]->number    = 1;
        graphic[i]->lineWidth = 3;

        graphic[i]->fdata = (float*) malloc(6*sizeof(float));
        graphic[i]->fdata[0] = 0;
        graphic[i]->fdata[1] = 0;
        graphic[i]->fdata[2] = 0;
        graphic[i]->fdata[3] = 1;
        graphic[i]->fdata[4] = 0;
        graphic[i]->fdata[5] = 0;
        graphic[i]->object->length = 1;
        graphic[i]->object->type.plines.len = (int*) malloc(sizeof(int));
        graphic[i]->object->type.plines.len[0] = 2;

        gv_adopt("Axes", graphic[i]);

        i++;
    }

    /* y-axis */
    mask = GV_FOREGROUND | GV_FORWARD | GV_ORIENTATION;

    color.red   = (float)0.0;
    color.green = (float)1.0;
    color.blue  = (float)0.0;

    sprintf(title, "Y axis");
    utype = 999;
    graphic[i] = gv_alloc(GV_NONINDEXED, GV_POLYLINES, mask, color, title, utype, 0);
    if (graphic[i] != NULL) {
        graphic[i]->number    = 1;
        graphic[i]->lineWidth = 3;

        graphic[i]->fdata = (float*) malloc(6*sizeof(float));
        graphic[i]->fdata[0] = 0;
        graphic[i]->fdata[1] = 0;
        graphic[i]->fdata[2] = 0;
        graphic[i]->fdata[3] = 0;
        graphic[i]->fdata[4] = 1;
        graphic[i]->fdata[5] = 0;
        graphic[i]->object->length = 1;
        graphic[i]->object->type.plines.len = (int*) malloc(sizeof(int));
        graphic[i]->object->type.plines.len[0] = 2;

        gv_adopt("Axes", graphic[i]);

        i++;
    }

    /* z-axis */
    mask = GV_FOREGROUND | GV_FORWARD | GV_ORIENTATION;

    color.red   = (float)0.0;
    color.green = (float)0.0;
    color.blue  = (float)1.0;

    sprintf(title, "Z axis");
    utype = 999;
    graphic[i] = gv_alloc(GV_NONINDEXED, GV_POLYLINES, mask, color, title, utype, 0);
    if (graphic[i] != NULL) {
        graphic[i]->number    = 1;
        graphic[i]->lineWidth = 3;
        graphic[i]->fdata = (float*) malloc(6*sizeof(float));
        graphic[i]->fdata[0] = 0;
        graphic[i]->fdata[1] = 0;
        graphic[i]->fdata[2] = 0;
        graphic[i]->fdata[3] = 0;
        graphic[i]->fdata[4] = 0;
        graphic[i]->fdata[5] = 1;
        graphic[i]->object->length = 1;
        graphic[i]->object->type.plines.len = (int*) malloc(sizeof(int));
        graphic[i]->object->type.plines.len[0] = 2;

        gv_adopt("Axes", graphic[i]);

        i++;
    }

    /* BReps */
    for (ibrep = 1; ibrep <= plotNbrep; ibrep++) {
        bface.BRep = ibrep;
        status = gem_getBRepInfo(plotBReps[ibrep-1], box, &type, &nnode, &nedge,
                                 &nloop, &nface, &nshell, &nattr);
        if (status != GEM_SUCCESS) continue;

        /* if the family does not exist, create it */
        sprintf(brepName, "BRep %d", ibrep);

        if (gv_getfamily( brepName, 1, &attr) == -1){
            gv_allocfamily(brepName);
        }

        /* create a graphic object for each Edge (in ivol) */
        for (iedge = 1; iedge <= nedge; iedge++) {

            /* get the Edge info */
            status = getEdge(plotBReps[ibrep-1], iedge, &npts, xyz);
            if (status != GEM_SUCCESS) {
                printf("gemEdge(ibrep=%d, iedge=%d) -> status = %d\n", ibrep, iedge, status);
            }

            /* set up the new graphic object */
            mask = GV_FOREGROUND | GV_FORWARD;
            color.red   = (float)0.0;
            color.green = (float)0.0;
            color.blue  = (float)1.0;

            sprintf(title, "Edge %d", iedge);
            utype = 1 + 10 * ibrep;
            graphic[i] = gv_alloc(GV_NONINDEXED, GV_POLYLINES, mask, color, title, utype, iedge);

            if (graphic[i] != NULL) {
                graphic[i]->number     = 1;
                graphic[i]->lineWidth  = 2;
                graphic[i]->pointSize  = 3;
                graphic[i]->mesh.red   = 0;
                graphic[i]->mesh.green = 0;
                graphic[i]->mesh.blue  = 0;

                /* load the data */
                graphic[i]->fdata = (float*) malloc(3*npts*sizeof(float));
                for (m = 0; m < 3*npts; m++) {
                    graphic[i]->fdata[m] = (float)xyz[m];
                }

                object = graphic[i]->object;
                object->length = 1;
                object->type.plines.len = (int *)   malloc(sizeof(int));
                object->type.plines.len[0] = npts;

                gv_adopt(brepName, graphic[i]);
            }
            i++;
        }

        /* create a graphic object for each Face */
        for (iface = 1; iface <= nface; iface++) {

            /* get the Face info */
            bface.index = iface;
            status = gem_getTessel(plotDRep, bface, &npts, &points, &uvf, &conn);
            if (status != GEM_SUCCESS) {
                printf(" BRep #%d: gem_getTessel status = %d\n", ibrep, status);
            }

            /* set up new graphic object */
            mask        = GV_FOREGROUND | GV_ORIENTATION;
            color.red   = (float)1.0;
            color.green = (float)0.0;
            color.blue  = (float)0.0;

            if (plotNbrep > 1) color.green = (float)(ibrep-1) / (float)(plotNbrep-1);

            sprintf(title, "Face %d ", iface);
            utype = 2 * 10 * ibrep;
            graphic[i] = gv_alloc(GV_INDEXED, GV_DISJOINTTRIANGLES, mask,
                                  color, title, utype, iface);

            if (graphic[i] != NULL) {
                graphic[i]->number     = 1;
                graphic[i]->back.red   = (float)0.5;
                graphic[i]->back.green = (float)0.5;
                graphic[i]->back.blue  = (float)0.5;

                /* load the data */
                object = graphic[i]->object;
                if ((npts <= 0) || (conn->nTris <= 0)) {
                    object->length = 0;
                } else {
                    graphic[i]->ddata = points;
                    object->length = conn->nTris;
                    object->type.distris.index = (int *) malloc(3*conn->nTris*sizeof(int));
                    if (object->type.distris.index != NULL) {
                        for (m = 0; m < 3*conn->nTris; m++) {
                            object->type.distris.index[m] = conn->Tris[m]-1;
                        }
                    }
                }

                gv_adopt(brepName, graphic[i]);
            }
            i++;
        }
    }
}
#endif

/*
 ************************************************************************
 *                                                                      *
 *   gvscalar -- get scalar value for color rendering for graphics objects *
 *                                                                      *
 ************************************************************************
 */
#ifdef GEM_GRAPHICS
int
gvscalar( int key, GvGraphic* graphic, int len, float* scalar )
{
    int     i, status, npts;
    double  *points, *uv;
    gemPair bface;
    gemConn *conn;

    bface.BRep  = graphic->utype;
    bface.index = graphic->uindex;
    status = gem_getTessel(plotDRep, bface, &npts, &points, &uv, &conn);
    if (status != GEM_SUCCESS) {
        printf(" BRep #%d: gem_getTessel = %d\n", bface.BRep, status);
    }

    if (npts != len) {
        printf(" BRep#%d/Face #%d: Length mismatch in gvscalar: len=%d, npts=%d\n",
               bface.BRep, bface.index, len, npts);
        return 0;
    }

    if (key == 0) {
        for (i = 0; i < len; i++) {
            scalar[i] = (float)uv[2*i  ];
        }
    } else {
        for (i = 0; i < len; i++) {
            scalar[i] = (float)uv[2*i+1];
        }
    }
    return 1;
}
#endif

/*
 ************************************************************************
 *                                                                      *
 *   gvevent -- process graphic callbacks                               *
 *                                                                      *
 ************************************************************************
 */
#ifdef GEM_GRAPHICS
void
gvevent(int       *win,                 /* (in)  window of event */
        int       *type,                /* (in)  type of event */
/*@unused@*/int   *xpix,                /* (in)  x-pixel location of event */
/*@unused@*/int   *ypix,                /* (in)  y-pixel location of event */
        int       *state)               /* (in)  aditional event info */
{
    FILE      *fp=NULL;
    char      jnlName[255], tempName[255];

    /* repeat as long as we are reading a script (or once if
       not reading a script) */
    do {

        /* get the next script line if we are reading a script (and insert
           a '$' if we have detected an EOF) */

        if (script != NULL) {
            if (fscanf(script, "%1s", (char*)state) != 1) {
                *state = '$';
            }
            *win  = ThreeD;
            *type = KeyPress;
        }

        if ((*win == ThreeD) && (*type == KeyPress)) {
            if (*state == '\0') {

            /* 'x' - look from +X direction */
            } else if (*state == 'x') {
                double size;
                size = 0.5 * sqrt(  pow(plotBox[3] - plotBox[0], 2)
                                  + pow(plotBox[4] - plotBox[1], 2)
                                  + pow(plotBox[5] - plotBox[2], 2));

                gv_xform[0][0] =  0;
                gv_xform[1][0] =  0;
                gv_xform[2][0] = (float)(-1 / size);
                gv_xform[3][0] = (float)(+(plotBox[2] + plotBox[5]) / 2 / size);
                gv_xform[0][1] =  0;
                gv_xform[1][1] = (float)(+1 / size);
                gv_xform[2][1] =  0;
                gv_xform[3][1] = (float)(-(plotBox[1] + plotBox[4]) / 2 / size);
                gv_xform[0][2] = (float)(+1 / size);
                gv_xform[1][2] =  0;
                gv_xform[2][2] =  0;
                gv_xform[3][2] = (float)(-(plotBox[0] + plotBox[3]) / 2 / size);
                gv_xform[0][3] =  0;
                gv_xform[1][3] =  0;
                gv_xform[2][3] =  0;
                gv_xform[3][3] =  1;

                numarg   = 0;
                new_data = 1;

            /* 'y' - look from +Y direction */
            } else if (*state == 'y') {
                double size;
                size = 0.5 * sqrt(  pow(plotBox[3] - plotBox[0], 2)
                                  + pow(plotBox[4] - plotBox[1], 2)
                                  + pow(plotBox[5] - plotBox[2], 2));

                gv_xform[0][0] = (float)(+1 / size);
                gv_xform[1][0] =  0;
                gv_xform[2][0] =  0;
                gv_xform[3][0] = (float)(-(plotBox[0] + plotBox[3]) / 2 / size);
                gv_xform[0][1] =  0;
                gv_xform[1][1] =  0;
                gv_xform[2][1] = (float)(-1 / size);
                gv_xform[3][1] = (float)(+(plotBox[2] + plotBox[5]) / 2 / size);
                gv_xform[0][2] =  0;
                gv_xform[1][2] = (float)(+1 / size);
                gv_xform[2][2] =  0;
                gv_xform[3][2] = (float)(-(plotBox[1] + plotBox[4]) / 2 / size);
                gv_xform[0][3] =  0;
                gv_xform[1][3] =  0;
                gv_xform[2][3] =  0;
                gv_xform[3][3] =  1;

                numarg   = 0;
                new_data = 1;

            /* 'z' - look from +Z direction */
            } else if (*state == 'z') {
                double size;
                size = 0.5 * sqrt(  pow(plotBox[3] - plotBox[0], 2)
                                  + pow(plotBox[4] - plotBox[1], 2)
                                  + pow(plotBox[5] - plotBox[2], 2));

                gv_xform[0][0] = (float)(+1 / size);
                gv_xform[1][0] =  0;
                gv_xform[2][0] =  0;
                gv_xform[3][0] = (float)(-(plotBox[0] + plotBox[3]) / 2 / size);
                gv_xform[0][1] =  0;
                gv_xform[1][1] = (float)(+1 / size);
                gv_xform[2][1] =  0;
                gv_xform[3][1] = (float)(-(plotBox[1] + plotBox[4]) / 2 / size);
                gv_xform[0][2] =  0;
                gv_xform[1][2] =  0;
                gv_xform[2][2] = (float)(+1 / size);
                gv_xform[3][2] = (float)(-(plotBox[2] + plotBox[5]) / 2 / size);
                gv_xform[0][3] =  0;
                gv_xform[1][3] =  0;
                gv_xform[2][3] =  0;
                gv_xform[3][3] =  1;

                numarg   = 0;
                new_data = 1;

            /* '0' - append "0" to numarg */
            } else if (*state == '0') {
                numarg = 0 + numarg * 10;
                printf("numarg = %d\n", numarg);

            /* '1' - append "1" to numarg */
            } else if (*state == '1') {
                numarg = 1 + numarg * 10;
                printf("numarg = %d\n", numarg);

            /* '2' - append "2" to numarg */
            } else if (*state == '2') {
                numarg = 2 + numarg * 10;
                printf("numarg = %d\n", numarg);

            /* '3' - append "3" to numarg */
            } else if (*state == '3') {
                numarg = 3 + numarg * 10;
                printf("numarg = %d\n", numarg);

            /* '4' - append "4" to numarg */
            } else if (*state == '4') {
                numarg = 4 + numarg * 10;
                printf("numarg = %d\n", numarg);

            /* '5' - append "5" to numarg */
            } else if (*state == '5') {
                numarg = 5 + numarg * 10;
                printf("numarg = %d\n", numarg);

            /* '6' - append "6" to numarg */
            } else if (*state == '6') {
                numarg = 6 + numarg * 10;
                printf("numarg = %d\n", numarg);

            /* '7' - append "7" to numarg */
            } else if (*state == '7') {
                numarg = 7 + numarg * 10;
                printf("numarg = %d\n", numarg);

            /* '8' - append "8" to numarg */
            } else if (*state == '8') {
                numarg = 8 + numarg * 10;
                printf("numarg = %d\n", numarg);

            /* '9' - append "9" to numarg */
            } else if (*state == '9') {
                numarg = 9 + numarg * 10;
                printf("numarg = %d\n", numarg);

           /* 'bksp' - erase last digit of numarg */
            } else if (*state == 65288) {
                numarg = numarg / 10;
                printf("numarg = %d\n", numarg);

            /* '>' - write viewpoint */
            } else if (*state == '>') {
                sprintf(tempName, "ViewMatrix%d.dat", numarg);
                fp = fopen(tempName, "w");
                fprintf(fp, "%f %f %f %f\n", gv_xform[0][0], gv_xform[1][0],
                                             gv_xform[2][0], gv_xform[3][0]);
                fprintf(fp, "%f %f %f %f\n", gv_xform[0][1], gv_xform[1][1],
                                             gv_xform[2][1], gv_xform[3][1]);
                fprintf(fp, "%f %f %f %f\n", gv_xform[0][2], gv_xform[1][2],
                                             gv_xform[2][2], gv_xform[3][2]);
                fprintf(fp, "%f %f %f %f\n", gv_xform[0][3], gv_xform[1][3],
                                             gv_xform[2][3], gv_xform[3][3]);
                fclose(fp);

                printf("%s has been saved\n", tempName);

                numarg = 0;

            /* '<' - read viewpoint */
            } else if (*state == '<') {
                sprintf(tempName, "ViewMatrix%d.dat", numarg);
                fp = fopen(tempName, "r");
                if (fp != NULL) {
                    printf("resetting to %s\n", tempName);

                    fscanf(fp, "%f%f%f%f", &(gv_xform[0][0]), &(gv_xform[1][0]),
                                           &(gv_xform[2][0]), &(gv_xform[3][0]));
                    fscanf(fp, "%f%f%f%f", &(gv_xform[0][1]), &(gv_xform[1][1]),
                                           &(gv_xform[2][1]), &(gv_xform[3][1]));
                    fscanf(fp, "%f%f%f%f", &(gv_xform[0][2]), &(gv_xform[1][2]),
                                           &(gv_xform[2][2]), &(gv_xform[3][2]));
                    fscanf(fp, "%f%f%f%f", &(gv_xform[0][3]), &(gv_xform[1][3]),
                                           &(gv_xform[2][3]), &(gv_xform[3][3]));
                    fclose(fp);
                } else {
                    printf("%s does not exist\n", tempName);
                }

                numarg   = 0;
                new_data = 1;

            /* '$' - read journal file */
            } else if (*state == '$') {
                printf("--> Option $ chosen (read journal file)\n");

                if (script == NULL) {
                    printf("Enter journal filename: \n");
                    scanf("%s", jnlName);

                    printf("Opening journal file \"%s\" ...", jnlName);

                    script = fopen(jnlName, "r");
                    if (script != NULL) {
                        printf("okay\n");
                    } else {
                        printf("ERROR detected\n");
                    }
                } else {
                    fclose(script);
                    printf("Closing journal file\n");

                    script = NULL;
                    *win   =    0;
                }

            /* '?' - help */
            } else if (*state == '?') {
                printf(" 0-9 build numeric arg (#) \n");
                printf("BKSP edit  numeric arg (#) \n");
                printf("                           \n");
                printf("   x view from +x direction\n");
                printf("   y view from +y direction\n");
                printf("   z view from +z direction\n");
                printf("   > write viewpoint (#)   \n");
                printf("   < read  viewpoint (#)   \n");
                printf("   $ read journal file     \n");
                printf("   ? help                  \n");
                printf(" ESC exit                  \n");

            /* 'ESC' - exit program */
            } else if (*state ==65307) {

            }

            continue;
        }

        /* repeat as long as we are in a script */
    } while (script != NULL);
}
#endif


/*
 ************************************************************************
 *                                                                      *
 *   initgem -- routine called by Python when "gem" is imported         *
 *                                                                      *
 ************************************************************************
 */

static PyMethodDef
GemMethods[] = {
    // routines defined in gem.h
    {"initialize",    gemInitialize,    METH_VARARGS, "Initialize GEM\n\n\
                                                       Input arguments:\n\
                                                       \t  <none>      \n\
                                                       Returns:        \n\
                                                       \t  contextObj  "},
    {"terminate",     gemTerminate,     METH_VARARGS, "Terminate GEM\n\n\
                                                       Input arguments:\n\
                                                       \t  contextObj  \n\
                                                       Returns:        \n\
                                                       \t  <none>      "},
    {"staticModel",   gemStaticModel,   METH_VARARGS,  "Generate empty static Model\n\n\
                                                       Input arguments:\n\
                                                       \t  contextObj  \n\
                                                       Returns:        \n\
                                                       \t  modelObj    "},
    {"loadModel",     gemLoadModel,     METH_VARARGS,  "Load a Model from a file\n\n\
                                                       Input arguments:\n\
                                                       \t  contextObj  \n\
                                                       \t  filename    \n\
                                                       Returns:        \n\
                                                       \t  modelObj    "},
    {"getBRepOwner",  gemGetBRepOwner,  METH_VARARGS,  "Get Model containing BRep\n\n\
                                                       Input arguments:\n\
                                                       \t  brepObj     \n\
                                                       Returns:        \n\
                                                       \t  modelObj    \n\
                                                       \t  instance    \n\
                                                       \t  branch      "},
    {"solidBoolean",  gemSolidBoolean,  METH_VARARGS,  "Execute solid boolean operation\n\n\
                                                       Input arguments:\n\
                                                       \t  brepObj1    \n\
                                                       \t  brepObj2    \n\
                                                       \t  type        either 'INTERSECT', 'SUBTRACT', or 'UNION'\n\
                                                       \t  (xform)     <optional>\n\
                                                       Returns:        \n\
                                                       \t  modelObj    "},
    {"getAttribute",  gemGetAttribute,  METH_VARARGS,  "Get Attribute of a GEMobject\n\n\
                                                       Input arguments:\n\
                                                       \t  gemObj      \n\
                                                       \t  otype       either 'CONTEXT', 'MODEL', 'BRANCH', 'PARAM', 'BREP',\n\
                                                       \t                     'NODE', 'EDGE', 'LOOP', 'FACE', or 'SHELL'\n\
                                                       \t  eindex      <bias-1>\n\
                                                       \t  aindex      <bias-1>\n\
                                                       Returns:        \n\
                                                       \t  aname       \n\
                                                       \t  (values)    "},
    {"retAttribute",  gemRetAttribute,  METH_VARARGS,  "Return Attribute of a GEMobject\n\n\
                                                       Input arguments:\n\
                                                       \t  gemObj      \n\
                                                       \t  otype       either 'CONTEXT', 'MODEL', 'BRANCH', 'PARAM', 'BREP',\n\
                                                       \t                     'NODE', 'EDGE', 'LOOP', 'FACE', or 'SHELL'\n\
                                                       \t  eindex      <bias-1>\n\
                                                       \t  name        \n\
                                                       Returns:        \n\
                                                       \t  aindex      <bias-1>\n\
                                                       \t  (values)    "},
    {"setAttribute",  gemSetAttribute,  METH_VARARGS,  "Set an Attribute for a GEMobject\n\n\
                                                       Input arguments:\n\
                                                       \t  gemObj      \n\
                                                       \t  otype       either 'CONTEXT', 'MODEL', 'BRANCH', 'PARAM', 'BREP',\n\
                                                       \t                     'NODE', 'EDGE', 'LOOP', 'FACE', or 'SHELL'\n\
                                                       \t  eindex      <bias-1>\n\
                                                       \t  name        \n\
                                                       \t  (values)    \n\
                                                       Returns:        \n\
                                                       \t  <none>      "},

    // routines defined in model.h
    {"add2Model",     gemAdd2Model,     METH_VARARGS,  "Add a BRep to a static Model\n\n\
                                                       Input arguments:\n\
                                                       \t  modelObj    \n\
                                                       \t  brepObj     \n\
                                                       \t  (xform)     <optional>\n\
                                                       Returns:        \n\
                                                       \t  <none>      "},
    {"saveModel",     gemSaveModel,     METH_VARARGS,  "Save an up-to-date Model\n\n\
                                                       Input arguments:\n\
                                                       \t  modelObj    \n\
                                                       \t  filename    \n\
                                                       Returns:        \n\
                                                       \t  <none>      "},
    {"releaseModel",  gemReleaseModel,  METH_VARARGS,  "Release a Model and all of its storage\n\n\
                                                       Input arguments:\n\
                                                       \t  modelObj    \n\
                                                       Returns:        \n\
                                                       \t  <none>      "},
    {"copyModel",     gemCopyModel,     METH_VARARGS,  "Copy a Model\n\
                                                       Input arguments:\n\
                                                       \t  oldModelObj \n\
                                                       Returns:        \n\
                                                       \t  newModelObj "},
    {"regenModel",    gemRegenModel,    METH_VARARGS,  "Regenerate a non-static Model\n\n\
                                                       Input arguments:\n\
                                                       \t  modelObj    \n\
                                                       Returns:        \n\
                                                       \t  <none>      "},
    {"getModel",      gemGetModel,      METH_VARARGS,  "Get info about a Model\n\n\
                                                       Input arguments:\n\
                                                       \t  modelObj    \n\
                                                       Returns:        \n\
                                                       \t  server      \n\
                                                       \t  filename    \n\
                                                       \t  modeler     \n\
                                                       \t  uptodate    \n\
                                                       \t  (BReps)     \n\
                                                       \t  nparam      \n\
                                                       \t  nbranch     \n\
                                                       \t  nattr       "},
    {"getBranch",     gemGetBranch,     METH_VARARGS,  "Get info about a Branch in a Model\n\n\
                                                       Input arguments:\n\
                                                       \t  modelObj    \n\
                                                       \t  ibranch     <bias-1>\n\
                                                       Returns:        \n\
                                                       \t  bname       \n\
                                                       \t  btype       \n\
                                                       \t  suppress    \n\
                                                       \t  (parents)   \n\
                                                       \t  (children)  \n\
                                                       \t  nattr       "},
    {"setSuppress",   gemSetSuppress,   METH_VARARGS,  "Change suppression state for a Branch\n\n\
                                                       Input arguments:\n\
                                                       \t  modelObj    \n\
                                                       \t  ibranch     <bias-1>\n\
                                                       \t  istate      \n\
                                                       Returns:        \n\
                                                       \t  <none>      "},
    {"getParam",      gemGetParam,      METH_VARARGS,  "Get info about a Parameter in a Model\n\n\
                                                       Input arguments:\n\
                                                       \t  modelObj    \n\
                                                       \t  iparam      <bias-1>\n\
                                                       Returns:        \n\
                                                       \t  pname       \n\
                                                       \t  bflag       \n\
                                                       \t  order       \n\
                                                       \t  (values)    \n\
                                                       \t  nattr       "},
    {"setParam",      gemSetParam,      METH_VARARGS,  "Set new value for a driving Parameter\n\n\
                                                       Input arguments:\n\
                                                       \t  modelObj    \n\
                                                       \t  iparam      <bias-1>\n\
                                                       \t  (values)    \n\
                                                       Returns:        \n\
                                                       \t  <none>      "},

    // routines defined in brep.h
    {"getBRepInfo",   gemGetBRepInfo,   METH_VARARGS,  "Get info about a BRep\n\n\
                                                       Input arguments:\n\
                                                       \t  brepObj     \n\
                                                       Returns:        \n\
                                                       \t  (box)       \n\
                                                       \t  type        \n\
                                                       \t  nnode       \n\
                                                       \t  nedge       \n\
                                                       \t  nloop       \n\
                                                       \t  nface       \n\
                                                       \t  nshell      \n\
                                                       \t  nattr       "},
    {"getShell",      gemGetShell,      METH_VARARGS,  "Get info about a Shell in a BRep\n\n\
                                                       Input arguments:\n\
                                                       \t  brepObj     \n\
                                                       \t  ishell      <bias-1>\n\
                                                       Returns:        \n\
                                                       \t  type        \n\
                                                       \t  (faces)     \n\
                                                       \t  nattr       "},
    {"getFace",       gemGetFace,       METH_VARARGS,  "Get info about a Face in a BRep\n\n\
                                                       Input arguments:\n\
                                                       \t  brepObj     \n\
                                                       \t  iface       <bias-1>\n\
                                                       Returns:        \n\
                                                       \t  ID          \n\
                                                       \t  (uvbox)     \n\
                                                       \t  norm        \n\
                                                       \t  (loops)     \n\
                                                       \t  nattr       "},
    {"getWire",       gemGetWire,       METH_VARARGS,  "Get info about a Wire in a BRep\n\n\
                                                       Input arguments:\n\
                                                       \t  brepObj     \n\
                                                       Returns:        \n\
                                                       \t  (loops)     "},
    {"getLoop",       gemGetLoop,       METH_VARARGS,  "Get info about a Loop in a BRep\n\n\
                                                       Input arguments:\n\
                                                       \t  brepObj     \n\
                                                       \t  iloop       <bias-1>\n\
                                                       Returns:        \n\
                                                       \t  face        \n\
                                                       \t  type        \n\
                                                       \t  (edges)     \n\
                                                       \t  nattr       "},
    {"getEdge",       gemGetEdge,       METH_VARARGS,  "Get data for an Edge in a BRep\n\n\
                                                       Input arguments:\n\
                                                       \t  brepObj     \n\
                                                       \t  iedge       <bias-1>\n\
                                                       Returns:        \n\
                                                       \t  (tlimit)    \n\
                                                       \t  (nodes)     \n\
                                                       \t  (faces)     \n\
                                                       \t  nattr       "},
    {"getNode",       gemGetNode,       METH_VARARGS,  "Get info about a Node in a BRep\n\n\
                                                       Input arguments:\n\
                                                       \t  brepObj     \n\
                                                       \t  inode       <bias-1>\n\
                                                       Returns:        \n\
                                                       \t  (xyz)       \n\
                                                       \t  nattr       "},
    {"getMassProps",  gemGetMassProps,  METH_VARARGS,  "Get mass properties about a BRep entity\n\n\
                                                       Input arguments:\n\
                                                       \t  brepObj     \n\
                                                       \t  etype       either 'FACE', 'SHELL', or 'BREP'\n\
                                                       \t  eindex      <bias-1>\n\
                                                       Returns:        \n\
                                                       \t  (props)     "},
    {"isEquivalent",  gemIsEquivalent,  METH_VARARGS,  "Determine is two BRep entities are the same\n\n\
                                                       Input arguments:\n\
                                                       \t  etype       either 'NODE', 'EDGE', or 'FACE'\n\
                                                       \t  brepObj1    \n\
                                                       \t  eindex1     <bias-1>\n\
                                                       \t  brepObj2    \n\
                                                       \t  eindex2     <bias-1>\n\
                                                       Returns:        \n\
                                                       \t  bool        "},

    // (selected) routines defined in drep.h
    {"newDRep",       gemNewDRep,       METH_VARARGS,  "Make a new DRep\n\n\
                                                       Input arguments:\n\
                                                       \t  modelObj    \n\
                                                       Returns:        \n\
                                                       \t  drepObj     "},
    {"tesselDRep",    gemTesselDRep,    METH_VARARGS,  "Tessellate a BRep into a DRep\n\n\
                                                       Input arguments:\n\
                                                       \t  drepObj     \n\
                                                       \t  ibrep       <bias-1>\n\
                                                       \t  maxang      \n\
                                                       \t  maxlen      \n\
                                                       \t  maxsag      \n\
                                                       Returns:        \n\
                                                       \t  <none>      "},
    {"getTessel",     gemGetTessel,     METH_VARARGS,  "Get tessellation data for a Face in a BRep\n\n\
                                                       Input arguments:\n\
                                                       \t  drepObj     \n\
                                                       \t  ibrep       <bias-1>\n\
                                                       \t  iface       <bias-1>\n\
                                                       Returns:        \n\
                                                       \t  xyzArray    \n\
                                                       \t  uvArray     \n\
                                                       \t  connArray   "},
     {"destroyDRep",  gemDestroyDRep,   METH_VARARGS,  "Destroy a DRep and its contents\n\n\
                                                       Input arguments:\n\
                                                       \t  drepObj     \n\
                                                       Returns         \n\
                                                       \t  <none>      "},

    // routine that links to gv
    {"plotDRep",      gemPlotDRep,      METH_VARARGS,  "Plot a DRep\n\n\
                                                       Input arguments:\n\
                                                       \t  drepObj     \n\
                                                       Returns         \n\
                                                       \t  <none>      "},

    {NULL, NULL, 0, NULL}        // Sentinel
};

#ifndef PyMODINIT_FUNC /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC
initgem(void)
{
    /* (re)initialize the gemObjects */
    if (gemObjects != NULL) free(gemObjects);
       gemObjects = NULL;
    numGemObjects = 0;
    maxGemObjects = 0;

    /* initialize the Module */
    Py_InitModule("gem", GemMethods);

    /* load the numpy C API */
    import_array();
}
