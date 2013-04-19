# test suite for gem.so

import os
import sys
import tempfile
import shutil

import unittest
from pygem_diamond import gem
from pygem_diamond.pygem import GEMParametricGeometry

sample_file = os.path.join(os.path.dirname(__file__), "sample.csm")

def dbg(msg):
    pass
    # sys.stderr.write('***** ')
    # sys.stderr.write(msg)
    # sys.stderr.write('\n')
    # sys.stderr.flush()

class PygemTestCase(unittest.TestCase):

    def test_ContextAttributes(self):
        myContext = gem.Context()

        myContext.setAttribute("CONTEXT", 0, "s_attr", "context attribute")
        myContext.setAttribute("CONTEXT", 0, "i_attrs", (1111111, 2222222, 3333333, 4444444))
        myContext.setAttribute("CONTEXT", 0, "r_attrs", (0.1234567890123456, 0.2345678901234567))

        aindex, values = myContext.getAttribute("CONTEXT", 0, "s_attr")
        self.assertEqual(aindex, 1)
        self.assertEqual(values, "context attribute")

        aindex, values = myContext.getAttribute("CONTEXT", 0, "i_attrs")
        self.assertEqual(aindex, 2)
        self.assertEqual(values[0], 1111111)
        self.assertEqual(values[1], 2222222)
        self.assertEqual(values[2], 3333333)
        self.assertEqual(values[3], 4444444)

        aindex, values = myContext.getAttribute("CONTEXT", 0, "r_attrs")
        self.assertEqual(aindex, 3)
        self.assertEqual(values[0], 0.1234567890123456)
        self.assertEqual(values[1], 0.2345678901234567)

    def test_LoadModel(self):
        myContext = gem.Context()
        myModel = myContext.loadModel(sample_file)

        dbg('returned from loadModel\n')

        myModel.setAttribute("MODEL", 0, "s_attr", "model attribute")
        myModel.setAttribute("MODEL", 0, "i_attrs", (2222222, 3333333, 4444444, 5555555))
        myModel.setAttribute("MODEL", 0, "r_attrs", (0.2345678901234567, 0.3456789012345678))

        copyModel = myModel.copy()
        dbg('returned from myModel.copy(). releaseing myModel')

        myModel.release()
        dbg('back from myModel.release\n')

        server, filename, modeler, uptodate, myBReps, nparam, nbranch, nattr = copyModel.getInfo()
        dbg('back from copyModel.getInfo')

        self.assertEqual(filename, sample_file)
        self.assertEqual(modeler,  "OpenCSM")
        self.assertEqual(uptodate,  1)
        self.assertEqual(nparam,   33)
        self.assertEqual(nbranch,  22)
        self.assertEqual(nattr,     3)

        aname, values = copyModel.getAttribute("MODEL", 0, 1)
        self.assertEqual(aname,  "s_attr")
        self.assertEqual(values, "model attribute")

        aname, values = copyModel.getAttribute("MODEL", 0, 2)
        self.assertEqual(aname,     "i_attrs")
        self.assertEqual(values[0], 2222222)
        self.assertEqual(values[1], 3333333)
        self.assertEqual(values[2], 4444444)
        self.assertEqual(values[3], 5555555)

        aname, values = copyModel.getAttribute("MODEL", 0, 3)
        self.assertEqual(aname,     "r_attrs")
        self.assertEqual(values[0], 0.2345678901234567)
        self.assertEqual(values[1], 0.3456789012345678)

        del myContext

    def test_StaticModel(self):
        myContext = gem.Context()
        myModel   = myContext.loadModel(sample_file)

        server, filename, modeler, uptodate, myBReps, nparam, nbranch, nattr = myModel.getInfo()

        newModel = myContext.staticModel()
        newModel.add2Model(myBReps[0])
        newModel.add2Model(myBReps[0], (0.5, 0, 0, 2,  0, 0.5, 0, 0,  0, 0, 0.5, 0))

        server, filename, modeler, uptodate, myBReps, nparam, nbranch, nattr = newModel.getInfo()

        massProps1 = myBReps[0].getMassProps("BREP", 0)
        massProps2 = myBReps[1].getMassProps("BREP", 0)

        self.assertAlmostEqual(massProps1[0],   8 * massProps2[0])     # volume
        self.assertAlmostEqual(massProps1[1],   4 * massProps2[1])     # surface area
        self.assertAlmostEqual(massProps1[2],   2 * massProps2[2] - 4)     # xcg
        self.assertAlmostEqual(massProps1[3],   2 * massProps2[3])     # ycg
        self.assertAlmostEqual(massProps1[4],   2 * massProps2[4])     # zcg
        self.assertAlmostEqual(massProps1[5],  32 * massProps2[5])     # Ixx
        self.assertAlmostEqual(massProps1[6],  32 * massProps2[6])     # Ixy
        self.assertAlmostEqual(massProps1[7],  32 * massProps2[7])     # Ixz
        self.assertAlmostEqual(massProps1[8],  32 * massProps2[8])     # Iyx
        self.assertAlmostEqual(massProps1[9],  32 * massProps2[9])     # Iyy
        self.assertAlmostEqual(massProps1[10], 32 * massProps2[10])     # Iyz
        self.assertAlmostEqual(massProps1[11], 32 * massProps2[11])     # Izx
        self.assertAlmostEqual(massProps1[12], 32 * massProps2[12])     # Izy
        self.assertAlmostEqual(massProps1[13], 32 * massProps2[13])     # Izz

        del myContext

    def test_Branches(self):
        myContext = gem.Context()
        myModel   = myContext.loadModel(sample_file)

        server, filename, modeler, uptodate, myBReps, nparam, nbranch, nattr = myModel.getInfo()

        for ibranch in range(2, nbranch+1):
            myModel.setAttribute("BRANCH", ibranch, "s_attr", "$branch attribute")

            bname, btype, suppress, parents, children, nattr = myModel.getBranch(ibranch)
            myModel.setAttribute("BRANCH", ibranch, "s_attr", "$branch attribute")

        ibranch = 9
        myModel.setSuppress(ibranch, 1)

        bname, btype, suppress, parents, children, nattr = myModel.getBranch(ibranch)
        self.assertEqual(bname,    "feature")
        self.assertEqual(btype,    "extrude")
        self.assertEqual(suppress, 1        )
        self.assertEqual(parents,  (8,)     )
        self.assertEqual(children, (10,)    )
        self.assertEqual(nattr,    1        )

        myModel.regenerate()

        for ibranch in range(2, nbranch+1):
            bname, btype, suppress, parents, children, nattr = myModel.getBranch(ibranch)
            if (ibranch == 2):
                self.assertEqual(nattr, 4, "ibranch=%d" % ibranch)
            else:
                self.assertEqual(nattr, 1, "ibranch=%d" % ibranch)

            aindex, values = myModel.getAttribute("BRANCH", ibranch, "s_attr")
            self.assertEqual(aindex, nattr, "ibranch=%d" % ibranch)
            self.assertEqual(values, "$branch attribute", "ibranch=%d" % ibranch)

        del myContext


    def test_Parameters(self):
        myContext = gem.Context()
        myModel   = myContext.loadModel(sample_file)

        server, filename, modeler, uptodate, myBReps, nparam, nbranch, nattr = myModel.getInfo()

        for iparam in range(1, nparam+1):
            myModel.setAttribute("PARAM", iparam, "s_attr", "param attribute")
            myModel.setAttribute("PARAM", iparam, "i_attr", (     iparam,)   )
            myModel.setAttribute("PARAM", iparam, "r_attr", (10.0*iparam,)   )

        iparam = 4
        meta = myModel.getParam(iparam, get_meta=True)
        self.assertEqual(meta['name'], "ymax")
        self.assertEqual(meta['iotype'], 'in' )
        self.assertEqual(meta['order'], 0 )
        self.assertEqual(meta['value'], 1.0 )
        self.assertEqual(meta['nattr'], 3 )

        myModel.setParam(iparam, (1.5,))

        myModel.regenerate()

        meta = myModel.getParam(iparam, get_meta=True)
        self.assertEqual(meta['name'], "ymax")
        self.assertEqual(meta['iotype'], 'in' )
        self.assertEqual(meta['order'], 0 )
        self.assertEqual(meta['value'], 1.5 )
        self.assertEqual(meta['nattr'], 3 )

        for iparam in range(1, nparam+1):
            meta = myModel.getParam(iparam, get_meta=True)
            self.assertEqual(meta['nattr'], 3, "iparam=%d" % iparam)

            aindex, values = myModel.getAttribute("PARAM", iparam, "s_attr")
            self.assertEqual(aindex, 1,                 "iparam=%d" % iparam)
            self.assertEqual(values, "param attribute", "iparam=%d" % iparam)

            aindex, values = myModel.getAttribute("PARAM", iparam, "i_attr")
            self.assertEqual(aindex,    2,      "iparam=%d" % iparam)
            self.assertEqual(values[0], iparam, "iparam=%d" % iparam)

            aindex, values = myModel.getAttribute("PARAM", iparam, "r_attr")
            self.assertEqual(aindex, 3,              "iparam=%d" % iparam)
            self.assertEqual(values[0], 10.0*iparam, "iparam=%d" % iparam)

        del myContext

    def test_BRep(self):
        myContext = gem.Context()
        myModel   = myContext.loadModel(sample_file)

        server, filename, modeler, uptodate, myBReps, nparam, nbranch, nattr = myModel.getInfo()
        self.assertEqual(nattr, 0)

        for myBRep in myBReps:
            box, typ, nnode, nedge, nloop, nface, nshell, nattr = myBRep.getInfo()
            self.assertEqual(nattr, 2)

            aname, values = myBRep.getAttribute("BREP", 0, 1)
            self.assertEqual(aname, "body")
            self.assertEqual(values[0], 18)

            # check that all nodes are in box
            for inode in range(1, nnode+1):
                xyz, nattr = myBRep.getNode(inode)
                self.assertEqual(nattr, 0)

                self.assertTrue(xyz[0] >= box[0], "x[%d] < box[0]" % inode)
                self.assertTrue(xyz[1] >= box[1], "y[%d] < box[1]" % inode)
                self.assertTrue(xyz[2] >= box[2], "z[%d] < box[2]" % inode)
                self.assertTrue(xyz[0] <= box[3], "x[%d] > box[3]" % inode)
                self.assertTrue(xyz[1] <= box[4], "y[%d] > box[4]" % inode)
                self.assertTrue(xyz[2] <= box[5], "z[%d] > box[5]" % inode)

            # check consistency of all loops
            for iloop in range(1, nloop+1):
                iface, typ, edges, nattr = myBRep.getLoop(iloop)
                self.assertEqual(nattr, 0, "iloop=%d" % iloop)

                # loop and face point to each other
                ID, uvbox, norm, loops, nattr = myBRep.getFace(iface)
                self.assertTrue(iloop in loops, "iloop=%d, iface=%d" % (iloop, iface))

                # edge and face point to each other
                for i in range(len(edges)):
                    tlimit, nodes, faces, nattr = myBRep.getEdge(edges[i])
                    self.assertTrue(iface in faces, "iloop-%d, i=%d" % (iloop, i))

                # edges link end-to-end in loop
                tlimit, nodes, faces, nattr = myBRep.getEdge(edges[0])
                iend = nodes[1]

                for i in range(1, len(edges)):
                    tlimit, nodes, faces, nattr = myBRep.getEdge(edges[i])
                    self.assertEqual(iend, nodes[0], "iloop=%d, i=%d" % (iloop, i))
                    iend = nodes[1]

            # check consistency of all shells
            for ishell in range(1, nshell+1):
                typ, faces, nattr = myBRep.getShell(ishell)
                self.assertEqual(nattr, 0)

        del myContext

    def test_DRep(self):
        myContext = gem.Context()
        myModel = myContext.loadModel(sample_file)
        myDRep = myModel.newDRep()
        dbg("in test, DRep created. tessellating...")
        myDRep.tessellate(0, 0, 0, 0)
        dbg("tesselDRep returned. calling getTessel")

        triArray, xyzArray = myDRep.getTessel(1, 1)
        dbg("getTessel returned")

        npnt = (xyzArray.shape)[0]
        ntri = (triArray.shape)[0]

        # make sure that triangle pointers are consistent
        for itri in range(1, ntri+1):
            for iside in [0, 1, 2]:
                jtri = triArray[itri-1,iside]
                if not ((jtri > 0) and (jtri <= npnt)):
                    self.fail('jtri not in (0 to %d)' % npnt)

        del myContext



class GEMParametricGeometryTestCase(unittest.TestCase):

    def setUp(self):
        self.csm_input = """
# bottle2 (from OpenCASCADE tutorial)
# written by John Dannenhoffer

# default design parameters
despmtr   width               10.00
despmtr   depth                4.00
despmtr   height              15.00
despmtr   neckDiam             2.50
despmtr   neckHeight           3.00
despmtr   wall                 0.20     wall thickness (in neck)
despmtr   filRad1              0.25     fillet radius on body of bottle
despmtr   filRad2              0.10     fillet radius between bottle and neck

# basic bottle shape (filletted)

set       baseHt    height-neckHeight

skbeg     -width/2  -depth/4  0
   cirarc 0         -depth/2  0         +width/2  -depth/4  0
   linseg +width/2  +depth/4  0
   cirarc 0         +depth/2  0         -width/2  +depth/4  0
   linseg -width/2  -depth/4  0
skend
extrude   0         0         baseHt
fillet    filRad1

# neck
cylinder  0         0         baseHt    0         0         height      neckDiam/2

# join the neck to the bottle and apply a fillet at the union
union
fillet    filRad2

# hollow out bottle
hollow    wall      18

end
        """
        self.tdir = tempfile.mkdtemp()
        self.model_file = os.path.join(self.tdir, 'bottle.csm')
        with open(self.model_file, 'w') as f:
            f.write(self.csm_input)

    def tearDown(self):
        shutil.rmtree(self.tdir)

    def test_GEMParametricGeometry(self):
        geom = GEMParametricGeometry()
        geom.model_file = self.model_file
        params = geom.list_parameters()
        expected_inputs = set(['width', 'depth', 'height', 
            'neckDiam', 'neckHeight', 'wall',
            'filRad1', 'filRad2'])
        self.assertEqual(expected_inputs, 
            set([k for k,v in params if v['iotype']=='in']))
        expected_outs = set(['zcg', 'zmax', 'xcg', 'zmin', 'Ixz', 'Izx', 'Ixx', 'Ixy', 
                'baseHt', 'xmin', 'Izy', 'Izz', 'ymin', 'ibody', 'ymax', 'nnode', 'ycg', 'nface', 
                'volume', 'Iyy', 'Iyx', 'Iyz', 'area', 'nedge', 'xmax'])
        self.assertEqual(expected_outs, set([k for k,v in params if v['iotype']=='out']))

        vals = geom.get_parameters(['baseHt'])
        baseHt = vals[0]
        self.assertEqual(baseHt, 12.0)
        geom.set_parameter('height', 20.0)
        geom.regen_model()
        vals = geom.get_parameters(['baseHt'])
        baseHt = vals[0]
        self.assertEqual(baseHt, 17.0)
        geom.terminate()


if __name__ == "__main__":
    unittest.main()
