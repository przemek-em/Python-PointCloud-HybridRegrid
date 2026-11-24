from pathlib import Path
from threading import Event, enumerate as threads
from tempfile import TemporaryDirectory
import json
import sys
import unittest

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from boundary_patch import complete
from cloud import Cloud, Cancelled
from cloud_io import read_cloud, write_cloud
from jobs import process_surface_file
from surface_fill import Parameters as SurfaceParameters
from uniform_rebuild import Parameters as RebuildParameters, rebuild


def plane():
    return np.array([[x+.2,y+.2,.2] for x in range(-15,16) for y in range(-15,16) if not(abs(x)<=4 and abs(y)<=4)])


class BoundaryPatchTests(unittest.TestCase):
    def test_entire_hole_including_centre_is_covered(self):
        source=Cloud(plane())
        result=complete(source,1,10,1,workers=4)
        self.assertIsNotNone(result.cloud)
        self.assertGreater(result.stats['patched_holes'],0)
        q=np.array([[x+.2,y+.2,.2] for x in range(-4,5) for y in range(-4,5)])
        distance,_=cKDTree(result.cloud.xyz).query(q)
        self.assertLess(distance.max(),.8)
        np.testing.assert_allclose(result.cloud.xyz[:,2],.2,atol=1e-10)
        np.testing.assert_array_equal(source.xyz,plane())

    def test_rotated_vertical_curved_hole(self):
        xyz=np.array([[20*np.cos(t),20*np.sin(t),z] for t in np.arange(-1.1,1.101,.05) for z in range(-15,16) if not(abs(t)<.28 and abs(z)<5)])
        axis=np.array([1.,2.,3.]);axis/=np.linalg.norm(axis)
        cross=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
        rotation=np.eye(3)*np.cos(.8)+(1-np.cos(.8))*np.outer(axis,axis)+np.sin(.8)*cross
        offset=np.array([650000.,5100000.,100.])
        result=complete(Cloud(xyz@rotation.T+offset),1,12,1,workers=4)
        self.assertIsNotNone(result.cloud)
        restored=(result.cloud.xyz-offset)@rotation
        self.assertLess(np.quantile(np.abs(np.linalg.norm(restored[:,:2],axis=1)-20),.95),.35)
        centre_distance,_=cKDTree(restored).query([[20,0,0]])
        self.assertLess(centre_distance[0],1)

    def test_parallel_sheets_are_not_blended(self):
        p=plane()
        xyz=np.concatenate([p,p+[0,0,2]])
        result=complete(Cloud(xyz),.5,10,1,workers=4)
        self.assertIsNotNone(result.cloud)
        z=result.cloud.xyz[:,2]
        self.assertLess(np.minimum(abs(z-.2),abs(z-2.2)).max(),1e-9)
        self.assertTrue((z<1).any() and (z>1).any())

    def test_open_notch_remains_open(self):
        xyz=np.array([[x+.2,y+.2,.2] for x in range(-15,16) for y in range(-15,16) if not(x<=3 and abs(y)<=4)])
        result=complete(Cloud(xyz),1,20,1,workers=4)
        self.assertIsNone(result.cloud)

    def test_reach_rejects_whole_large_patch(self):
        result=complete(Cloud(plane()),1,2,1,workers=4)
        self.assertIsNone(result.cloud)
        self.assertGreater(result.stats['beyond_reach'],0)

    def test_attributes_and_serial_parallel_equivalence(self):
        xyz=plane()
        cloud=Cloud(xyz,np.full_like(xyz,.4),np.full(len(xyz),200),np.full(len(xyz),7))
        a=complete(cloud,1,10,1,workers=1)
        b=complete(cloud,1,10,1,workers=4)
        for name in ('xyz','rgb','intensity','classification','generated','source_indices'):
            np.testing.assert_array_equal(getattr(a.cloud,name),getattr(b.cloud,name))
        np.testing.assert_allclose(a.cloud.rgb,.4)
        self.assertTrue(a.cloud.generated.all())
        self.assertTrue((a.cloud.source_indices==-1).all())
        self.assertFalse(any(t.name.startswith('boundary-patch') for t in threads()))

    def test_limits_and_cancellation(self):
        event=Event();event.set()
        with self.assertRaises(Cancelled):
            complete(Cloud(plane()),1,10,1,cancel=event)
        with self.assertRaisesRegex(ValueError,'too large'):
            complete(Cloud(plane()),1,10,1,max_candidates=4)
        self.assertFalse(any(t.name.startswith('boundary-patch') for t in threads()))

    def test_append_mode_file_job_preserves_originals_and_reports_patches(self):
        with TemporaryDirectory() as folder:
            source, output = Path(folder)/'source.ply', Path(folder)/'patches.laz'
            cloud=Cloud(plane(),np.full_like(plane(),.4))
            write_cloud(source,cloud,1)
            report=process_surface_file(source,output,SurfaceParameters(spacing=1,max_gap_distance=10,method='boundary',workers=4),Event(),lambda f,m:None,lambda m:None)
            result=read_cloud(output)
            np.testing.assert_allclose(result.xyz[:len(cloud.xyz)],cloud.xyz,atol=.001)
            self.assertGreater(report['boundary_patches']['added_output_cells'],0)
            self.assertTrue(result.generated[len(cloud.xyz):].all())
            saved=json.loads(Path(str(output)+'.regrid.json').read_text())
            self.assertEqual(saved['parameters']['method'],'boundary')
            distance,_=cKDTree(result.xyz).query([[.2,.2,.2]])
            self.assertLess(distance[0],1.5)

    def test_strong_rebuild_runs_boundary_completion(self):
        xyz=np.array([[8*np.cos(t),8*np.sin(t),z] for t in np.arange(-1.1,1.101,.125) for z in range(-10,11) if not(abs(t)<.5 and abs(z)<4)])
        result=rebuild(Cloud(xyz),RebuildParameters(spacing=.5,positions='surface',completion='strong',reach=8,workers=4))
        self.assertGreater(result.boundary_patches['patched_holes'],0)
        self.assertGreater(result.boundary_patches['added_output_cells'],0)
        distance,_=cKDTree(result.cloud.xyz).query([[8,0,0]])
        self.assertLess(distance[0],1)
        regular=rebuild(Cloud(xyz),RebuildParameters(spacing=.5,positions='grid',completion='strong',reach=8,workers=4))
        self.assertGreater(regular.boundary_patches['added_output_cells'],0)
        lattice=(regular.cloud.xyz-regular.origin)/regular.spacing-.5
        np.testing.assert_allclose(lattice,np.rint(lattice),atol=1e-10)


if __name__=='__main__':
    unittest.main()
