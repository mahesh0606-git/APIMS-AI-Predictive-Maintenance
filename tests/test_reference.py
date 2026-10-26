import unittest
import numpy as np
from apims.telemetry import normalize,windows
from apims.scheduler import Task,optimize

class TestReference(unittest.TestCase):
    def test_normalize(self):
        a=np.array([[45,0,0,0,0,0,0,4000,12,-10]],dtype=np.float32)
        np.testing.assert_allclose(normalize(a),0,atol=1e-6)
    def test_windows(self):
        self.assertEqual(windows(np.zeros((130,10))).shape,(3,128,10))
    def test_schedule(self):
        self.assertEqual(len(optimize([Task('t',1,2,10)],generations=2)),3)

if __name__=='__main__': unittest.main()
