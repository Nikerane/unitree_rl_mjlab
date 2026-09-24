import numpy as np
import pytest
import impact_window as subject


@pytest.mark.parametrize("terminal,source_complete",[(1,False),(2,True)])
def test_complete_window_boundary_and_censoring(terminal,source_complete):
    contact=np.zeros((8,3),bool)
    contact[1:5,0]=True; contact[6:,1]=True
    signed=np.full((8,3,6),-2.)
    force=np.full((8,3),3.)
    out=subject.reduce_window(signed,force,contact,.01,np.array([terminal,3,3]))
    np.testing.assert_array_equal(out["started"],[True,True,False])
    np.testing.assert_array_equal(out["complete"],[True,False,False])
    np.testing.assert_array_equal(out["source_complete"],[source_complete,False,False])
    np.testing.assert_allclose(out["joint_impulse"][0],[.08]*6)
    assert out["delivered"][0]==pytest.approx(.12)
    assert np.isnan(out["delivered"][1:]).all()
    assert np.isnan(out["joint_impulse"][1:]).all()
