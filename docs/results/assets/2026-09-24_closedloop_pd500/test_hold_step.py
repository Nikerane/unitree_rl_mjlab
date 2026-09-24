from types import SimpleNamespace
import mujoco
import torch
import hold_step as subject


def test_native_step_uses_held_torque_until_next_500hz_tick():
    model=mujoco.MjModel.from_xml_string('''<mujoco><option timestep="0.000125" gravity="0 0 0"/>
      <worldbody><body><joint name="j" armature="1"/><geom type="sphere" size=".1" mass="1"/></body></worldbody>
      <actuator><position joint="j" kp="100" kv="20" forcerange="-3 3"/></actuator></mujoco>''')
    data=mujoco.MjData(model)
    live=SimpleNamespace(actuator_gainprm=torch.from_numpy(model.actuator_gainprm).unsqueeze(0),
        actuator_biasprm=torch.from_numpy(model.actuator_biasprm).unsqueeze(0))
    state=SimpleNamespace(**{k:torch.from_numpy(getattr(data,k)).unsqueeze(0)
        for k in ("qpos","qvel","ctrl","actuator_force")})
    sim=SimpleNamespace(model=live,data=state,step=lambda:mujoco.mj_step(model,data))
    step=subject.HeldTorqueStep(sim,torch.tensor([0]),torch.tensor([0]),
        torch.tensor([[10.]],dtype=torch.float64),torch.tensor([[2.]],dtype=torch.float64),
        torch.tensor([[-3.,3.]],dtype=torch.float64),.000125)
    data.ctrl[0]=1
    step()
    assert data.actuator_force[0]==3
    data.ctrl[0]=-1
    for _ in range(15):
        step()
        assert data.actuator_force[0]==3
    step()
    assert data.actuator_force[0]==-3
    assert step.pd.updates==2
    assert step.count==17
    assert step.max_error==0
