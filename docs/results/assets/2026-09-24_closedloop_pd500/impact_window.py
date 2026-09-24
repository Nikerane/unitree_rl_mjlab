"""Offline complete-first-contact-window reduction for this diagnostic."""
import math
import numpy as np


def reduce_window(signed, force, contact, dt, terminal):
    count=round(.05/dt)
    assert count>0 and math.isclose(count*dt,.05,abs_tol=1e-10)
    started=contact.any(axis=0)
    onset=contact.argmax(axis=0)
    complete=started & (onset+count<=len(contact))
    source_complete=complete & ((onset+count)*dt<=(terminal+1)*.02+1e-10)
    slots=np.arange(len(contact))[:,None]
    active=contact & started & (slots>=onset) & (slots<onset+count)
    joint=(np.abs(signed.astype(np.float64))*active[:,:,None]).sum(axis=0)*dt
    delivered=(np.maximum(force.astype(np.float64),0)*active).sum(axis=0)*dt
    joint[~complete]=np.nan
    delivered[~complete]=np.nan
    return dict(started=started,complete=complete,source_complete=source_complete,
        onset_s=np.where(started,(onset+1)*dt,np.nan),
        joint_impulse=joint,delivered=delivered)
