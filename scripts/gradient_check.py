import numpy as np 
from tcnvol.tcn import TCN
from tcnvol.losses import bce_loss

def gc_loss():
    p = tcn_gc.forward(X_gc, training=False)
    return bce_loss(p, y_gc, pos_w_gc)


def relative_error(a, n):
    """L2-norm relative error — robust to near-zero gradient elements."""
    return (np.linalg.norm(a.ravel() - n.ravel()) /
            (np.linalg.norm(a.ravel()) + np.linalg.norm(n.ravel()) + 1e-12))