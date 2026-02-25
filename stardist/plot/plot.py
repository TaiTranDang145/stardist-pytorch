import matplotlib
import colorsys
import numpy as np
def random_label_cmap(n=2**16, h = (0,1), l = (.4,1), s =(.2,.8), seed=None):
    rng = np.random.default_rng(seed)
    
    h, l, s = rng.uniform(*h, n), rng.uniform(*l, n), rng.uniform(*s, n)
    cols = np.stack([colorsys.hls_to_rgb(_h,_l,_s) for _h,_l,_s in zip(h,l,s)],axis=0)
    cols[0] = 0
    return matplotlib.colors.ListedColormap(cols)
