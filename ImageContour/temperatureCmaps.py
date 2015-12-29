import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from np_utils import totuple

def makeCmap(name,rgbVals,blackened=True):
    cnames = 'red','green','blue'
    l = len(rgbVals[0])-1
    cpos = [ 1.0*i/l for i in range(l+1) ]
    vblack = [ [0.]+v[1:] for v in rgbVals ]
    vals1,vals2 = (vblack if blackened else rgbVals), rgbVals
    d = { c:totuple(zip(cpos,v1,v2))
         for c,v1,v2 in zip(cnames,vals1,vals2) }
    return mpl.colors.LinearSegmentedColormap(name,d)

temperatureRGBs = 0.5*np.array([[0,1,2,2,2],[0,1,2,2,0],[2,2,2,0,0]])
temperatureCmap = makeCmap('temperature',temperatureRGBs.tolist())
temperatureLightCmap = makeCmap('temperature',np.clip(temperatureRGBs+0.25,0,1).tolist())
temperatureDarkCmap = makeCmap('temperature',(0.75*temperatureRGBs).tolist())
