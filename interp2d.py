
import warnings

import math
from torch import nn
from torch.autograd import Function
import torch
import spatial.qhull as qhull
import numpy as np
import torch.nn.functional as F



class Interp2D(nn.Module):
'''
 New 2d Interpolation in Pytorch
 Reference to scipy.griddata
 Args；
    w, h:  width,height of input
    points: points to interpote shape: [num, 2]
    values:  values of points shape:[num, valuedim]
return: 
   2D interpolate result, shape: [valuedim, w, h]
 '''
    def __init__(self, w, h, add_corner=True):
        super(Interp2D,self).__init__()
        row_coord = np.arange(h).repeat([w]).reshape([h,w])
        col_coord = np.arange(w).repeat([h]).reshape([w,h]).T
        self.coord = np.stack([row_coord, col_coord])
        self.coord = self.coord.transpose([1,2,0]).reshape([-1,2])
        self.add_corner = add_corner
        self.w = w
        self.h = h
        if self.add_corner==False:
            raise Exception('Now add_corner must be true')

    def forward(self, points, values):
        if self.add_corner:
            points = torch.cat([points, torch.Tensor([[0,0], [0, self.w-1],
                                  [self.h-1,0], [self.h-1, self.w-1]])], dim=0)
            values = torch.cat([values, torch.zeros([4,values.shape[1]])], dim=0)
           # Add 4 zeros corner points 
        self.tri = qhull.Delaunay(points)
        vdim = values.shape[-1]
        print(points.shape)
        isimplex, weights = self.tri.find_simplex(self.coord, return_c=True)
        #the array `weights` is filled with the corresponding barycentric coordinates.
        weights = torch.from_numpy(weights).float()

        isimplex = torch.from_numpy(isimplex)
        isimplex = isimplex.long()
        isimplex = isimplex.reshape([-1,1])
        print(isimplex.shape, weights.shape)
        
        # shape: isimplex: [h*w,1]      c: [h,w,c]
       
        simplices =torch.from_numpy(self.tri.simplices).long()
        
        tri_map = torch.gather(simplices, dim=0, index=isimplex.repeat([1,3]))
        print(tri_map.shape, values.shape)
        value_corr = [torch.gather(values, dim=0, index=tri_map[:,i].
                                    reshape([-1,1]).repeat([1,vdim])) for i in range(3)]
        value_corr = torch.stack(value_corr)
        print('vc',value_corr.shape)
        print(weights.shape)
        weights = weights.transpose(1,0).unsqueeze(2).repeat([1,1,2])
        print(weights.dtype, value_corr.dtype)
        out = torch.mul(value_corr, weights).sum(dim=0)
        print(out.shape)
        return out.reshape([self.h, self.w, vdim]).transpose(2,0,1)


if __name__=='__main__':
    interp2d = Interp2D(10,10)
    points = torch.rand([10,2])*10
    values = torch.rand([10,2])
    out = interp2d(points,values)
    print('out shape', out.shape)
    #print(out)

