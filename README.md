# Pytorch-Interp2d
Pytorch implement of interp 2d  
(Perform same as griddata in scipy)   
I just modify some of scipy.spatial code

### Usage
Please first compile the Cython code. Use:
```
cd spatial
python setup.py build_ext --inplace
```
Then see interp2d.py code.

#### Notice
This Repo is not stable, I haven't test the function of code. Cause the core code is clone from scipy, the code should work.    
If there is any bugs, please let me know.

#### Algorithm
The algorithm is just same as scipy do. The problem is to interpolate 2D scatter points to a continuous image.   
First, divide the image to multiple triangles. Then, perform linear interpolation inside the triangles. Use pytorch to compute the linear interpolation process, the back propagation should be autoly performed.

#### Acknowledge
Scipy https://github.com/scipy/scipy  
qhull http://www.qhull.org/




