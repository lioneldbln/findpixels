import pyopencl as cl
import numpy as np
import os
import Image

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
 
img = Image.open("test.bmp")
a = np.asarray(img).astype(np.uint8)
dim = a.shape
res = np.zeros((dim[0], dim[1]),np.int32)
print res.nbytes
 
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
 
mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, res.nbytes)
 
prg = cl.Program(ctx, """
    __kernel void copy(__global const uchar *a, __global int *c)
    {
      int rowid = get_global_id(0);
      int colid = get_global_id(1);

		int ncols = %d;
		int npix = %d; //number of pixels, 3 for RGB 4 for RGBA

		int index = rowid * ncols * npix + colid * npix;
		int cindex = rowid * ncols + colid;
		c[cindex] = a[index + 0] + a[index + 1] + a[index + 2];
    }
    """ % (dim[1], dim[2])).build()
 
prg.copy(queue, (dim[0], dim[1]), None, a_buf, dest_buf)
 
cl.enqueue_copy(queue, res, dest_buf)
 
print a
print "--------------------------------------------------"
print res

dim = res.shape

brightestpixel = np.zeros((1, 2), np.int32)
brightestvalue = np.zeros((1), np.int32)

darkestpixel = np.zeros((1, 2), np.int32)
darkestvalue = np.zeros((1), np.int32)

brightestvalue[0] = 150;
darkestvalue[0] = 88;

print "before: ", brightestvalue
print "before: ", darkestvalue

mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=res)

brightestpixel_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=brightestpixel)
brightestvalue_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=brightestvalue)

darkestpixel_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=darkestpixel)
darkestvalue_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=darkestvalue)

prg = cl.Program(ctx, """
    __kernel void find(__global const int *a, __global int *brightestvalue, __global int *darkestvalue, __global int *brightestpixel, __global int *darkestpixel)
    {
			int rowid = get_global_id(0);
			int colid = get_global_id(1);

			int ncols = %d;

			int index = rowid * ncols + colid;

			if(a[index] > brightestvalue[0])
			{
				brightestvalue[0] = a[index];
				brightestpixel[0] = rowid;
				brightestpixel[1] = colid;
			}

			if(a[index] < darkestvalue[0])
			{
				darkestvalue[0] = a[index];
				darkestpixel[0] = rowid;
				darkestpixel[1] = colid;
			}
    }
    """ % (dim[1])).build()

prg.find(queue, (dim[0], dim[1]), None, a_buf, brightestvalue_buf, darkestvalue_buf, brightestpixel_buf, darkestpixel_buf)


cl.enqueue_copy(queue, brightestpixel, brightestpixel_buf)
cl.enqueue_copy(queue, brightestvalue, brightestvalue_buf)

cl.enqueue_copy(queue, darkestpixel, darkestpixel_buf)
cl.enqueue_copy(queue, darkestvalue, darkestvalue_buf)

print "--------------------------------------------------"
print "brightestpixel: ", brightestpixel
print "brightestvalue: ", brightestvalue

print "darkestpixel: ", darkestpixel
print "darkestvalue: ", darkestvalue
