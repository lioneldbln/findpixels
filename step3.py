import pyopencl as cl
import numpy as np
import os
import Image

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
 
img = Image.open("test.bmp")
imgArray = np.asarray(img).astype(np.uint8)
dimImgArray = imgArray.shape
sumRgbArrayRes = np.zeros((dimImgArray[0], dimImgArray[1]),np.int32)
 
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
 
mf = cl.mem_flags
imgArray_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=imgArray)
sumRGBArrayRes_buf = cl.Buffer(ctx, mf.WRITE_ONLY, sumRgbArrayRes.nbytes)
 
prg = cl.Program(ctx, """
    __kernel void sumRGB(__global const uchar *imgArray, __global int *sumRGBArrayRes)
    {
      int rowid = get_global_id(0);
      int colid = get_global_id(1);

		int ncols = %d;
		int npix = %d; //number of pixels, 3 for RGB 4 for RGBA

		int index = rowid * ncols * npix + colid * npix;
		int indexRes = rowid * ncols + colid;
		sumRGBArrayRes[indexRes] = imgArray[index + 0] + imgArray[index + 1] + imgArray[index + 2];
    }
    """ % (dimImgArray[1], dimImgArray[2])).build()
 
prg.sumRGB(queue, (dimImgArray[0], dimImgArray[1]), None, imgArray_buf, sumRGBArrayRes_buf)
 
cl.enqueue_copy(queue, sumRgbArrayRes, sumRGBArrayRes_buf)
 
print imgArray
print "--------------------------------------------------"
print sumRgbArrayRes

dimSumRgbArrayRes = sumRgbArrayRes.shape

brightestPixel = np.zeros((1, 2), np.int32)
brightestValue = np.zeros((1), np.int32)

darkestPixel = np.zeros((1, 2), np.int32)
darkestValue = np.zeros((1), np.int32)

brightestValue[0] = 150;
darkestValue[0] = 88;

print "before: ", brightestValue
print "before: ", darkestValue

mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sumRgbArrayRes)

brightestpixel_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=brightestPixel)
brightestvalue_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=brightestValue)

darkestpixel_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=darkestPixel)
darkestvalue_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=darkestValue)

prg = cl.Program(ctx, """
    __kernel void find(__global const int *imgArray, __global int *brightestValue, __global int *darkestValue, __global int *brightestPixel, __global int *darkestPixel)
    {
			int rowid = get_global_id(0);
			int colid = get_global_id(1);

			int ncols = %d;

			int index = rowid * ncols + colid;

			if(imgArray[index] > brightestValue[0])
			{
				brightestValue[0] = imgArray[index];
				brightestPixel[0] = rowid;
				brightestPixel[1] = colid;
			}

			if(imgArray[index] < darkestValue[0])
			{
				darkestValue[0] = imgArray[index];
				darkestPixel[0] = rowid;
				darkestPixel[1] = colid;
			}
    }
    """ % (dimSumRgbArrayRes[1])).build()

prg.find(queue, (dimSumRgbArrayRes[0], dimSumRgbArrayRes[1]), None, a_buf, brightestvalue_buf, darkestvalue_buf, brightestpixel_buf, darkestpixel_buf)


cl.enqueue_copy(queue, brightestPixel, brightestpixel_buf)
cl.enqueue_copy(queue, brightestValue, brightestvalue_buf)

cl.enqueue_copy(queue, darkestPixel, darkestpixel_buf)
cl.enqueue_copy(queue, darkestValue, darkestvalue_buf)

print "--------------------------------------------------"
print "brightestPixel: ", brightestPixel
print "brightestValue: ", brightestValue

print "darkestPixel: ", darkestPixel
print "darkestValue: ", darkestValue
