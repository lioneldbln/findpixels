import pyopencl as cl
import numpy as np
from time import time

def Do(ctx, queue, img):
	imgArray = np.asarray(img).astype(np.uint8)
	sumRgbArrayRes = np.zeros((imgArray.shape[0], imgArray.shape[1]),np.int32)

	mf = cl.mem_flags
	imgArray_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=imgArray)
	sumRGBArrayRes_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=sumRgbArrayRes)

	prg = cl.Program(ctx, """
		__kernel void sumRGB(__global const uchar *imgArray, __global int *sumRGBArrayRes, uint ncols, uint npix)
		{
		  int rowid = get_global_id(0);
		  int colid = get_global_id(1);

			int index = rowid * ncols * npix + colid * npix;
			int indexRes = rowid * ncols + colid;
			sumRGBArrayRes[indexRes] = imgArray[index + 0] + imgArray[index + 1] + imgArray[index + 2];
		}
		""").build()

	prg.sumRGB(queue, imgArray.shape, None, imgArray_buf, sumRGBArrayRes_buf, np.uint32(imgArray.shape[1]), np.uint32(imgArray.shape[2]))

	cl.enqueue_copy(queue, sumRgbArrayRes, sumRGBArrayRes_buf)

	print imgArray
	print "--------------------------------------------------"
	print sumRgbArrayRes

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
		__kernel void find(__global const int *imgArray, __global int *brightestValue, __global int *darkestValue, __global int *brightestPixel, __global int *darkestPixel, uint ncols)
		{
				int rowid = get_global_id(0);
				int colid = get_global_id(1);

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
		""").build()

	prg.find(queue, sumRgbArrayRes.shape, None, a_buf, brightestvalue_buf, darkestvalue_buf, brightestpixel_buf, darkestpixel_buf, np.uint32(sumRgbArrayRes.shape[1]))

	cl.enqueue_copy(queue, brightestPixel, brightestpixel_buf)
	cl.enqueue_copy(queue, brightestValue, brightestvalue_buf)

	cl.enqueue_copy(queue, darkestPixel, darkestpixel_buf)
	cl.enqueue_copy(queue, darkestValue, darkestvalue_buf)

	print "--------------------------------------------------"
	print "brightestPixel: ", brightestPixel
	print "brightestValue: ", brightestValue

	print "darkestPixel: ", darkestPixel
	print "darkestValue: ", darkestValue
	
def Find(ctx, queue, img):
	time1 = time()
	Do(ctx, queue, img)
	time2 = time()
	print "Execution time of step1: ", time2 - time1, "s"