import pyopencl as cl
import numpy as np
from time import time
import Image

def Do(ctx, queue, file):	
	img = Image.open(file)
	imgArray = np.asarray(img).astype(np.uint8)
	sumRgbArray = np.zeros((imgArray.shape[0], imgArray.shape[1]),np.int32)

	mf = cl.mem_flags
	imgArray_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=imgArray)
	sumRgbArray_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=sumRgbArray)

	prg = cl.Program(ctx, """
		__kernel void sumRGB(__global const uchar *imgArray, __global int *sumRgbArray, uint ncols, uint npix)
		{
		  int rowid = get_global_id(0);
		  int colid = get_global_id(1);

			int index = rowid * ncols * npix + colid * npix;
			int indexRes = rowid * ncols + colid;
			sumRgbArray[indexRes] = imgArray[index + 0] + imgArray[index + 1] + imgArray[index + 2];
		}
		""").build()

	prg.sumRGB(queue, imgArray.shape, None, imgArray_buf, sumRgbArray_buf, np.uint32(imgArray.shape[1]), np.uint32(imgArray.shape[2]))

	cl.enqueue_copy(queue, sumRgbArray, sumRgbArray_buf)

	brightestPixel = np.zeros((1, 2), np.int32)
	brightestValue = np.zeros((1), np.int32)

	darkestPixel = np.zeros((1, 2), np.int32)
	darkestValue = np.zeros((1), np.int32)

	brightestValue[0] = sumRgbArray[0][0];
	darkestValue[0] = sumRgbArray[0][0];

	mf = cl.mem_flags
	a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sumRgbArray)

	brightestpixel_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=brightestPixel)
	brightestvalue_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=brightestValue)

	darkestpixel_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=darkestPixel)
	darkestvalue_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=darkestValue)

	prg = cl.Program(ctx, """
		__kernel void find(__global const int *sumRgbArray, __global int *brightestValue, __global int *darkestValue, __global int *brightestPixel, __global int *darkestPixel, uint ncols)
		{
				int rowid = get_global_id(0);
				int colid = get_global_id(1);

				int index = rowid * ncols + colid;

				if(sumRgbArray[index] > brightestValue[0])
				{
					brightestValue[0] = sumRgbArray[index];
					brightestPixel[0] = rowid;
					brightestPixel[1] = colid;
				}

				if(sumRgbArray[index] < darkestValue[0])
				{
					darkestValue[0] = sumRgbArray[index];
					darkestPixel[0] = rowid;
					darkestPixel[1] = colid;
				}
		}
		""").build()

	prg.find(queue, sumRgbArray.shape, None, a_buf, brightestvalue_buf, darkestvalue_buf, brightestpixel_buf, darkestpixel_buf, np.uint32(sumRgbArray.shape[1]))

	cl.enqueue_copy(queue, brightestPixel, brightestpixel_buf)
	cl.enqueue_copy(queue, brightestValue, brightestvalue_buf)

	cl.enqueue_copy(queue, darkestPixel, darkestpixel_buf)
	cl.enqueue_copy(queue, darkestValue, darkestvalue_buf)

	print "==============================================================="
	print "Search in", file
	print "---------------------------------------------------------------"
	print "Brightest pixel at:", brightestPixel, "with the value:", imgArray[brightestPixel[0][0]][brightestPixel[0][1]]
	print "---------------------------------------------------------------"
	print "Darkest pixel at:", darkestPixel, "with the value:", imgArray[darkestPixel[0][0]][darkestPixel[0][1]]
	
def FindBrightessdarkestPixel(ctx, queue, file):
	time1 = time()
	Do(ctx, queue, file)
	time2 = time()
	print("===============================================================")
	print "Time the program took: ", time2 - time1, "s"