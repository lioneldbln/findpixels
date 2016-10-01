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

	brightestPixel = np.zeros((1, 2), np.int32)
	brightestValue = np.zeros((1), np.int32)

	darkestPixel = np.zeros((1, 2), np.int32)
	darkestValue = np.zeros((1), np.int32)

	brightestValue[0] = 150;
	darkestValue[0] = 150;

	mf = cl.mem_flags
	a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sumRgbArrayRes)

	brightestpixel_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=brightestPixel)
	brightestvalue_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=brightestValue)

	darkestpixel_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=darkestPixel)
	darkestvalue_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=darkestValue)
	
	local_Pixels = cl.LocalMemory(4)

	prg = cl.Program(ctx, """
		__kernel void find(__global const int *sumRgbArrayRes, __global int *brightestValue, __global int *darkestValue, __global int *brightestPixel, __global int *darkestPixel, uint ncols, __local int *local_Pixels)
		{				
				int rowid = get_global_id(0);
				int colid = get_global_id(1);

				int index = rowid * ncols + colid;

				if(sumRgbArrayRes[index] > brightestValue[0])
				{
					brightestValue[0] = sumRgbArrayRes[index];
					local_Pixels[0] = rowid;
					local_Pixels[1] = colid;
				}

				if(sumRgbArrayRes[index] < darkestValue[0])
				{
					darkestValue[0] = sumRgbArrayRes[index];
					local_Pixels[2] = rowid;
					local_Pixels[3] = colid;
				}
				
				barrier(CLK_GLOBAL_MEM_FENCE);
				brightestPixel[0] = local_Pixels[0];
				brightestPixel[1] = local_Pixels[1];
				darkestPixel[0] = local_Pixels[2];
				darkestPixel[1] = local_Pixels[3];
		}
		""").build()

	prg.find(queue, sumRgbArrayRes.shape, None, a_buf, brightestvalue_buf, darkestvalue_buf, brightestpixel_buf, darkestpixel_buf, np.uint32(sumRgbArrayRes.shape[1]), local_Pixels)

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
	print "Execution time of step2: ", time2 - time1, "s"