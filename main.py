import pyopencl as cl
import os
import Image
import step1

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

if __name__ == "__main__":
	ctx = cl.create_some_context()
	queue = cl.CommandQueue(ctx)
	img = Image.open("test.bmp")
	step1.Find(ctx, queue, img)