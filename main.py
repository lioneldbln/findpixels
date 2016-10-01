import pyopencl as cl
import os
import Image
import step1
import step2

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

if __name__ == "__main__":
	ctx = cl.create_some_context()
	queue = cl.CommandQueue(ctx)
	img = Image.open("landscape.jpg")
	step1.Find(ctx, queue, img)
	step2.Find(ctx, queue, img)