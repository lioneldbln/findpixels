import pyopencl as cl
import os
import core
import sys

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

if __name__ == "__main__":
	ctx = cl.create_some_context()
	queue = cl.CommandQueue(ctx)
	core.FindBrightestdarkestPixel(ctx, queue, sys.argv[1])