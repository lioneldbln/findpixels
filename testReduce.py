import pyopencl as cl
import os
import numpy

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

sizeOfTheArray = 8
myarrayOne = numpy.array(range(sizeOfTheArray), dtype=numpy.int32)
print myarrayOne
output = numpy.zeros(sizeOfTheArray, dtype=numpy.int32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
inputOne_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=myarrayOne)
output_buffer = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=output)
local_array = cl.LocalMemory(sizeOfTheArray*32)

prg = cl.Program(ctx, """
	__kernel void reduce0(__global const int *g_idata , __global int *g_odata, unsigned int arraySize, __local int *ldata)
		{
			unsigned int lid = get_local_id(0);
			unsigned int i = get_global_id(0);
			ldata[lid] = (i < arraySize) ? g_idata[i] : 0;
			barrier(CLK_LOCAL_MEM_FENCE);
			for (unsigned int s=1; s < get_local_size(0); s*=2)
			{
				if ((lid % (2*s)) == 0)
					ldata[lid] += ldata[lid + s];
				barrier(CLK_LOCAL_MEM_FENCE);
			}
			if (lid == 0) g_odata[get_group_id(0)] = ldata[0];
		}
		""").build()

prg.reduce0(queue, myarrayOne.shape, myarrayOne.shape, inputOne_buffer, output_buffer, numpy.uint32(myarrayOne.size), local_array)

cl.enqueue_copy(queue, output, output_buffer)

print myarrayOne
print output

