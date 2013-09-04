module vectorAdd;

import opencl.all;

import std.stdio;
import std.exception;
import std.datetime;

void main(string[] args)
{
    auto platforms = CLHost.getPlatforms();
    if (platforms.length < 1)
    {
	writeln("No platforms available.");
	return;
    }

    auto platform = platforms[0];

    writefln("%s\n\t%s\n\t%s\n\t%s\n\t%s", platform.name, platform.vendor, platform.clversion, 
	     platform.profile, platform.extensions);
	
    auto devices = platform.allDevices;
    if (devices.length < 1)
    {
	writeln("No devices available.");
	return;
    }
	
	
    foreach(CLDevice device; devices)
    {
	writefln("%s\n\t%s\n\t%s\n\t%s\n\t%s", device.name, device.vendor, device.driverVersion, 
		 device.clVersion, device.profile, device.extensions);
    }
	
    auto context = CLContext(devices);
	
    // Create a command queue and use the first device
    auto queue = CLCommandQueue(context, devices[0]);
    auto program = context.createProgram( 
	clDbgInfo!() ~ q{
	    __kernel void sum(__global const int* a, __global const int* b, __global int* c)
	    {
		int i = get_global_id(0);
		c[i] = a[i] + b[i];
	    }
	});
    program.build("-w -Werror");
    writeln(program.buildLog(devices[0]));
	
    writeln("program built");

    auto kernel = CLKernel(program, "sum");
    writeln("kernel created");
	
    // create input vectors
    immutable VECTOR_SIZE = 10000;
    int[VECTOR_SIZE] va = void; foreach(int i, ref e; va) e = i;
    int[VECTOR_SIZE] vb = void; foreach(int i, ref e; vb) e = cast(int) vb.length - i;
    int[VECTOR_SIZE] vc;
	
    // Create CL buffers
    auto bufferA = CLBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, va.sizeof, va.ptr);
    writeln("created bufferA");
    auto bufferB = CLBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, vb.sizeof, vb.ptr);
    auto bufferC = CLBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, vc.sizeof, vc.ptr);
    writeln("created buffers");

    // Copy lists A and B to the memory buffers
    //	queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, va.sizeof, va.ptr);
    //	queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, vb.sizeof, vb.ptr);
	
    // Set arguments to kernel
    kernel.setArgs(bufferA, bufferB, bufferC);
    writeln("arguments set");
	
    // Run the kernel on specific ND range
    auto global	= NDRange(VECTOR_SIZE);
    //auto local	= NDRange(1);
    auto sw = StopWatch();
    sw.start();
    CLEvent execEvent = queue.enqueueNDRangeKernel(kernel, global);
//    queue.flush();
    // wait for the kernel to be executed
    execEvent.wait();
    sw.stop();
    writeln(sw.peek().usecs);
    // Read buffer vc into a local list
    // TODO: figure out why this call is needed even though CL_MEM_USE_HOST_PTR is used
//    queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, vc.sizeof, vc.ptr);
	
    foreach(i,e; vc)
    {
	enforce(e == VECTOR_SIZE);
    }
}
