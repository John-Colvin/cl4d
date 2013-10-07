module vectorAdd;

import cl4d.all;

import std.stdio;
import std.exception;
import std.datetime;

static this()
{
    DerelictCL.load();
}

void main(string[] args)
{
    auto platforms = CLHost.getPlatforms();
    if (platforms.length < 1)
    {
	writeln("No platforms available.");
	return;
    }

    auto platform = platforms[0];

    debug writefln("%s\n\t%s\n\t%s\n\t%s\n\t%s", platform.name, platform.vendor, platform.clVersion, 
	     platform.profile, platform.extensions);
    
    debug writeln(DerelictCL.loadedVersion);
    DerelictCL.reload(platform.clVersionId);
    debug writeln(DerelictCL.loadedVersion);

    auto devices = platform.allDevices;
    if (devices.length < 1)
    {
	writeln("No devices available.");
	return;
    }
	
	
    debug foreach(CLDevice device; devices)
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
    debug writeln(program.buildLog(devices[0]));
	
    debug writeln("program built");

    auto kernel = CLKernel(program, "sum");
    debug writeln("kernel created");
	
    // create input vectors
    enum VECTOR_SIZE = 100000;
    int[] va = new int[VECTOR_SIZE]; foreach(int i, ref e; va) e = i;
    int[] vb = new int[VECTOR_SIZE]; foreach(int i, ref e; vb) e = cast(int) vb.length - i;
    int[] vc = new int[VECTOR_SIZE];
	
    // Create CL buffers
    auto bufferA = CLBuffer(context, CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR, VECTOR_SIZE * int.sizeof, va.ptr);
    auto bufferB = CLBuffer(context, CL_MEM_READ_ONLY  | CL_MEM_USE_HOST_PTR, VECTOR_SIZE * int.sizeof, vb.ptr);
    auto bufferC = CLBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, VECTOR_SIZE * int.sizeof, vc.ptr);
    debug writeln("allocated buffers");

    // Set arguments to kernel
    kernel.setArgs(bufferA, bufferB, bufferC);
    debug writeln("arguments set");
	
    // Run the kernel on specific ND range
    auto global	= NDRange(VECTOR_SIZE);
    
    version(bench)
    {
        auto sw = StopWatch();
	sw.start();
    }
    CLEvent execEvent = queue.enqueueNDRangeKernel(kernel, global);

    // wait for the kernel to be executed
    execEvent.wait();

    version(bench)
    {
        sw.stop();
        writeln("\n\n",sw.peek().usecs,"\n\n");
    }

    //map the memory back to the host.
    auto tmp = cast(ubyte[])vc;
    queue.enqueueMapBuffer(bufferC, true, CL_MAP_READ, 0, VECTOR_SIZE * int.sizeof, tmp);
	
    foreach(i,e; vc)
    {
        enforce(e == VECTOR_SIZE);
    }
}
