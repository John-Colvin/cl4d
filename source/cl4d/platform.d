/**
 *  cl4d - object-oriented wrapper for the OpenCL C API
 *  written in the D programming language
 *
 *  Copyright:
 *      (C) 2009-2011 Andreas Hollandt
 *
 *  License:
 *      see LICENSE.txt
 */
module cl4d.platform;

public import derelict.opencl.cl;
import cl4d.device;
import cl4d.error;
import cl4d.wrapper;

import std.algorithm, std.conv;

//! Platform collection
alias CLObjectCollection!CLPlatform CLPlatforms;

//! Platform class
struct CLPlatform
{
    mixin(CLWrapper!("cl_platform_id", "clGetPlatformInfo"));

public:
    /// get the platform name
    @property string name()
    {
	return getStringInfo(CL_PLATFORM_NAME);
    }
    
    /// get platform vendor
    @property string vendor()
    {
	return getStringInfo(CL_PLATFORM_VENDOR);
    }

    /// get platform version
    @property string clVersion()
    {
	return getStringInfo(CL_PLATFORM_VERSION);
    }

    @property CLVersion clVersionId()
    {
	auto id = this.clVersion();
	id.findSkip(" ");
	auto ver = id.findSplit(" ")[0];
        auto tmp = ver.findSplit(".");
	auto major = tmp[0].to!ubyte();
	auto minor = tmp[2].to!ubyte();
	
	assert(major == 1);
	switch(minor)
	{
	    case 0:
		return CLVersion.CL10;
	    case 1:
		return CLVersion.CL11;
	    case 2:
		return CLVersion.CL12;
	    default:
		assert(false);
	}
    }

    /// get platform profile
    string profile()
    {
	return getStringInfo(CL_PLATFORM_PROFILE);
    }

    /// get platform extensions
    string extensions()
    {
	return getStringInfo(CL_PLATFORM_EXTENSIONS);
    }
    
    /// returns a list of all devices available on the platform matching deviceType
    package CLDevices getDevices(cl_device_type deviceType)
    {
        cl_uint numDevices;
        cl_errcode res;
        
        // get number of devices
        res = clGetDeviceIDs(this._object, deviceType, 0, null, &numDevices);
        
        mixin(exceptionHandling(
		  ["CL_INVALID_PLATFORM",     ""],
		  ["CL_INVALID_DEVICE_TYPE",  "There's no such device type"],
		  ["CL_DEVICE_NOT_FOUND",     "Couldn't find an OpenCL device matching the given type"]
		  ));
        
        // get device IDs
        auto deviceIDs = new cl_device_id[numDevices];

        res = clGetDeviceIDs(this._object, deviceType, cast(cl_uint) deviceIDs.length, deviceIDs.ptr, null);
        if(res != CL_SUCCESS)
	{
            throw new CLException(res);
	}        

        // create CLDevice array
        return CLDevices(deviceIDs);
    }
    
    /// returns a list of all devices
    CLDevices allDevices()  {return getDevices(CL_DEVICE_TYPE_ALL);}
    
    /// returns a list of all CPU devices
    CLDevices cpuDevices()  {return getDevices(CL_DEVICE_TYPE_CPU);}
    
    /// returns a list of all GPU devices
    CLDevices gpuDevices()  {return getDevices(CL_DEVICE_TYPE_GPU);}
    
    /// returns a list of all accelerator devices
    CLDevices accelDevices() {return getDevices(CL_DEVICE_TYPE_ACCELERATOR);}

    /**
     * allows the implementation to release the resources allocated by the OpenCL compiler for
     * platform. This is a hint from the application and does not guarantee that the compiler will not be
     * used in the future or that the compiler will actually be unloaded by the implementation. Calls to
     * clBuildProgram, clCompileProgram or clLinkProgram after clUnloadPlatformCompiler
     * will reload the compiler, if necessary, to build the appropriate program executable
     */
    void unloadCompiler()
    {
        if(DerelictCL.loadedVersion < CLVersion.CL12)
	{
            throw new CLVersionException();
	}

        clUnloadPlatformCompiler(_object);
    }
}


/**
* Macro to facilitate debugging
* Usage:
* Place clDbgInfo!() on the line before the first line of your source.
* The first line ends with: CL_PROGRAM_STRING_BEGIN \"
* Each line thereafter of OpenCL C source must have a line end
* The last line is empty;
*
* Example:
*
* string code = clDbgInfo!() ~ q{
* kernel void foo( int a, float * b )
* {
* // my comment
* *b[ get_global_id(0)] = a;
* }
* };
*
* This should correctly set up the line, (column) and file information for your source
* string so you can do source level debugging.
*/
template clDbgInfo(ulong line = __LINE__, string file = __FILE__)
{
    import std.conv : to;
    enum clDbgInfo = "#line " ~ line.to!string() ~ " \"" ~ file ~ "\" \n\n";
}
