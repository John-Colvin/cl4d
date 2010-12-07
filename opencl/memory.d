/**
 *	cl4d - object-oriented wrapper for the OpenCL C API
 *	written in the D programming language
 *
 *	Copyright:
 *		(C) 2009-2010 Andreas Hollandt
 *
 *	License:
 *		see LICENSE.txt
 */
module opencl.memory;

import opencl.c.cl;
import opencl.context;
import opencl.error;
import opencl.wrapper;

/**
 *	Memory objects are reserved regions of global device memory that can serve as containers for your data
 */
abstract class CLMemory
{
	mixin(CLWrapper("cl_mem", "clGetMemObjectInfo"));

public:
	version(CL_VERSION_1_1)
	/**
	 *	registers a user callback function with a memory object
	 *	Each call registers the specified user callback function on a callback stack associated with memobj.
	 *	The registered user callback functions are called in the reverse order in which they were registered.
	 *	The user callback functions are called and then the memory object's resources are freed and the memory object is deleted.
	 *
	 *	This provides a mechanism to be notified when the memory referenced by host_ptr, specified when the memory object was created
	 *	and used as the storage bits for the memory object, can be reused or freed
	 */
	void setDestructorCallback(mem_notify_fn fpNotify, void* userData = null)
	{
		cl_int res = clSetMemObjectDestructorCallback(this.getObject(), fpNotify, userData);
		
		mixin(exceptionHandling(
			["CL_INVALID_MEM_OBJECT",	""],
			["CL_INVALID_VALUE",		"fpNotify is null"],
			["CL_OUT_OF_RESOURCES",		""],
			["CL_OUT_OF_HOST_MEMORY",	""]
		));
	}

@property
{
	version(CL_VERSION_1_1)
	//! ditto
	void destructorCallback(mem_notify_fn fpNotify)
	{
		setDestructorCallback(fpNotify);
	}

	//! context specified when memory object was created
	CLContext context()
	{
		return new CLContext(getInfo!cl_context(CL_MEM_CONTEXT));
	}

	//! Map count
	cl_uint mapCount()
	{
		return getInfo!cl_uint(CL_MEM_MAP_COUNT);
	}

	/**
	 *	If memobj is a Buffer or Image = hostPtr argument value used for creation
	 *	For a sub-buffer, return the hostPtr + origin value specified when created.
	 */
	void* hostPtr()
	{
		return getInfo!(void*)(CL_MEM_HOST_PTR);
	}

	//! actual size of this CLMemory's data store in bytes
	size_t size()
	{
		return getInfo!size_t(CL_MEM_SIZE);
	}

	//! the flags argument value specified when memobj was created
	cl_mem_flags flags()
	{
		return getInfo!cl_mem_flags(CL_MEM_FLAGS);
	}
} // of @property
} // of CLMemory