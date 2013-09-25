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
module cl4d.image;

import derelict.opencl.cl;
import derelict.opencl.cl_gl;
import derelict.opengl3.constants; //needed for GL enum values. GL never actually loaded
import cl4d.context;
import cl4d.error;
import cl4d.memory;
import cl4d.wrapper;

struct CLImage
{
    CLMemory sup;
    alias sup this;

    this(T)(const CLContext context, cl_mem_flags flags, const cl_image_format format, const cl_image_desc description, T host)
    if(is(T U : U[]) || is(T U : U*))
    {
	static if(is(T == typeof(null)))
	{
	    void* ptr = null;
	}
	else static if(is(T U : U[]))
	{
	    //Check dimensions here???
	    auto ptr = cast(void*)host.ptr;
	}
	else
	{
	    auto ptr = cast(void*)host;
	}

	cl_errcode res;
	cl_mem mem;
	if(DerelictCL.loadedVersion >= CLVersion.CL12)
	{
	    mem = clCreateImage(context.cptr, flags, &format, &description, ptr, &res);

	    mixin(exceptionHandling(
		      ["CL_INVALID_CONTEXT",                  "context is not a valid context"],
		      ["CL_INVALID_VALUE",                    "values specified in flags are not valid, or a 1D image buffer is being created and the buffer object was created with CL_MEM_WRITE_ONLY and flags specifies CL_MEM_READ_WRITE or CL_MEM_READ_ONLY, or the buffer object was created with CL_MEM_READ_ONLY and flags specifies CL_MEM_READ_WRITE or CL_MEM_WRITE_ONLY, or flags specifies CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR or CL_MEM_COPY_HOST_PTR, or a 1D image buffer is being created and the buffer object was created with CL_MEM_HOST_WRITE_ONLY and flags specifies CL_MEM_HOST_READ_ONLY, or the buffer object was created with CL_MEM_HOST_READ_ONLY and flags specifies CL_MEM_HOST_WRITE_ONLY, or the buffer object was created with CL_MEM_HOST_NO_ACCESS and flags specifies CL_MEM_HOST_READ_ONLY or CL_MEM_HOST_WRITE_ONLY"],
		      ["CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",  "values specified in format are not valid or format is null"],
		      ["CL_INVALID_IMAGE_DESCRIPTOR",         "values specified in description are not valid or description is null"],
		      ["CL_INVALID_IMAGE_SIZE",               "image dimensions specified in description exceed the minimum maximum image dimensions"],
		      ["CL_INVALID_HOST_PTR",                 "host is null and CL_MEM_USE_HOST_PTR or CL_MEM_COPY_HOST_PTR are set in flags, or host is not null but CL_MEM_COPY_HOST_PTR or CL_MEM_USE_HOST_PTR are not set in flags"],
		      ["CL_IMAGE_FORMAT_NOT_SUPPORTED",       "format is not supported"],
		      ["CL_MEM_OBJECT_ALLOCATION_FAILURE",    "failure to allocate memory for image object"],
		      ["CL_INVALID_OPERATION",                "there are no devices in context that support images (i.e. CL_DEVICE_IMAGE_SUPPORT is CL_FALSE)"],
		      ["CL_OUT_OF_RESOURCES",                 ""],
		      ["CL_OUT_OF_HOST_MEMORY",               ""]
		      ));
	}
	else if(DerelictCL.loadedVersion <= CLVersion.CL11)
	{
	    if(description.image_type == CL_MEM_OBJECT_IMAGE2D)
	    {
		mem = clCreateImage2D(context.cptr, flags, &format, description.image_width, description.image_height, description.image_row_pitch, ptr, &res);

		mixin(exceptionHandling(
			  ["CL_INVALID_CONTEXT",                  ""],
			  ["CL_INVALID_VALUE",                    "invalid image flags"],
			  ["CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",  "values specified in format are not valid or format is null"],
			  ["CL_INVALID_IMAGE_SIZE",               "width or height are 0 OR exceed CL_DEVICE_IMAGE2D_MAX_WIDTH or CL_DEVICE_IMAGE2D_MAX_HEIGHT resp. OR rowPitch is not valid"],
			  ["CL_INVALID_HOST_PTR",                 "hostPtr is null and CL_MEM_USE_HOST_PTR or CL_MEM_COPY_HOST_PTR are set in flags or if hostPtr is not null but CL_MEM_COPY_HOST_PTR or CL_MEM_USE_HOST_PTR are not set in"],
			  ["CL_IMAGE_FORMAT_NOT_SUPPORTED",       "format is not supported"],
			  ["CL_MEM_OBJECT_ALLOCATION_FAILURE",    "couldn't allocate memory for image object"],
			  ["CL_INVALID_OPERATION",                "there are no devices in context that support images (i.e. CL_DEVICE_IMAGE_SUPPORT is CL_FALSE)"],
			  ["CL_OUT_OF_RESOURCES",                 ""],
			  ["CL_OUT_OF_HOST_MEMORY",               ""]
			  ));
	    }
	    else if(description.image_type == CL_MEM_OBJECT_IMAGE3D)
	    {
		mem = clCreateImage3D(context.cptr, flags, &format, 
				      description.image_width, 
				      description.image_height, 
				      description.image_depth, 
				      description.image_row_pitch,
				      description.image_slice_pitch,
				      ptr, &res);

		mixin(exceptionHandling(
			  ["CL_INVALID_CONTEXT",                  ""],
			  ["CL_INVALID_VALUE",                    "invalid image flags"],
			  ["CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",  "values specified in format are not valid or format is null"],
			  ["CL_INVALID_IMAGE_SIZE",               "width or height are 0 or depth <= 1 OR exceed CL_DEVICE_IMAGE3D_MAX_WIDTH or CL_DEVICE_IMAGE3D_MAX_HEIGHT or CL_DEVICE_IMAGE3D_MAX_DEPTH resp. OR rowPitch or slicePitch is not valid"],
			  ["CL_INVALID_HOST_PTR",                 "hostPtr is null and CL_MEM_USE_HOST_PTR or CL_MEM_COPY_HOST_PTR are set in flags or if hostPtr is not null but CL_MEM_COPY_HOST_PTR or CL_MEM_USE_HOST_PTR are not set in"],
			  ["CL_IMAGE_FORMAT_NOT_SUPPORTED",       "format is not supported"],
			  ["CL_MEM_OBJECT_ALLOCATION_FAILURE",    "couldn't allocate memory for image object"],
			  ["CL_INVALID_OPERATION",                "there are no devices in context that support images (i.e. CL_DEVICE_IMAGE_SUPPORT is CL_FALSE)"],
			  ["CL_OUT_OF_RESOURCES",                 ""],
			  ["CL_OUT_OF_HOST_MEMORY",               ""]
			  ));
	    }
	    else
	    {
		throw new CLException(0, "invalid image_type: not available in cl < 1.2"); //NOTE: is 0 ok here?
	    }
	}
	sup = CLMemory(mem);
    }

    /**
     *  creates an OpenCL image object from an OpenGL texture object
     *
     *  Params:
     *      flags   = only CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY and CL_MEM_READ_WRITE may be used
     *      target  = used only to define the image type of texture. No reference to a bound GL 
     *                texture object is made or implied by this parameter. If 3D then must be GL_TEXTURE_3D
     *      miplevel= mipmap level to be used
     *      texobj  = name of a complete GL 3D texture object
     */
    this(const CLContext context, cl_mem_flags flags, cl_GLenum target, cl_GLint  miplevel, cl_GLuint texobj)
    {
        cl_errcode res;
	cl_mem mem;

	if(DerelictCL.loadedVersion >= CLVersion.CL12)
	{
	    mem = clCreateFromGLTexture(context.cptr, flags, target, miplevel, texobj, &res);
	}
	else if(DerelictCL.loadedVersion <= CLVersion.CL11)
	{
	    if(target == GL_TEXTURE_3D)
	    {
		mem = clCreateFromGLTexture3D(context.cptr, flags, target, miplevel, texobj, &res);
	    }
	    else
	    {
		mem = clCreateFromGLTexture2D(context.cptr, flags, target, miplevel, texobj, &res);
	    }
	}

	mixin(exceptionHandling(
		  ["CL_INVALID_CONTEXT",      "context is not a valid context or was not created from a GL context"],
		  ["CL_INVALID_VALUE",        "flags or target not valid"],
		  ["CL_INVALID_MIP_LEVEL",    "miplevel is less than minimum allowed OR OpenGL implementation does not support creating from mipmap levels != 0"],
		  ["CL_INVALID_GL_OBJECT",    "texobj is not a GL texture object whose type matches target OR the specified miplevel of texture is not defined OR width or height of the specified miplevel is zero"],
		  ["CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",  "the OpenGL texture internal format does not map to a supported OpenCL image format"],
		  ["CL_INVALID_OPERATION",    "texobj is a GL texture object created with a border width value greater than zero (OR implementation does not support miplevel values > 0?)"],
		  ["CL_OUT_OF_RESOURCES",     ""],
		  ["CL_OUT_OF_HOST_MEMORY",   ""]
		  ));

	sup = CLMemory(mem);
    }

    this(cl_mem obj)
    {
        sup = CLMemory(obj);
    }

    @property
    {
        //!image format descriptor specified when image was created
        auto format()
        {
            return this.getInfo!(cl_image_format, clGetImageInfo)(CL_IMAGE_FORMAT);
        }
        
        /**
         *  size of each element of the image memory object given by image. An
         *  element is made up of n channels. The value of n is given in cl_image_format descriptor.
         */
        size_t elementSize()
        {
            return this.getInfo!(size_t, clGetImageInfo)(CL_IMAGE_ELEMENT_SIZE);
        }
        
        //! size in bytes of a row of elements of the image object given by image
        size_t rowPitch()
        {
            return this.getInfo!(size_t, clGetImageInfo)(CL_IMAGE_ROW_PITCH);
        }

        /**
         *  calculated slice pitch in bytes of a 2D slice for the 3D image object or size of each image
	 *  in a 1D or 2D image array given by image. 

	 *  For a 1D image, 1D image buffer and 2D image object return 0size in bytes of a 2D slice for
	 *  the 3D image object given by image.
         */
        size_t slicePitch()
        {
            return this.getInfo!(size_t, clGetImageInfo)(CL_IMAGE_SLICE_PITCH);
        }

        //! width in pixels
        size_t width()
        {
            return this.getInfo!(size_t, clGetImageInfo)(CL_IMAGE_WIDTH);
        }

        /**
	 *  Return height of image in pixels.
	 *
	 *  For a 1D image, 1D image buffer and 1D image array object, height = 0.
	 */
        size_t height()
        {
            return this.getInfo!(size_t, clGetImageInfo)(CL_IMAGE_HEIGHT);
        }

        /**
         *  depth of the image in pixels
         *
         *  For a 1D image, 1D image buffer, 2D image or 1D and 2D image array object, depth = 0. 
         */
        size_t depth()
        {
            return this.getInfo!(size_t, clGetImageInfo)(CL_IMAGE_DEPTH);
        }

        /**
	 *  number of images in the image array. If image is not an image array, 0 is returned.
	 */
	//NOTE: What happens for 1.1 ??
	size_t arraySize()
        {
            return this.getInfo!(size_t, clGetImageInfo)(CL_IMAGE_ARRAY_SIZE);
        }

	//NOTE: TODO: Need more of these.

        //! The target argument specified in CLImage2DGL, CLImage3DGL constructors
        cl_GLenum textureTarget()
        {
            return this.getInfo!(cl_GLenum, clGetGLTextureInfo)(CL_GL_TEXTURE_TARGET);
        }

        //! The miplevel argument specified in CLImage2DGL, CLImage3DGL constructors
        cl_GLint mipmapLevel()
        {
            return this.getInfo!(cl_GLint, clGetGLTextureInfo)(CL_GL_MIPMAP_LEVEL);
        }
    } // of @property
}
