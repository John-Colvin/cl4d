import std.stdio;
import cl4d.all;

import std.exception : enforce;
import std.file : readText;
import std.conv : to;
import std.math : sin, abs;
import std.random : uniform;

static this()
{
    DerelictCL.load();
}

void main()
{    
    size_t imageHeight = 500, imageWidth = 500;
    auto inputImage = testImage(imageHeight, imageWidth);
    
    // Size of the input and output images on the host
    auto dataSize = imageHeight * imageWidth;
    
    // Output image on the host
    auto outputImage = new float[dataSize];
    auto refImage = new float[dataSize];
    
    enum uint filterWidth = 3;
    enum uint filterSize  = filterWidth*filterWidth;  // Assume a square kernel
    //boxcar smooth
    enum float q = 1.0/filterSize;
    float[9] filter = q;
    
    auto platforms = CLHost.getPlatforms();
    enforce(platforms.length > 0);
    auto platform = platforms[0];

    DerelictCL.reload(platform.clVersionId);

    debug writefln("%s\n\t%s\n\t%s\n\t%s\n\t%s\n", platform.name, platform.vendor, platform.clVersion, 
                   platform.profile, platform.extensions);
    
    auto devices = platform.allDevices;
    enforce(devices.length > 0);

    debug foreach(CLDevice device; devices)
    {
	writefln("%s\n\t%s\n\t%s\n\t%s\n\t%s\n", device.name, device.vendor, device.driverVersion,
		     device.clVersion, device.profile, device.extensions);
    }
    	
    auto context = CLContext(devices);

    auto queue = CLCommandQueue(context, devices[0]);
    debug writeln("\ncreated queue\n");

    // The image format describes how the data will be stored in memory
    cl_image_format format;
    format.image_channel_order     = CL_R;     // single channel
    format.image_channel_data_type = CL_FLOAT; // float data type
    
    // Create space for the source image on the device
    auto d_inputImage = CLImage(context, 0, format,
                                cl_image_desc(CL_MEM_OBJECT_IMAGE2D, imageWidth, imageHeight), 
                                null);
    debug writeln("\ncreated input image buffer\n");

    // Create space for the output image on the device
    auto d_outputImage = CLImage(context, 0, format,
				cl_image_desc(CL_MEM_OBJECT_IMAGE2D, imageWidth, imageHeight), 
				null);
    debug writeln("\ncreate output image buffer\n");

    // Create space for the 7x7 filter on the device
    auto d_filter = CLBuffer(context, 0, filterSize * float.sizeof, null);

    debug writeln("\nallocated all buffers\n");

    // Copy the source image to the device
    size_t[3] origin = [0, 0, 0];  // Offset within the image to copy from
    size_t[3] region = [imageWidth, imageHeight, 1]; // Elements to per dimension

    auto event = queue.enqueueWriteImage(d_inputImage, false, origin, region, cast(void*)inputImage.ptr);
    event.wait();
    // Copy the 7x7 filter to the device
    event = queue.enqueueWriteBuffer(d_filter, false, 0, filterSize * float.sizeof, filter.ptr);
    event.wait();
    // Create the image sampler
    auto sampler = CLSampler(context, false, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST);


    auto source = readText("convolution.cl");
    debug writeln("\nread convolution.cl\n");
    
    // Create a program object with source and build it
    auto program = context.createProgram(clDbgInfo!() ~ source);
    program.build("-O3 -w -Werror");
    debug writeln("\nbuilt program\n");
    
    // Create the kernel object
    auto kernel = CLKernel(program, "convolution");
    debug writeln("\ncreated kernel\n");
    
    kernel.setArgs(d_inputImage,
		   d_outputImage,
		   to!int(imageHeight),
		   to!int(imageWidth), 
		   d_filter,
		   to!int(filterWidth),
		   sampler
	);
    debug writeln("\nset kernel args\n");
    
    // Set the work item dimensions
    auto global = NDRange(imageWidth, imageHeight);
    event = queue.enqueueNDRangeKernel(kernel, global);

    //wait for work to finish
    event.wait();

    // Read the image back to the host
    queue.enqueueReadImage(d_outputImage, true, origin,
			   region, cast(void*)outputImage.ptr);
        
    // Compute the reference image
    for(size_t i = 0; i < imageHeight; i++)
    {
        for(size_t j = 0; j < imageWidth; j++)
        {
            refImage[i*imageWidth+j] = 0;
        }
    }

    // Iterate over the rows of the source image
    int halfFilterWidth = (filterWidth-1)/2;
    float sum;
    for(size_t i = 0; i < imageHeight; i++)
    {
        // Iterate over the columns of the source image
        for(size_t j = 0; j < imageWidth; j++)
        {
            sum = 0; // Reset sum for new source pixel
            // Apply the filter to the neighborhood
            for(int k = - halfFilterWidth; k <= halfFilterWidth; k++)
            {
                for(int l = - halfFilterWidth; l <= halfFilterWidth; l++)
                {
		    auto imVOff = (cast(int)(i + k)).clip(0, imageHeight - 1);
		    auto imHOff = (cast(int)(j + l)).clip(0, imageWidth - 1);
		    
		    sum += inputImage[imVOff*imageWidth + imHOff]
                         * filter[(k+halfFilterWidth)*filterWidth + l + halfFilterWidth];
		}
            }
            refImage[i*imageWidth+j] = sum;
        }
    }
    
    bool failed = false;
    for(size_t i = 0; i < imageHeight; i++)
    {
	for(size_t j = 0; j < imageWidth; j++)
	{
	    if(abs(outputImage[i*imageWidth+j]-refImage[i*imageWidth+j]) > 0.01)
	    {
		printf("Results are INCORRECT\n");
		printf("Pixel mismatch at <%d,%d> (%f vs. %f)\n", i, j,
		       outputImage[i*imageWidth+j], refImage[i*imageWidth+j]);
		failed = true;
	    }
	    if(failed) break;
	}
	if(failed) break;
    }
    if(!failed)
    {
	printf("Results are correct\n");
    }
/+
    printData(inputImage, imageHeight, imageWidth);
    writeln();
    printData(outputImage, imageHeight, imageWidth);
    writeln();
    printData(refImage, imageHeight, imageWidth);+/
}

void printData(float[] data, size_t height, size_t width)
{
    foreach(i; 0..height)
    {
	foreach(j; 0..width)
	{
	    writef("%8.5f ",data[i*width + j]);
	}
	writeln();
    }
}


// This function takes a positive integer and rounds it up to
// the nearest multiple of another provided integer
auto roundUp(T)(T value, T multiple)
if(isUnsigned!T)
{
    // Determine how far past the nearest multiple the value is
    auto remainder = value % multiple;
    
    // Add the difference to make the value a multiple
    if(remainder != 0)
    {
	    value += (multiple-remainder);
    }
    
    return value;
}

float[] testImage(in size_t imageHeight, in size_t imageWidth)
{
    auto im = new float[imageHeight * imageWidth];
    foreach(i; 0 .. imageHeight)
    {
	    foreach(j; 0 .. imageWidth)
	    {
	        im[i*imageWidth + j] = sin(i * j / 10.0)/* + uniform(0.0,0.2)*/;
	    }
    }
    return im;
}

auto clipUp(T0, T1)(T0 val, T1 thresh)
{
    if(val >= thresh)
    {
	return val;
    }
    else
    {
	return thresh;
    }
}

auto clipDown(T0, T1)(T0 val, T1 thresh)
{
    if(val <= thresh)
    {
	return val;
    }
    else
    {
	return thresh;
    }
}

auto clip(T0,T1,T2)(T0 val, T1 bot, T2 top)
{
    return val.clipUp(bot).clipDown(top);
}
