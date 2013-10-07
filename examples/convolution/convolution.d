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
    immutable size_t imageHeight = 500, imageWidth = 500;
    auto inputImage = testImage(imageHeight, imageWidth);
    
    // Size of the input and output images on the host
    auto dataSize = imageHeight * imageWidth;
    
    // Output image on the host
    auto outputImage = new float[dataSize];
    
    enum uint filterWidth = 3;
    enum uint filterSize  = filterWidth*filterWidth;  // Assume a square kernel
    //boxcar smooth
    enum float q = 1.0/filterSize;
    float[9] filter = q;
    
    auto platforms = CLHost.getPlatforms();
    enforce(platforms.length > 0);
    auto platform = platforms[0];

    debug writeln(DerelictCL.loadedVersion);
    DerelictCL.reload(platform.clVersionId);
    debug writeln(DerelictCL.loadedVersion);

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
    debug writeln("\ncreated context\n");


    auto queue = CLCommandQueue(context, devices[0]);

    // The image format describes how the data will be stored in memory
    cl_image_format format;
    format.image_channel_order     = CL_R;     // single channel
    format.image_channel_data_type = CL_FLOAT; // float data type

    // Create space for the source image on the device
    auto d_inputImage = CLImage(context, 0, format,
                                cl_image_desc(CL_MEM_OBJECT_IMAGE2D, imageWidth, imageHeight), 
                                null);

    // Create space for the output image on the device
    auto d_outputImage = CLImage(context, 0, format,
				cl_image_desc(CL_MEM_OBJECT_IMAGE2D, imageWidth, imageHeight), 
				null);

    // Create space for the filter on the device
    auto d_filter = CLBuffer(context, 0, filterSize * float.sizeof, null);

    // Copy the source image to the device
    size_t[3] origin = [0, 0, 0];  // Offset within the image to copy from
    size_t[3] region = [imageWidth, imageHeight, 1]; // Elements to per dimension

    queue.enqueueWriteImage(d_inputImage, true, origin, region, cast(void*)inputImage.ptr);

    // Copy the 7x7 filter to the device
    queue.enqueueWriteBuffer(d_filter, true, 0, filterSize * float.sizeof, filter.ptr);

    debug writeln("\ndata moved to device\n");

    // Create the image sampler
    auto sampler = CLSampler(context, false, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST);

    debug writeln("\nsampler created\n");

    auto source = readText("convolution.cl");
    
    // Create a program object with source and build it
    auto program = context.createProgram(clDbgInfo!() ~ source);
    
    debug writeln("\nprogram created\n");

    program.build("-w -Werror");
    
    debug writeln("\nprogram built\n");

    // Create the kernel object
    auto kernel = CLKernel(program, "convolution");
    
    kernel.setArgs(d_inputImage,
		   d_outputImage,
		   to!int(imageHeight),
		   to!int(imageWidth), 
		   d_filter,
		   to!int(filterWidth),
		   sampler
	);
    
    // Set the work item dimensions
    auto global = NDRange(imageWidth, imageHeight);
    auto event = queue.enqueueNDRangeKernel(kernel, global);

    //wait for work to finish
    event.wait();

    // Read the image back to the host
    queue.enqueueReadImage(d_outputImage, true, origin,
			   region, cast(void*)outputImage.ptr);


    //check the result is correct
    verify(inputImage, imageHeight, imageWidth, filter, filterWidth, outputImage);
/+
    printData(inputImage, imageHeight, imageWidth);
    writeln();
    printData(outputImage, imageHeight, imageWidth);
    writeln();
    printData(refImage, imageHeight, imageWidth);+/
}


/**
 * Extras for verification of results and printing. Not used at all for the openCL code itself
 */

void verify(float[] inputImage, size_t imageHeight, size_t imageWidth, float[] filter, size_t filterWidth, float[] outputImage)
{
    auto refImage = new float[inputImage.length];
    
    // Iterate over the rows of the source image
    long halfFilterWidth = (filterWidth-1)/2;
    foreach(i; 0 .. imageHeight)
    {
        // Iterate over the columns of the source image
        foreach(j; 0 .. imageWidth)
        {
            float sum = 0; // Reset sum for new source pixel
            // Apply the filter to the neighborhood
            foreach(k; -halfFilterWidth .. halfFilterWidth + 1)
            {
                foreach(l; -halfFilterWidth .. halfFilterWidth + 1)
                {
		    auto imVOff = (cast(long)(i + k)).clip(0, imageHeight - 1);
		    auto imHOff = (cast(long)(j + l)).clip(0, imageWidth - 1);
		    
		    sum += inputImage[imVOff*imageWidth + imHOff]
                         * filter[(k+halfFilterWidth)*filterWidth + l + halfFilterWidth];
		}
            }
            refImage[i*imageWidth+j] = sum;
        }
    }
    
    foreach(i; 0 .. imageHeight)
    {
	foreach(j; 0 .. imageWidth)
	{
	    if(abs(outputImage[i*imageWidth+j] - refImage[i*imageWidth+j]) > 0.01)
	    {
		printf("Results are INCORRECT\n");
		printf("Pixel mismatch at <%d,%d> (%f vs. %f)\n", i, j,
		       outputImage[i*imageWidth+j], refImage[i*imageWidth+j]);
		return;
	    }
	}
    }
    
    printf("Results are correct\n");
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
