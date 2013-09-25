import std.stdio;
import cl4d.all;
import std.exception : enforce;
import std.file : readText;
import std.math;
import std.random;

static this()
{
    DerelictCL.load();
    writeln(DerelictCL.loadedVersion);
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

void main()
{    
    size_t imageHeight = 50, imageWidth = 50;
    //read in image
    auto inputImage = testImage(imageHeight, imageWidth);
    
    // Size of the input and output images on the host
    auto dataSize = imageHeight * imageWidth;
    
    // Output image on the host
    auto outputImage = new float[dataSize];
    auto refImage = new float[dataSize];
    
    // 45 degree motion blur
    float[49] filter =
	[0, 0, 0, 0, 0, 0, 0,
	 0, 0, 0, 0, 0, 0, 0,
	 0, 0,-1, 0, 1, 0, 0,
	 0, 0,-2, 0, 2, 0, 0,
	 0, 0,-1, 0, 1, 0, 0,
	 0, 0, 0, 0, 0, 0, 0,
	 0, 0, 0, 0, 0, 0, 0];
    
    // The convolution filter is 7x7
    uint filterWidth = 7;  
    uint filterSize  = filterWidth*filterWidth;  // Assume a square kernel
    
    auto platforms = CLHost.getPlatforms();
    enforce(platforms.length > 0);
    auto platform = platforms[0];

    debug writefln("%s\n\t%s\n\t%s\n\t%s\n\t%s\n", platform.name, platform.vendor, platform.clversion, 
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
    debug writeln("created queue");

    // The image format describes how the data will be stored in memory
    cl_image_format format;
    format.image_channel_order     = CL_R;     // single channel
    format.image_channel_data_type = CL_FLOAT; // float data type
    
    // Create space for the source image on the device
    auto d_inputImage = CLImage(context, 0, format,
				cl_image_desc(CL_MEM_OBJECT_IMAGE2D, imageWidth, imageHeight), 
				null);
    debug writeln("created input image buffer");

    // Create space for the output image on the device
    auto d_outputImage = CLImage(context, 0, format,
				cl_image_desc(CL_MEM_OBJECT_IMAGE2D, imageWidth, imageHeight), 
				null);
    debug writeln("create output image buffer");

    // Create space for the 7x7 filter on the device
    auto d_filter = CLBuffer(context, 0, filterSize * float.sizeof, null);

    debug writeln("allocated all buffers");

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
    debug writeln("read convolution.cl");
    
    // Create a program object with source and build it
    auto program = context.createProgram(clDbgInfo!() ~ source);
    program.build("-O3 -w -Werror");
    
    
    // Create the kernel object
    auto kernel = CLKernel(program, "convolution");
    
    kernel.setArgs(d_inputImage,
		   d_outputImage,
		   imageHeight,
		   imageWidth, 
		   d_filter,
		   filterWidth,
		   sampler
	);
    
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
    int halfFilterWidth = filterWidth/2;
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
		    if(i+k >= 0 && i+k < imageHeight && j+l >= 0 && j+l < imageWidth)
		    {
			sum += inputImage[(i+k)*imageWidth + j+l]
			       * filter[(k+halfFilterWidth)*filterWidth 
			       + l+halfFilterWidth];
		    }
		}
	    }
	    refImage[i*imageWidth+j] = sum;
	}
    }
    
    int failed = 0;
    for(size_t i = 0; i < imageHeight; i++)
    {
	for(size_t j = 0; j < imageWidth; j++)
	{
	    if(abs(outputImage[i*imageWidth+j]-refImage[i*imageWidth+j]) > 0.01)
	    {
		printf("Results are INCORRECT\n");
		printf("Pixel mismatch at <%d,%d> (%f vs. %f)\n", i, j,
		       outputImage[i*imageWidth+j], refImage[i*imageWidth+j]);
		failed = 1;
	    }
	    if(failed) break;
	}
	if(failed) break;
    }
    if(!failed)
    {
	printf("Results are correct\n");
    }
}
