/*
	要在主机(ARM)上运行任务，必须要做四件事情：
		1. 分配和填充用于收发卡数据的缓存。
		2. 在主机(ARM)内存空间和FPGA全局内存间传输缓冲器。
		3. 运行内核以操作这些缓存。
		4. 将内核运算的结果传输回主机(ARM)内存空间，以便ARM处理器访问。
*/

// Host code to run the pre-processing pipeline
// 主机(ARM)端应用程序，用于流水线预处理

#include "xf_headers.h"
#include <sys/time.h>

#include <CL/cl.h>
#include "xcl2.hpp"

// Init function to Load xclbin and get cl kernel and context
// 初始化设置函数，用于加载.xclbin文件并获得CL kernel与context(上下文)
int pp_kernel_init(PPHandle * &handle,
					char *xclbin,
					const char *kernelName,
					int deviceIdx)
{
	PPHandle *my_handle = new PPHandle;
	handle = my_handle = (PPHandle *)my_handle;

	// 允许Multiprocess模式
    char mps_env[] = "XCL_MULTIPROCESS_MODE=1";
   	if (putenv(mps_env) != 0) {
        std::cout << "putenv failed" << std::endl;
    	} //else

	// ------------------------------ 平台层 ----------------------------------

	// Find xilinx device and create clContext
	// [1] 找出系统中的xlinx器件并为其编号(查询平台+设备)
	std::vector<cl::Device> devices = xcl::get_xil_devices();
	cl::Device device = devices[deviceIdx];

	/*
	 [2] 初始化context(上下文) 

		context(上下文)主要包含如下一些硬件和软件资源：
			1. 设备：OpenCL应用需要调用的计算设备。
			2. 内核：在计算设备上执行的并行程序。
			3. 程序对象：内核程序的源代码和可执行文件。
			4. 内存对象：计算设备执行OpenCL程序所需的变量。

		context被成功创建好之后，CL工作环境就被搭建出来了.
	*/
	cl::Context context(device); 

	// 加载xclbin，准备烧录FPGA
	unsigned fileBufSize;
	std::string binaryFile = xclbin;
	auto fileBuf = xcl::read_binary_file(binaryFile, fileBufSize);

	/*
		将bin文件烧录进FPGA。是实际触发编程操作的地方。在编程阶段，运行时将检查器件当前配置。
		如果已经编程，在从xclbin加载器件元数据后就可以返回。如果没有编程，就立即编程器件。
	*/
    cl::Program::Binaries bins{{fileBuf, fileBufSize}}; 

	devices.resize(1);

    //Check for device name
    /*std::string dev_name ("xilinx_u200_xdma_201830_1");
	if(dev_name.compare(devices[0].getInfo<CL_DEVICE_NAME>()) != 0){
		std::cout << "Device Not Supported" << std::endl;
		return -1;
	}*/
	//Create clKernel and clProgram
	//cl_device_info info;

	cl_int errr;

	OCL_CHECK(errr,cl::Program program(context, devices, bins, NULL, &errr));

	std::string kernelName_s = kernelName;
	OCL_CHECK(errr,cl::Kernel krnl(program,kernelName,&errr));

	// ------------------------------ Runtime ----------------------------------

	// [3] 初始化命令队列,命令队列是向device发送指令的信使
	cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE); 

	// [4] 创建内存对象,将分配的缓存映射到OpenCL缓存对象
    cl::Buffer paramsbuf(context, CL_MEM_READ_ONLY, 9*4);

	float params[9];
	//	params[0] = 123.68;
	//	params[1] = 116.78f;
	//	params[2] = 103.94f;
	params[0] = 0;
	params[1] = 0;
	params[2] = 0;
	params[3] = params[4] = params[5] = 0.0;
	int th1=255,th2=255;

	// 设置内核参数
	krnl.setArg(8, paramsbuf);
	krnl.setArg(9, th1);
	krnl.setArg(10, th2);

	q.enqueueWriteBuffer(
                         paramsbuf,
                         CL_TRUE,
                         0,
                         9*4,
                         params);

	my_handle->kernel = krnl;
	my_handle->contxt = context;
	my_handle->device = device;
	my_handle->q = q;
    // my_handle->imageToDevice=imageToDevice;
    // my_handle->imageFromDevice=imageFromDevice;
    my_handle->paramsbuf=paramsbuf;

	if(errr == 0)
	return 0;
	else
	return -1;
}




// pre-processing kernel execution
// 执行预处理kernel
/**
 * @brief Pre-process before image data get into DPU
 *
 * @param handle pointer 
 * @param img input image data
 * @param out_ht output height
 * @param out_wt output weight
 * @param data_ptr output image data
 *
 * @return none
 */
int preprocess(PPHandle * &handle, cv::Mat img, int out_ht, int out_wt, float *data_ptr)
{
	//	struct timeval start_,end_;
	//	struct timeval start_imread,end_imread;
	//	struct timeval start_fx2fl,end_fx2fl;
	//	double lat_ = 0.0f;
	//	double lat_imread = 0.0f;
	//	double lat_fx2fl = 0.0f;
	//  gettimeofday(&start_, 0);
	
	//  CV Mat to store input image and output data
	cv::Mat result;
	// Read input image
	//  gettimeofday(&start_imread, 0);
	//	img = cv::imread(img_name, 1);
			
	// gettimeofday(&end_imread, 0);
	// lat_imread = (end_imread.tv_sec * 1e6 + end_imread.tv_usec) - (start_imread.tv_sec * 1e6 + start_imread.tv_usec);
	// std::cout << "\n\n imread latency " << lat_imread / 1000 << "ms" << std::endl;

	if(!img.data){
		fprintf(stderr,"\n input image not found");
		return -1;
	}
	int in_width,in_height;
	int out_width,out_height;

	in_width = img.cols;
	in_height = img.rows;
	
	// output image dimensions 224x224
	out_height = out_ht;
	out_width = out_wt;

    float scale_height = (float)out_height/(float)in_height;
    float scale_width = (float)out_width/(float)in_width;
	int out_height_resize, out_width_resize;
    if(scale_width<scale_height){
    	out_width_resize = out_width;
    	out_height_resize = (int)((float)(in_height*out_width)/(float)in_width);
    }
    else
    {

    	out_width_resize = (int)((float)(in_width*out_height)/(float)in_height);
    	out_height_resize = out_height;
    }
    int dx = (out_width - out_width_resize)/2;
    int dy = (out_height - out_height_resize)/2;
	
	std::cout << "\nInput ht:wd " << in_height << ":" << in_width << std::endl;
	std::cout << "Output ht:wd " << out_height << ":" << out_width << std::endl;
	std::cout << "Resize ht:wd " << out_height_resize << ":" << out_width_resize << "\n" << std::endl;


//=================================== CL ===================================

	cl::Context context = handle->contxt;
	cl::Kernel krnl = handle->kernel;
	cl::Device device = handle->device;

	/*  
		Buffer creation
	  	创建Buffer,缓存大小和图片适配

	  	新增Flag：CL_MEM_READ_ONLY 和 CL_MEM_WRITE_ONLY 向运行时说明这些缓存对内核而言的可见性。换句话说，主机(ARM)
		写入加速kernel的image数据对内核而言只读。随后，主机从加速卡读回处理好的image，此时对kernel而言，它是只写。
		我们额外地将这些缓存对象添加给向量的目的在于：能一次性传输多个缓存(请注意，我们只给向量添加指针，并非添加数据缓存本身)。
	*/
	cl::CommandQueue q; //(context, device, CL_QUEUE_PROFILING_ENABLE);
	q = handle->q;
	std::vector<cl::Memory> inBufVec, outBufVec, paramasbufvec;
	cl::Buffer imageToDevice(context, CL_MEM_READ_ONLY, in_height*in_width*3);
	cl::Buffer imageFromDevice(context,CL_MEM_WRITE_ONLY,out_height*out_width*3*4);
	cl::Buffer paramsbuf; //(context, CL_MEM_READ_ONLY,9*4);
	paramsbuf = handle->paramsbuf;

	// Set kernel arguments
	// 设置 Kernel 参数

	krnl.setArg(0, imageToDevice);
	krnl.setArg(1, imageFromDevice);
	krnl.setArg(2, in_height);
	krnl.setArg(3, in_width);
	krnl.setArg(4, out_height_resize);
	krnl.setArg(5, out_width_resize);
	krnl.setArg(6, out_height);
	krnl.setArg(7, out_width);

	//	krnl.setArg(6, paramsbuf);
	//	krnl.setArg(7, th1);
	//	krnl.setArg(8, th2);

	// Copy data from host to FPGA
	// 将数据从主机(ARM)端拷贝至FPGA上进行计算(将输入的图片数据写入缓存)
	q.enqueueWriteBuffer(imageToDevice,
                         CL_TRUE,
                         0,
                         in_height*in_width*3,
                         img.data);

	// Profiling Objects
	cl_ulong start = 0;
	cl_ulong end = 0;
	double diff_prof = 0.0f;
	cl::Event event_sp;

	
	/*
		Launch the kernel
		启动 kernel

		接下来，将内核本身添加到命令队列，以便开始执行。一般来说，
		会将传输与内核加入到队列，让它们依次执行，而非同步执行。
	*/
	q.enqueueTask(krnl,NULL,&event_sp); // 如果不想在这一点等待，可以传入NULL而非 cl::Event 对象。
	clWaitForEvents(1, (const cl_event*) &event_sp);

	// Copy data from device to Host
	// 将数据从FPGA端拷贝回主机(ARM)上
	q.enqueueReadBuffer( imageFromDevice,
                         CL_TRUE,
                         0,
                         out_height*out_width*3*4,
                         data_ptr);
                    	//result.data);
	// Profiling
	//	event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START,&start);
	//	event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END,&end);
	//	diff_prof = end-start;
	//	std::cout<<"kernel latency = "<<(diff_prof/1000000)<<"ms"<<std::endl;

	q.finish();

//=================================== end of CL ===================================

	for(int i=0;i<(3*out_width*out_height);i++)
		data_ptr[i]=data_ptr[i]/256;

	// gettimeofday(&end_, 0);

	// lat_ = (end_.tv_sec * 1e6 + end_.tv_usec) - (start_.tv_sec * 1e6 + start_.tv_usec);
	// std::cout << "\n\n Overall latency " << lat_ / 1000 << "ms" << std::endl;

	return 0;
}

