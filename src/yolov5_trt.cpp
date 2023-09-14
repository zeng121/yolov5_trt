
// yolov5进行tensorrt部署的源文件
#include <fstream>
#include <iostream>
#include<opencv2/dnn/dnn.hpp>
#include "yolov5_trt.h"
#include <cassert>
#include"logger.h"
#include"util.h"
#define FACE

using namespace cv;
using namespace nvinfer1;

const char* Input_blob_name = "images";
const char* Output_blob_name = "output0";
YOLOv5::YOLOv5(Configuration config)
{
    confThreshold = config.confThreshold;
    nmsThreshold = config.nmsThreshold;
    objThreshold = config.objThreshold;
    inpHeight = 640;
    inpWidth = 640;

    std::string model_path = config.modelpath;  // 模型权重路径
	// 加载模型
    std::string strTrtName = config.modelpath;      // 加载模型权重
    assert(ifFileExists(strTrtName.c_str()));
    loadTrt(strTrtName);

	// 利用加载的模型获取输入输出信息
	// 使用输入和输出blob名来获取输入和输出索引
    m_iInputIndex = m_CudaEngine->getBindingIndex(Input_blob_name);     // 输入索引
    m_iOutputIndex = m_CudaEngine->getBindingIndex(Output_blob_name);   // 输出  
	//std::cout << " m_iInputIndex " << m_iInputIndex << std::endl;
	//std::cout << " m_iOutputIndex " << m_iOutputIndex << std::endl;
    
	Dims dims_i = m_CudaEngine->getBindingDimensions(m_iInputIndex);  // 输入，
	//Dims dims_i = m_CudaEngine->getTensorShape(Input_blob_name);  // 输入，
    int size1 = dims_i.d[0] * dims_i.d[1] * dims_i.d[2] * dims_i.d[3];   // 展平
	//cout << "size1 " << size1 << endl;
    //m_InputSize = cv::Size(dims_i.d[3], dims_i.d[2]);   // 输入尺寸(W,H)
	
    Dims dims_o = m_CudaEngine->getBindingDimensions(m_iOutputIndex);  // 输出，维度[0,1,2,3]NHWC
	//Dims dims_o = m_CudaEngine->getTensorShape(Output_blob_name);  // 输出，维度[0,1,2,3]NHWC
	
	
    int size2 = dims_o.d[0] * dims_o.d[1] * dims_o.d[2];   // 所有大小
	//cout << "size2 " << size2 << endl;
    ClassNums = dims_o.d[2] - 5;    // [,,classes+5]
    BoxNums = dims_o.d[1];    
	 

	// 分配内存大小
    cudaMalloc(&m_ArrayDevMemory[m_iInputIndex], size1 * sizeof(float));
    m_ArrayHostMemory[m_iInputIndex] = malloc(size1 * sizeof(float));
    m_ArraySize[m_iInputIndex] = size1 *sizeof(float);
    cudaMalloc(&m_ArrayDevMemory[m_iOutputIndex], size2 * sizeof(float));
    m_ArrayHostMemory[m_iOutputIndex] = malloc( size2 * sizeof(float));
    m_ArraySize[m_iOutputIndex] = size2 *sizeof(float);
    //cout << "内存分配成功" <<endl;
	
	

    // 原地构造
    // m_InputWrappers.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1, m_ArrayHostMemory[m_iInputIndex]);
    // m_InputWrappers.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1, m_ArrayHostMemory[m_iInputIndex] + sizeof(float) * dims_i.d[2] * dims_i.d[3] );
    // m_InputWrappers.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1, m_ArrayHostMemory[m_iInputIndex] + 2 * sizeof(float) * dims_i.d[2] * dims_i.d[3]); 
}   


int YOLOv5::loadTrt(const std::string &strName)
{
    TRT_Logger gLogger;
	// 序列化引擎被保留并保存到文件中
    m_CudaRuntime = createInferRuntime(gLogger);    
    std::ifstream fin(strName);
	if(!fin.is_open())
	{
		std::cout<<"engine文件打开失败"<<std::endl;
		return -1;
	}
    auto filesize = FileSize(strName.c_str());
    char* buffer = new char[filesize];
    fin.read(buffer,filesize);
    fin.close();
    //m_CudaEngine = m_CudaRuntime->deserializeCudaEngine(cached_engine.c_str(), cached_engine.size(), nullptr); // runtime对象反序列化，序列化失败会返回nullptr
	m_CudaEngine = m_CudaRuntime->deserializeCudaEngine(buffer, filesize, nullptr);
    if(m_CudaEngine==nullptr)
	{
		std::cout<<"反序列化失败"<<std::endl;
		return -1;
	}
    m_CudaContext = m_CudaEngine->createExecutionContext();  //可以查询引擎获取有关网络的输入和输出的张量信息--维度/数据格式/数据类型
    m_CudaRuntime->destroy();
    return 0;
}


// 初始化


void YOLOv5::UnInit()
{

    for(auto &p: m_ArrayDevMemory)
    {      
        cudaFree(p);
        p = nullptr;            
    }        
    for(auto &p: m_ArrayHostMemory)
    {        
        free(p);
        p = nullptr;        
    }        
    cudaStreamDestroy(m_CudaStream);
    m_CudaContext->destroy();   
    m_CudaEngine->destroy();

}

YOLOv5::~YOLOv5()
{
    UnInit();   
}


void YOLOv5::resize_image(Mat& img)
{
    int src_h = img.rows;
    int src_w = img.cols;
    float ratio_h = (float)inpHeight / (float)src_h;
    float ratio_w = (float)inpWidth / (float)src_w;
    ratio = ratio_h < ratio_w ? ratio_h : ratio_w;
    int newh = int(src_h*ratio);
    int neww = int(src_w*ratio);
   
   if(src_h!=inpHeight || src_w!=inpWidth)
   {
        resize(img,img,Size(neww,newh),INTER_LINEAR);

   }
    padh = inpHeight - newh;
    padw = inpWidth - neww;
    //cout << "padh " << padh << endl;
    int top = padh / 2;
    int bottom = padh - top;
    int left = padw / 2;
    int right = padw - left;
    copyMakeBorder(img,img,top,bottom,left,right,BORDER_CONSTANT, Scalar(114,114,114));
   
    
}


void YOLOv5::nms(std::vector<BoxInfo>& input_boxes)
{
	
	sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; }); // 降序排列
	std::vector<bool> remove_flags(input_boxes.size(),false);
	auto iou = [](const BoxInfo& box1,const BoxInfo& box2)
	{
		float xx1 = max(box1.x1, box2.x1);
		float yy1 = max(box1.y1, box2.y1);
		float xx2 = min(box1.x2, box2.x2);
		float yy2 = min(box1.y2, box2.y2);
		// 交集
		float w = max(0.0f, xx2 - xx1 );
		float h = max(0.0f, yy2 - yy1 );
		float inter_area = w * h;
		// 并集
		float union_area = max(0.0f,box1.x2-box1.x1) * max(0.0f,box1.y2-box1.y1)
						   + max(0.0f,box2.x2-box2.x1) * max(0.0f,box2.y2-box2.y1) - inter_area;
		return inter_area / union_area;
	};
	for (int i = 0; i < input_boxes.size(); ++i)
	{
		if(remove_flags[i]) continue;
		for (int j = i + 1; j < input_boxes.size(); ++j)
		{
			if(remove_flags[j]) continue;
			if(input_boxes[i].label == input_boxes[j].label && iou(input_boxes[i],input_boxes[j])>=this->nmsThreshold)
            
			{
				remove_flags[j] = true;
			}
		}
	}
	int idx_t = 0;
    // remove_if()函数 remove_if(beg, end, op) //移除区间[beg,end)中每一个“令判断式:op(elem)获得true”的元素
	auto it = remove_if(input_boxes.begin(),input_boxes.end(),[&idx_t,&remove_flags](const BoxInfo&){return remove_flags[idx_t++]; });
	input_boxes.erase(it, input_boxes.end());
	
}

// void YOLOv5::pre(Mat& img)
// {
//     this->resize_image(img);
//     imwrite("../fill.jpg",img);
//     cvtColor(img, img, cv::COLOR_BGR2RGB);   // 由BGR转成RGB
//     int H = img.rows;
//     int W =img.cols;
//     int C = img.channels();
//     float* ptr = (float*)m_ArrayHostMemory[0];
//     uchar * data = img.data;
//     // cout << "h " << H << endl;
//     // cout << "w " << W << endl;
//     // cout << "c " << C << endl;
//     for(int c = 0;c < C;++c)
//     {
//         for(int h = 0;h < H;++h)
//         {
//             for(int w = 0;w<W;++w)
//             {
//                 int newidx = c * H * W + h * W + w;
                
//                 int oldidx = h * W * C + w * C + c;
//                 ptr[newidx] = (float)data[oldidx] /255.f ;
//             }
//         }
//     }
    
// }

void YOLOv5::pre(Mat& img)
{
    this->resize_image(img);
    imwrite("../fill.jpg",img);
    vector<Mat> N_img = {img};
    Mat dstimg = cv::dnn::blobFromImages(N_img,1.0 / 255.0, cv::Size(inpWidth, inpHeight), cv::Scalar(0, 0, 0), true, false); //默认FP32
    memcpy(m_ArrayHostMemory[0],dstimg.data,3*inpHeight*inpWidth*sizeof(float));
    
   
}

void YOLOv5::pro()
{
    float* pdata = (float*)m_ArrayHostMemory[m_iOutputIndex];

	
	for(int i = 0; i < BoxNums; ++i) // 遍历所有的num_pre_boxes
	{
		int index = i * (ClassNums + 5);      // prob[b*num_pred_boxes*(classes+5)]  
		float obj_conf = pdata[index + 4];  // 置信度分数
		if (obj_conf > this->objThreshold)  // 大于阈值
		{
			float* max_class_pos = std::max_element(pdata + index + 5, pdata + index + 5 + ClassNums);   //
			float confidence = (*max_class_pos) * obj_conf;   // 最大的类别分数*置信度
			if (confidence > this->confThreshold) // 再次筛选
			{ 
				//const int class_idx = classIdPoint.x;
				float cx = pdata[index] ;  //x
				float cy = pdata[index+1];  //y
				float w = pdata[index+2];  //w
				float h = pdata[index+3];  //h

				float xmin = (cx - 0.5 * w - padw/2) / ratio;
				float ymin = (cy - 0.5 * h - padh/2) / ratio;
				float xmax = (cx + 0.5 * w - padw/2) / ratio;
				float ymax = (cy + 0.5 * h - padh/2) / ratio;

				generate_boxes.push_back(BoxInfo{xmin, ymin, xmax, ymax, confidence, (int)(max_class_pos-(pdata + index + 5))});
				//generate_boxes.emplace_back(BoxInfo{xmin, ymin, xmax, ymax, confidence, (int)(max_class_pos-(pdata + index + 5))});
			}
		}
	}
    
	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	std::cout << "nms之前框的数量 "<<generate_boxes.size() << std::endl;
	nms(generate_boxes);
	std::cout << "nms之后框的数量 "<<generate_boxes.size() << std::endl;
	//end = clock();
	//std::cout<<"postprocess time: "<<((double)end-strat)/CLOCKS_PER_SEC*1000<<"ms"<<std::endl;
	
}

void YOLOv5::detect(Mat img)
{
	pre(img);
   // cout  << "预处理成功" << endl;
    auto start_time = std::chrono::system_clock::now();
	auto ret = cudaMemcpyAsync(m_ArrayDevMemory[m_iInputIndex], m_ArrayHostMemory[m_iInputIndex], m_ArraySize[m_iInputIndex], cudaMemcpyHostToDevice, m_CudaStream); 
	//cout  << "从主机到gpu拷贝成功" << endl;
	auto ret1 = m_CudaContext->enqueueV2(m_ArrayDevMemory, m_CudaStream, nullptr);    // TensorRT 执行通常是异步的，因此将内核排入 CUDA 流：
	//auto ret1 = m_CudaContext->enqueue(1,m_ArrayDevMemory, m_CudaStream, nullptr);
    //cout << "推理成功" << endl;
	ret = cudaMemcpyAsync(m_ArrayHostMemory[m_iOutputIndex], m_ArrayDevMemory[m_iOutputIndex], m_ArraySize[m_iOutputIndex], cudaMemcpyDeviceToHost, m_CudaStream); //输出传回给CPU，数据从显存到内存
	ret = cudaStreamSynchronize(m_CudaStream);
    auto end_time = std::chrono::system_clock::now();
    cout << "infer time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms" << endl;
	pro();  
    
   
}


 void YOLOv5::drow(Mat& frame)
 {
    vector<Scalar> colors;
    for(int i = 0;i<ClassNums;++i)
    {
        int b = rand()%256;
        int g = rand()%256;
        int r = rand()%256;
        colors.push_back(Scalar(b,g,r));
    }

    for (size_t i = 0; i < generate_boxes.size(); ++i)
	{
		int xmin = int(generate_boxes[i].x1);
		int ymin = int(generate_boxes[i].y1);
		
		rectangle(frame, Point(xmin, ymin), Point(int(generate_boxes[i].x2), int(generate_boxes[i].y2)), colors[generate_boxes[i].label], 2);
		std::string score = format("%.2f", generate_boxes[i].score);
		#if defined(FACE)
		std::string label = std::string("face") + ":" + score;
		#else
		std::string label = this->classes[generate_boxes[i].label] + ":" + score;
		#endif
		putText(frame, label, Point(xmin, ymin - 5), FONT_HERSHEY_SIMPLEX, 0.75, colors[generate_boxes[i].label], 2);
	}
 }
