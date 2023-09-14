// yolov5使用tensorrt进行部署的头文件

#pragma once
#define USE_INT8
//#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui.hpp>
#include <NvInfer.h>       // nvidia加载模型进行推理的插件
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include<opencv2/opencv.hpp>
#include<chrono>
using namespace cv;
using namespace nvinfer1;
using namespace std;

// 自定义配置结构
struct Configuration
{ 
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	float objThreshold;  //Object Confidence threshold
	std::string modelpath;
   
};

// 定义BoxInfo结构类型
typedef struct BoxInfo
{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	int label;
} BoxInfo;


class YOLOv5
{
public:
	YOLOv5(Configuration config);
	~YOLOv5();
	void UnInit();
	void detect(Mat img);
	//void loadOnnx(const std::string &strName);
    int loadTrt(const std::string &strName); 
	void pre(Mat& img);
    void pro();
	void nms(std::vector<BoxInfo>& input_boxes);  
	//cv::Mat resize_image(cv::Mat srcimg, int *newh, int *neww, int *top, int *left);
    void resize_image(Mat& img);
    void drow(Mat& frame);
	float confThreshold;
	float nmsThreshold;
	float objThreshold;
	int inpWidth;
	int inpHeight;
    int padw;
    int padh;
    float ratio;
    Mat srcimg;
    vector<BoxInfo> generate_boxes;
	std::string classes[80] = {"person", "bicycle", "car", "motorbike", "aeroplane", "bus",
							"train", "truck", "boat", "traffic light", "fire hydrant",
							"stop sign", "parking meter", "bench", "bird", "cat", "dog",
							"horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
							"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
							"skis", "snowboard", "sports ball", "kite", "baseball bat",
							"baseball glove", "skateboard", "surfboard", "tennis racket",
							"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
							"banana", "apple", "sandwich", "orange", "broccoli", "carrot",
							"hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant",
							"bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
							"remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
							"sink", "refrigerator", "book", "clock", "vase", "scissors",
							"teddy bear", "hair drier", "toothbrush"};
	

	//const bool keep_ratio = true;


	nvinfer1::ICudaEngine *m_CudaEngine; 
	nvinfer1::IRuntime *m_CudaRuntime;
    nvinfer1::IExecutionContext *m_CudaContext;
	cudaStream_t m_CudaStream;    // //初始化流,CUDA流的类型为cudaStream_t 
private:
	int m_iInputIndex;
    int m_iOutputIndex; 
	int ClassNums;
    int BoxNums;
	//cv::Size m_InputSize;
	void* m_ArrayDevMemory[2]{0};
    void* m_ArrayHostMemory[2]{0}; 
	int m_ArraySize[2]{0};
	//std::vector<cv::Mat> m_InputWrappers{}; 
};




