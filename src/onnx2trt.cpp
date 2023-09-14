#include<NvInfer.h>
#include<NvOnnxParser.h>
#include<iostream>
#include<cstring>
#include<string>
#include<cassert>
#include<fstream>
#include"calibrator.h"
#include"logger.h"
#define USE_INT8
#define FACE
#define INPUT_W 640
#define INPUT_H 640
#define BATCHSIZE 1
using namespace nvinfer1;


void onnx2trt(const std::string& onnxfile,const std::string& trtfile)
{
    TRT_Logger gLogger;   // 日志
    //根据tensorrt pipeline 构建网络
    IBuilder* builder = createInferBuilder(gLogger);    // 
    //builder->setMaxBatchSize(1);  //对显示批处理的网络无影响
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);  // 显式批处理
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);                      // 定义模型
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);              // 使用nvonnxparser 定义一个可用的onnx解析器
    parser->parseFromFile(onnxfile.c_str(), static_cast<int>(ILogger::Severity::kWARNING));   // 解析onnx
	// 使用builder对象构建engine
    IBuilderConfig* config = builder->createBuilderConfig();   // 
	// 特别重要的属性是最大工作空间大小
    //config->setMaxWorkspaceSize(1U << 30);    TensorRT 8.3后被遗弃              
	config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 2 * 1U << 30);   //分配最大工作空间2GB
#if defined(USE_INT8)
	std::cout << "支持INT8: " << (builder->platformHasFastInt8() ? "true":"false") << std::endl; 
	assert(builder->platformHasFastInt8());
	
  	config->setFlag(BuilderFlag::kINT8);
    #if defined(COCO)
    Calibrator* calibrator = new  Calibrator(1,INPUT_W,INPUT_H,"../coco_calib/","../coco_int8calib.table","images",true);
    #elif defined(FACE)
    Calibrator* calibrator = new  Calibrator(1,INPUT_W,INPUT_H,"../widerface_calib/","../widerface_int8calib.table","images",true);
    #endif
    config->setInt8Calibrator(calibrator);
	
#endif
    ICudaEngine* CudaEngine = builder->buildEngineWithConfig(*network, *config);    // 来创建一个 ICudaEngine 类型的对象，在构建引擎时，TensorRT会复制权重

    std::string strTrtName = trtfile;
   
    IHostMemory *gieModelStream = CudaEngine->serialize();    // 将引擎序列化
    //std::string serialize_str;     // 
    std::ofstream serialize_output_stream(strTrtName.c_str(),std::ios::out|std::ios::binary);
    assert(serialize_output_stream.is_open());
    char* buffer = new char[gieModelStream->size()];
   // serialize_str.resize(gieModelStream->size()); 
	// memcpy内存拷贝函数 ，从源内存地址的起始位置开始拷贝若干个字节到目标内存地址中
    //memcpy((void*)serialize_str.data(),gieModelStream->data(),gieModelStream->size()); 
    memcpy(buffer,gieModelStream->data(),gieModelStream->size());
    serialize_output_stream.write(buffer,gieModelStream->size());
    //serialize_output_stream.open(strTrtName.c_str());  
    //serialize_output_stream << serialize_str;     // 将引擎序列化数据转储到文件中
    serialize_output_stream.close();   
    delete[] buffer;
    buffer = nullptr;
    //m_CudaContext = m_CudaEngine->createExecutionContext();    //执行上下文用于执行推理
	// 使用一次，销毁parser，network, builder, and config 
    parser->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
    CudaEngine->destroy();
}

int main(int arg,char* argv[])
{
    assert(arg == 3);
    std::string onnx_path = argv[1];
    std::string trt_path = argv[2];
    onnx2trt(onnx_path,trt_path);
    std::cout << "engine序列化成功"<<std::endl;
    return 0;
}