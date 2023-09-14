#pragma once

#include<vector>
#include<string>
#include<NvInfer.h>
using namespace nvinfer1;

class Calibrator: public IInt8EntropyCalibrator2
{
public:

    Calibrator(int batchsize,int input_w,int input_h,const char* img_dir,const char*calib_table_name,const char* input_blob_name,bool read_cache);
    virtual ~Calibrator();
    int getBatchSize() const noexcept override;
    bool getBatch(void* bindings[],const char* names[],int nbBindings) noexcept override;
    const void* readCalibrationCache(size_t& length) noexcept override;
    void writeCalibrationCache(const void* cache,size_t length) noexcept override;
private:
    int batchsize_;
    int input_w_;
    int input_h_;
    int img_idx_;
    std::string img_dir_;
    std::vector<std::string> img_files_;
    size_t input_count_;
    std::string calib_table_name_;
    const char* input_blob_name_;
    bool read_cache_;
    void* device_input_;
    
    std::vector<char> calib_cache_;  
    void* _cache_;   //自己添加
};