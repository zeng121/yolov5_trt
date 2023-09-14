#include "calibrator.h"
#include"util.h"
#include<opencv2/opencv.hpp>
#include<opencv2/dnn/dnn.hpp>
#include<cuda_runtime_api.h>
#include<fstream>
#include<iostream>
#include<iterator>
#include<cstring>
/*Refer to https://github.com/wang-xinyu/tensorrtx*/
using namespace cv;

static Mat preprocess_int8(Mat img,int input_w,int input_h)
{
    int src_w = img.cols;
    int src_h = img.rows;
    float ratio_w = input_w / (src_w*1.0);
    float ratio_h = input_h / (src_h*1.0);
    float ratio = ratio_w < ratio_h ? ratio_w:ratio_h;
    int new_w = int(src_w*ratio);
    int new_h = int(src_h*ratio);
    //Mat dstimg(img);
    if(new_w!=input_w || new_h!=input_h)
    {
        resize(img,img,Size(new_w,new_h),INTER_LINEAR);
    }
    int top,bottom,left,right;
    int padw = input_w - new_w;
    int padh = input_h - new_h;
    top = padh / 2; bottom = padh - top;
    left = padw / 2; right = padw - left;
    copyMakeBorder(img,img,top,bottom,left,right,BORDER_CONSTANT, Scalar(114,114,114));
    return img;

}

Calibrator::Calibrator(int batchsize,int input_w,int input_h,const char* img_dir,const char*calib_table_name,const char* input_blob_name,bool read_cache)
                    : batchsize_(batchsize),
                      input_w_(input_w),
                      input_h_(input_h),
                      img_idx_(0),
                      img_dir_(img_dir),
                      calib_table_name_(calib_table_name),
                      input_blob_name_(input_blob_name),
                      read_cache_(read_cache){
                            input_count_ = input_h*input_w*3*batchsize;
                            cudaMalloc(&device_input_,input_count_*sizeof(float));
                            read_files_in_dir(img_dir, img_files_);
}



Calibrator:: ~Calibrator()
{
    cudaFree(device_input_);
    device_input_ = nullptr;
}

int Calibrator:: getBatchSize() const noexcept
{
    return batchsize_;
}

bool Calibrator:: getBatch(void* bindings[],const char* names[],int nbBindings) noexcept
{
    if(img_idx_ + batchsize_ > img_files_.size())
    {
        return false;
    }
    std::vector<cv::Mat> input_imgs_;
    for (int i = img_idx_; i < img_idx_ + batchsize_; i++) {
    std::cout << img_files_[i] << "  " << i << std::endl;
    cv::Mat temp = cv::imread(img_dir_ + img_files_[i]);
    if (temp.empty()) {
      std::cerr << "Fatal error: image cannot open!" << std::endl;
      return false;
    }
    cv::Mat pr_img = preprocess_int8(temp, input_w_, input_h_);
    input_imgs_.push_back(pr_img);
  }
  img_idx_ += batchsize_;
  cv::Mat blob = cv::dnn::blobFromImages(input_imgs_, 1.0 / 255.0, cv::Size(input_w_, input_h_), cv::Scalar(0, 0, 0), true, false);
  cudaMemcpy(device_input_,blob.data,input_count_*sizeof(float),cudaMemcpyHostToDevice);
  bindings[0] = device_input_;
  return true;
}

// const void* Calibrator:: readCalibrationCache(size_t& length) noexcept
// {
//     std::cout << "reading calib cache: " << calib_table_name_ << std::endl;
//     calib_cache_.clear();
//     std::ifstream input(calib_table_name_, std::ios::binary);
//     input >> std::noskipws;
//      if (read_cache_ && input.good()) {
//     std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(calib_cache_));
//     }
//     length = calib_cache_.size();
//     return length ? calib_cache_.data() : nullptr;
// }

const void* Calibrator:: readCalibrationCache(size_t& length) noexcept
{
    std::ifstream fin(calib_table_name_.c_str(),std::ios::in|std::ios::binary);
    if(!fin.is_open())
    {
        fin.close();
        return nullptr;
    }
    _cache_ = malloc(length*sizeof(char));
    fin.read((char*)_cache_,length);
    return length ? _cache_:nullptr;
}

void Calibrator::writeCalibrationCache(const void* cache,size_t length) noexcept
{
    std::cout << "writing calib cache: " << calib_table_name_ << " size: " << length << std::endl;
    std::ofstream output(calib_table_name_, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
}