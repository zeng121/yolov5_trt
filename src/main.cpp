#include"yolov5_trt.h"
#include<iostream>
#include<chrono>

int main(int arg,char* argv[])
{
    
    assert(arg == 3);
    string img_path = argv[2];
    string model_path = argv[1];
    
    
    Configuration yolo_nets = { 0.25, 0.45, 0.25,model_path};  // 注意路径
	YOLOv5 yolo_model(yolo_nets);
    Mat srcimg = imread(img_path.c_str());
    yolo_model.detect(srcimg);
    
    yolo_model.drow(srcimg);
    imwrite("../res.jpg",srcimg);
    printf("检测结果保存到%s\n","res.jpg");
    return 0;
}