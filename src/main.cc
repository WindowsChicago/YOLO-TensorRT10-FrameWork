#include <iostream>

#include "trt_engine.h"
#include "yolov8.h"
#include "yolov10.h"
#include "yolov5a.h"

// 在main.cc中添加
enum class Models {
    YOLOv8,
    YOLOv10,
    YOLOv5A  
};


struct Options {
    bool build = false;
    bool timing = false;
    std::string video_path;
    std::string image_path;
    std::string load_path;
    std::string precision;
    Models model;
};


bool parse_arguments(int argc, char* argv[], Options &opts) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--model") {
            if (i + 1 < argc) {
                std::string model_name = argv[++i];
                if(model_name == "yolov8") {
                    opts.model = Models::YOLOv8;
                } else if (model_name == "yolov10") {
                    opts.model = Models::YOLOv10;
                } else if (model_name == "yolov5a") {  // 新增
                    opts.model = Models::YOLOv5A;
                } else {
                    std::cerr << "Unknown model\n";
                    return false;
                }
            } else {
                std::cerr << "Error: --model requires a model name\n";
                return false;
            }
        } 
        else if (arg == "--video") {
            if (i + 1 < argc) {
                opts.video_path = argv[++i];
            } else {
                std::cerr << "Error: --video requires a path\n";
                return false;
            }
        } 
        else if (arg == "--image") {
            if (i + 1 < argc) {
                opts.image_path = argv[++i];
            } else {
                std::cerr << "Error: --image requires a path\n";
                return false;
            }
        } 
        else if (arg == "--build") {
            if (i + 1 < argc) {
                opts.precision = argv[++i];
                opts.build = true;
            } else {
                std::cerr << "Error: --build requires precision\n";
                return false;
            }
        }
        else if (arg == "--timing") {
            opts.timing = true;
        } 
        else {
            std::cerr << "Error: Unknown option " << arg << '\n';
            return false;
        }
    }
    return true;
}




int main(int argc, char* argv[]) {

    Options opts;
    if(!parse_arguments(argc, argv, opts)) {
        return 1;
    }
    
    Logger logger;
    auto trt_engine = std::make_unique<TrtEngine>(logger);
    
    if(opts.build) {

        TrtPrecision precision;
        if(opts.precision == "fp32") {
            precision = TrtPrecision::kFP32;
        } else if (opts.precision == "fp16"){
            precision = TrtPrecision::kFP16;
        } else if (opts.precision == "int8") {
            precision = TrtPrecision::kINT8;
        } else {
            std::cerr << "Precision not supported, only fp32, fp18 and int8 are supported" << std::endl;
        }
        
        if(opts.model == Models::YOLOv8) {
            trt_engine->BuildEngine("../models/yolov8n.onnx", "../models/yolov8n.engine", precision);
        } else if (opts.model == Models::YOLOv10) {
            trt_engine->BuildEngine("../models/yolov10n.onnx", "../models/yolov10n.engine", precision);
        } else if (opts.model == Models::YOLOv5A) {  // 新增
            trt_engine->BuildEngine("../models/yolov5a.onnx", "../models/yolov5a.engine", precision);
        }

    }


    if(!opts.image_path.empty()) {

        std::unique_ptr<YOLO> yolo;

        // load tensorrt engine previously built
        if(opts.model == Models::YOLOv8) {
            trt_engine->LoadEngine("../models/yolov8n.engine");
            yolo = std::make_unique<YOLOv8>(std::move(trt_engine));
        } else if (opts.model == Models::YOLOv10) {
            trt_engine->LoadEngine("../models/yolov10n.engine");
            yolo = std::make_unique<YOLOv10>(std::move(trt_engine));
        } else if (opts.model == Models::YOLOv5A) {  // 新增
            trt_engine->LoadEngine("../models/yolov5a.engine");
            yolo = std::make_unique<YOLOv5A>(std::move(trt_engine));
        }

        cv::Mat img = cv::imread(opts.image_path, cv::IMREAD_COLOR);

        auto begin = std::chrono::steady_clock::now();
        cv::Mat detection = yolo->Detect(img);
        auto end = std::chrono::steady_clock::now();
        
        if(opts.timing)
            std::cout << "Inference time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

        cv::imwrite("detection_output.jpg", detection);
    }


    else if(!opts.video_path.empty()){

        std::unique_ptr<YOLO> yolo;

        // load tensorrt engine previously built
        if(opts.model == Models::YOLOv8) {
            trt_engine->LoadEngine("../models/yolov8n.engine");
            yolo = std::make_unique<YOLOv8>(std::move(trt_engine));
        } else if (opts.model == Models::YOLOv10) {
            trt_engine->LoadEngine("../models/yolov10n.engine");
            yolo = std::make_unique<YOLOv10>(std::move(trt_engine));
        } else if (opts.model == Models::YOLOv5A) {  // 新增
            trt_engine->LoadEngine("../models/yolov5a.engine");
            yolo = std::make_unique<YOLOv5A>(std::move(trt_engine));
        }

        cv::VideoCapture cap(opts.video_path);

        if (!cap.isOpened()) {
            std::cout << "Cannot open the video file. :(\n";
            return -1;
        }

        int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        cv::VideoWriter video("detection_output.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, cv::Size(frame_width,frame_height));    
        cv::Mat frame;

        while(true){
            cap >> frame;

            if (frame.empty())
                break;

            auto begin = std::chrono::steady_clock::now();
            cv::Mat detection = yolo->Detect(frame);
            auto end = std::chrono::steady_clock::now();
        
            if(opts.timing)
                std::cout << "Inference time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
            
            video.write(detection);
        }
        
        cap.release();
        video.release();
    }
}
