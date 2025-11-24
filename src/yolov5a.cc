// yolov5a.cc
#include "yolov5a.h"
#include <cmath>
#include <iomanip>

YOLOv5A::YOLOv5A(std::unique_ptr<TrtEngine> trt_engine) 
    : YOLO(std::move(trt_engine)), conf_threshold_(0.65f), nms_threshold_(0.45f) {
    
    std::vector<size_t> input_dim = trt_engine_->GetInputDim();
    assert(input_dim.size() == 4); // [batch_size, channels, height, width]
    assert(input_dim[0] == 1); // batch size of 1
    assert(input_dim[1] == 3); // 3 channels
    assert(input_dim[2] == input_dim[3]); // square images
    
    input_size_ = input_dim[2];
}

const std::array<std::string, 4> YOLOv5A::kColorNames = {
    "blue", "red", "gray", "purple"
};

const std::array<cv::Scalar, 4> YOLOv5A::kColorColors = {
    cv::Scalar(0, 0, 255),   // red
    cv::Scalar(255, 0, 0),   // blue  
    cv::Scalar(128, 128, 128), // gray
    cv::Scalar(128, 0, 128)  // purple
};

const std::array<std::string, 9> YOLOv5A::kNumberNames = {
    "G", "1", "2", "3", "4", "5", "O", "Bs", "Bb"
};

double YOLOv5A::sigmoid(double x) {
    if (x > 0)
        return 1.0 / (1.0 + std::exp(-x));
    else
        return std::exp(x) / (1.0 + std::exp(x));
}

double YOLOv5A::calculateIOU(const cv::Rect& r1, const cv::Rect& r2) {
    float x_left = std::fmax(r1.x, r2.x);
    float y_top = std::fmax(r1.y, r2.y); 
    float x_right = std::fmin(r1.x + r1.width, r2.x + r2.width);
    float y_bottom = std::fmin(r1.y + r1.height, r2.y + r2.height);

    if (x_right < x_left || y_bottom < y_top) {
        return 0.0; 
    }

    double in_area = (x_right - x_left) * (y_bottom - y_top);
    double un_area = r1.area() + r2.area() - in_area; 

    return in_area / un_area;
}

cv::Mat YOLOv5A::PreprocessImage(const cv::Mat& original_img) {
    // Convert BGR to RGB (与OpenVINO一致)
    cv::Mat rgb_img;
    cv::cvtColor(original_img, rgb_img, cv::COLOR_BGR2RGB);

    // Resize the image
    cv::Mat resized_img;
    cv::resize(rgb_img, resized_img, cv::Size(input_size_, input_size_));

    // Normalize the image to [0,1] (与OpenVINO一致)
    resized_img.convertTo(resized_img, CV_32FC3, 1.0 / 255.0);

    return cv::dnn::blobFromImage(resized_img);
}

std::vector<cv::Rect> YOLOv5A::RescaleBoxes(const std::vector<cv::Rect>& boxes, 
                                          const cv::Size& original_size, int input_size) {
    std::vector<cv::Rect> rescaled_boxes;

    float scale_x = static_cast<float>(original_size.width) / input_size;
    float scale_y = static_cast<float>(original_size.height) / input_size;

    for (const auto& box : boxes) {
        cv::Rect rescaled_box;
        rescaled_box.x = box.x * scale_x;
        rescaled_box.y = box.y * scale_y;
        rescaled_box.width = box.width * scale_x;
        rescaled_box.height = box.height * scale_y;
        rescaled_boxes.push_back(rescaled_box);
    }

    return rescaled_boxes;
}

std::vector<std::vector<cv::Point2f>> YOLOv5A::RescaleLandmarks(
    const std::vector<std::vector<cv::Point2f>>& landmarks, 
    const cv::Size& original_size, int input_size) {
    
    std::vector<std::vector<cv::Point2f>> rescaled_landmarks;

    float scale_x = static_cast<float>(original_size.width) / input_size;
    float scale_y = static_cast<float>(original_size.height) / input_size;

    for (const auto& landmark : landmarks) {
        std::vector<cv::Point2f> rescaled_landmark;
        for (const auto& point : landmark) {
            rescaled_landmark.push_back(cv::Point2f(point.x * scale_x, point.y * scale_y));
        }
        rescaled_landmarks.push_back(rescaled_landmark);
    }

    return rescaled_landmarks;
}

void YOLOv5A::DrawBBoxes(const cv::Mat& image, std::vector<cv::Rect> bboxes, 
                        std::vector<int> class_ids, std::vector<float> confidences,
                        std::vector<std::vector<cv::Point2f>> landmarks,
                        std::vector<int> colors) {
    
    assert(bboxes.size() == class_ids.size());
    assert(bboxes.size() == confidences.size());
    assert(bboxes.size() == landmarks.size());
    assert(bboxes.size() == colors.size());

    for (size_t i = 0; i < bboxes.size(); ++i) {
        std::stringstream stream;
        stream << std::fixed << std::setprecision(2) << confidences[i];
        std::string conf_str = stream.str();
        
        // 绘制边界框（使用颜色类别的颜色）
        cv::rectangle(image, bboxes[i], kColorColors[colors[i]], 3);
        
        // 绘制关键点
        for (const auto& point : landmarks[i]) {
            cv::circle(image, point, 3, cv::Scalar(0, 255, 0), -1); // 绿色关键点
        }
        
        // 绘制标签（颜色 + 数字 + 置信度）
        std::string label = kColorNames[colors[i]] + " " + kNumberNames[class_ids[i]] + " " + conf_str;
        cv::putText(image, label, cv::Point(bboxes[i].x, bboxes[i].y - 10), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    }
}

cv::Mat YOLOv5A::PostprocessImage(cv::Mat& output, const cv::Mat& original_img, 
                                 const cv::Size& original_size) {
    
    std::vector<int> class_ids;
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<std::vector<cv::Point2f>> landmarks;
    std::vector<int> colors;

    // 输出格式：25200 x 22
    // [0-7]: 8个关键点坐标 (x1,y1,x2,y2,x3,y3,x4,y4)
    // [8]: 置信度
    // [9-12]: 颜色概率 (红,蓝,灰,紫)
    // [13-21]: 数字类别概率 (G,1,2,3,4,5,O,Bs,Bb)

    for (int i = 0; i < output.rows; ++i) {
        float confidence = output.at<float>(i, 8);
        confidence = sigmoid(confidence); // 对置信度使用sigmoid
        
        if (confidence < conf_threshold_) continue;

        // 提取关键点 (8个坐标值)
        std::vector<cv::Point2f> landmark_points;
        for (int j = 0; j < 8; j += 2) {
            float x = output.at<float>(i, j);
            float y = output.at<float>(i, j + 1);
            landmark_points.push_back(cv::Point2f(x, y));
        }

        // 提取颜色概率并找到最大值
        cv::Mat color_scores(1, 4, CV_32F, output.ptr<float>(i) + 9);
        cv::Point color_id;
        double color_score;
        cv::minMaxLoc(color_scores, NULL, &color_score, NULL, &color_id);

        // 提取数字类别概率并找到最大值
        cv::Mat number_scores(1, 9, CV_32F, output.ptr<float>(i) + 13);
        cv::Point number_id;
        double number_score;
        cv::minMaxLoc(number_scores, NULL, &number_score, NULL, &number_id);

        // 过滤不需要的颜色（根据OpenVINO代码逻辑）
        if (color_id.x == 2 || color_id.x == 3) { // 灰色或紫色
            continue;
        }

        // 从关键点计算边界框
        float min_x = landmark_points[0].x;
        float max_x = landmark_points[0].x;
        float min_y = landmark_points[0].y;
        float max_y = landmark_points[0].y;

        for (const auto& point : landmark_points) {
            if (point.x < min_x) min_x = point.x;
            if (point.x > max_x) max_x = point.x;
            if (point.y < min_y) min_y = point.y;
            if (point.y > max_y) max_y = point.y;
        }

        cv::Rect box(min_x, min_y, max_x - min_x, max_y - min_y);

        boxes.push_back(box);
        class_ids.push_back(number_id.x); // 使用数字类别
        confidences.push_back(confidence);
        landmarks.push_back(landmark_points);
        colors.push_back(color_id.x); // 使用颜色类别
    }

    // NMS处理
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, nms_threshold_, indices);

    std::vector<cv::Rect> filtered_boxes;
    std::vector<int> filtered_class_ids;
    std::vector<float> filtered_confidences;
    std::vector<std::vector<cv::Point2f>> filtered_landmarks;
    std::vector<int> filtered_colors;

    for (int idx : indices) {
        filtered_boxes.push_back(boxes[idx]);
        filtered_class_ids.push_back(class_ids[idx]);
        filtered_confidences.push_back(confidences[idx]);
        filtered_landmarks.push_back(landmarks[idx]);
        filtered_colors.push_back(colors[idx]);
    }

    // 缩放回原始尺寸
    filtered_boxes = RescaleBoxes(filtered_boxes, original_size, input_size_);
    filtered_landmarks = RescaleLandmarks(filtered_landmarks, original_size, input_size_);

    cv::Mat output_img = original_img.clone();
    DrawBBoxes(output_img, filtered_boxes, filtered_class_ids, filtered_confidences, 
               filtered_landmarks, filtered_colors);

    return output_img;
}

cv::Mat YOLOv5A::Detect(cv::Mat& input_img) {
    cv::Size original_size = input_img.size();
    cv::Mat preprocessed_img = PreprocessImage(input_img);

    std::vector<size_t> input_dim = trt_engine_->GetInputDim();
    std::vector<size_t> output_dim = trt_engine_->GetOutputDim();

    size_t input_count = 3 * input_size_ * input_size_;
    size_t output_count = output_dim[1] * output_dim[2]; // 25200 * 22

    float *input_tensor, *output_tensor;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&input_tensor), sizeof(float) * input_count));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&output_tensor), sizeof(float) * output_count));

    CUDA_CHECK(cudaMemcpy(input_tensor, preprocessed_img.ptr<float>(0), 
                         sizeof(float) * input_count, cudaMemcpyHostToDevice));

    trt_engine_->Inference(input_tensor, output_tensor);

    // 输出形状: [1, 25200, 22]
    cv::Mat host_output_tensor(output_dim[1], output_dim[2], CV_32F);
    CUDA_CHECK(cudaMemcpy(host_output_tensor.ptr<float>(0), output_tensor, 
                         sizeof(float) * output_count, cudaMemcpyDeviceToHost));

    cv::Mat postprocessed_image = PostprocessImage(host_output_tensor, input_img, original_size);

    CUDA_CHECK(cudaFree(input_tensor));
    CUDA_CHECK(cudaFree(output_tensor));

    return postprocessed_image;
}