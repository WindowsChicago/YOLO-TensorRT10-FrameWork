#include "yolov10.h"

YOLOv10::YOLOv10(std::unique_ptr<TrtEngine> trt_engine) : YOLO(std::move(trt_engine)), conf_treshold_{0.2} {
    std::vector<size_t> input_dim = trt_engine_->GetInputDim();
    assert(input_dim.size() == 4); // support only dim [batch_size, channels, height, width]
    assert(input_dim[0] == 1); // support only batch size of 1
    assert(input_dim[1] == 3); // support only 3 channels

    std::vector<size_t> output_dim = trt_engine_->GetOutputDim();
    assert(output_dim.size() == 3); // support only dim [batch_size, anchors, xyhw + conf + label]
    assert(output_dim[0] = 1); // support only batch_size of 1
    assert(output_dim[2] == 6); // support only xyhw + conf + label
    assert(input_dim[2] == input_dim[3]); // support only square images
    
    input_size_  = input_dim[2];
    n_anchors_ = output_dim[1];
}

cv::Mat YOLOv10::PreprocessImage(const cv::Mat& original_img) {
    
    // Convert BGR to RGB
    cv::Mat rgb_img;
    cv::cvtColor(original_img, rgb_img, cv::COLOR_BGR2RGB);

    // Resize the image
    cv::Mat resized_img;
    cv::resize(rgb_img, resized_img, cv::Size(input_size_, input_size_));

    // Normalize the image
    resized_img.convertTo(resized_img, CV_32FC3, 1.0 / 255.0);

    return cv::dnn::blobFromImage(resized_img);
}

std::vector<cv::Rect> YOLOv10::RescaleBoxes(const std::vector<cv::Rect>& boxes, const cv::Size& original_size, int input_size) {
    std::vector<cv::Rect> rescaled_boxes;

    // Scale factors
    float scale_x = static_cast<float>(original_size.width) / input_size;
    float scale_y = static_cast<float>(original_size.height) / input_size;

    // Rescale each box
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

void YOLOv10::DrawBBoxes(const cv::Mat& image, std::vector<cv::Rect> bboxes, std::vector<int> class_ids, std::vector<float> confidences) {

    assert(bboxes.size() == class_ids.size());
    assert(bboxes.size() == confidences.size());

    for(size_t i=0; i < bboxes.size(); ++i) {
        std::stringstream stream;
        stream << std::fixed << std::setprecision(2) << confidences[i];
        std::string conf_str = stream.str();
        
        cv::rectangle(image, bboxes[i], kClassColors[class_ids[i]], 3);
        cv::putText(image, kClassNames[class_ids[i]] + " " + conf_str, cv::Point(bboxes[i].x, bboxes[i].y-10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 255, 255), 2);
    }
}


cv::Mat YOLOv10::PostprocessImage(cv::Mat& output, const cv::Mat& original_img, const cv::Size& original_size) {

    std::vector<int> class_ids;
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;

    for (size_t i=0; i<output.rows; ++i) {

        float conf = output.at<float>(i, 4);
        if(conf < conf_treshold_) continue;

        int x1 = static_cast<int>(output.at<float>(i, 0));
        int y1 = static_cast<int>(output.at<float>(i, 1));
        int x2 = static_cast<int>(output.at<float>(i, 2));
        int y2 = static_cast<int>(output.at<float>(i, 3));
        int class_id = static_cast<int>(output.at<float>(i, 5));

        boxes.push_back(cv::Rect{cv::Point{x1,y1}, cv::Point{x2, y2}});
        class_ids.push_back(class_id);
        confidences.push_back(conf);
    }

    boxes = RescaleBoxes(boxes, original_size, input_size_);

    cv::Mat output_img{original_img};

    DrawBBoxes(output_img, boxes, class_ids, confidences);

    return output_img;
}

cv::Mat YOLOv10::Detect(cv::Mat& input_img) {

    cv::Size original_size = input_img.size();
    cv::Mat preprocessed_img = PreprocessImage(input_img);

    float *input_tensor, *output_tensor;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&input_tensor), sizeof(float) * 3 * input_size_ * input_size_));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&output_tensor), sizeof(float) * n_anchors_ * 6));

    CUDA_CHECK(cudaMemcpy(input_tensor, preprocessed_img.ptr<float>(0), sizeof(float) * 3 * input_size_ * input_size_, cudaMemcpyHostToDevice));

    trt_engine_->Inference(input_tensor, output_tensor);

    cv::Mat host_ouput_tensor{n_anchors_, 6, CV_32FC2};
    CUDA_CHECK(cudaMemcpy(host_ouput_tensor.ptr<float>(0), output_tensor, sizeof(float) * n_anchors_ * 6, cudaMemcpyDeviceToHost));

    cv::Mat postprocessed_image = PostprocessImage(host_ouput_tensor, input_img, original_size);

    CUDA_CHECK(cudaFree(input_tensor));
    CUDA_CHECK(cudaFree(output_tensor));

    return postprocessed_image;
}