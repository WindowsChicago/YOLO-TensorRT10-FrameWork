# YOLO-TensorRT10-Detect-Framework
#### by ZYL

### TensorRT10 YOLO C++ 高性能目标检测推理框架

### 支持推理：

- YOLOv5/X/v8/v10/v11的标准/魔改版目标检测模型
- 无需pt文件，只要是能够通过trtexec转换为engine的onnx即可
- 低代码量，容易扩展模型和标签格式支持

### 支持设备：

- 带NVIDIA独显（MAXWELL架构之后的）的x86架构的PC
- Jetson Orin 系列（不支持XAVIER及以前的Jetson）

### 依赖：
- TensorRT >= 10.2.0
- CUDA >= 11.8
- OpenCV (with CUDA)>= 4.8.0

### Build
(Optional) build the docker image to avoid managing dependencies

```bash
docker build -t yolofast . # might takes a long time bc of the opencv build (~1h on my modest machine)
docker run --gpus all -it --name yolofast -v $(pwd):/workspace/yolofast yolofast
```

```bash
mkdir build && cd build
cmake .. 
make -j4
```

### Usage
```bash
Usage: yolofast [options]
Options:
  --model <name>          Specify YOLOv8n or YOLOv10n model.
  --video <path>          Run inference on video and save it as 'detection_output.avi'.
  --image <path>          Run inference on image and save it as 'detection_output.jpg'.
  --build <precision>     Specify precision optimization (e.g., fp32, fp12 or int8).
  --timing                Enable timing information.

Example:
  ./yolofast --model yolov8 --build fp16 --video ../samples/video.mp4 --timing
```

### Results
![image](.assets/image1_fp32.jpg)

The model had also no problem to run on a video :

![Alt Text](.assets/output_video.gif)
M1:添加对yolo5的支持（深大的装甲板识别模型特化支持）
M2:修复新增的yolov5模型推理坐标的放缩错误
M3:修改CmakeLists.txt，修改调用TensorRT的方式
