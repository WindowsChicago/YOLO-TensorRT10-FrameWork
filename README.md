# yolo-tensorrt
Optimization of YOLOv8 and YOLOv10 inference with TensorRT.

### Dependencies
- TensorRT >= 10.2.0
- CUDA >= 11.8
- Opencv >= 4.8.0

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

