# YOLO Real-Time Edge Deployment & Runtime

A production-ready edge deployment container for high-throughput YOLO inference with OpenVINO optimization, multi-camera streaming, heatmaps, and speed tracking.

## Components
- **Docker Environment**: Containerized Python 3.10 runtime with OpenCV, OpenVINO, and Ultralytics dependencies.
- **OpenVINO Hardware Acceleration**: Native INT8/FP16 models for CPU and integrated GPU execution.
- **Multi-Stream Analytics**: Concurrent multi-camera feed ingestion, dynamic speed estimation, and activity heatmaps.

## Docker Build & Run
```bash
docker build -t yolo-edge-deployment .
docker run --rm -it yolo-edge-deployment
```
