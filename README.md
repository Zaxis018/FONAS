# FONAS: FPGA-based Optimization for Neural Architecture Search

## Summary
This project focuses on searching for efficient deep neural architectures (FPGANets) tailored for image classification tasks while adhering to constraints such as arithmetic intensity and latency for FPGA deployment. Our proposed FPGANets outperform existing networks in terms of both latency and accuracy on the ImageNet-1k dataset.

## Project Work and Methodologies
- **Compressing EfficientNet-V2**: Implementation of channel pruning techniques to reduce the model size and enhance inference speed for FPGA deployment.
- **Leveraging NAS techniques**: Automation of task-specific neural network creation to target latency requirements on FPGA platforms.
- **Hardware Optimization**: Co-designing models and hardware for improved efficiency and performance.

### Hardware NAS Focus
- **Optimization Goals**: Minimizing latency, maximizing accuracy, and efficient resource utilization on FPGA platforms.
- **Architecture Sampling**: Generating diverse architectures meeting hardware constraints like latency and resource usage.
- **Evaluation Metrics**: Assessing performance based on accuracy, inference speed, and resource utilization for optimal architecture selection.

## Key Results and Findings
- **Compressed EfficientNet-V2**: Achieved an 88% reduction in channel count resulting in a model that is 14 times smaller, 2.5 times faster in inference speed, and has significantly fewer parameters and MAC operations.
- **Latency Dataset**: Constructed a latency dataset for Ultra96v2 FPGA boards.
- **Latency-aware Networks**: Discovered through an evolutionary search process, networks optimized for latency.
- **Improved Performance**: The searched architectures (FPGANets) demonstrate superior performance in terms of the trade-off between latency and accuracy compared to existing architectures.
- ![res](https://github.com/FPGA-Vision/FONAS/assets/50907565/e37a749d-6905-4a5b-b943-37ee1592b7f8)

## Future Directions
- **Integration Challenges**: Seamless implementation of optimized architectures on FPGA platforms.
- **Validation Process**: Verification of the effectiveness of optimized architectures on Ultra96-v2 FPGA boards.
- **Framework Refinement**: Continuously enhancing the HW-NAS pipeline for improved efficiency and real-time performance.

This project aims to significantly contribute to the field of image classification by demonstrating the benefits of HW-NAS and FPGA-based acceleration in achieving superior efficiency and real-time performance metrics.
