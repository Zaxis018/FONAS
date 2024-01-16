This repository follows a straightforward file structure to organize compiled .xmodel files and related components:
1. Folders for Each Compiled .xmodel:
        -> Each model (e.g., ResidualBlock) has its own dedicated folder.
        -> xrt.run_summary: This file is accessible through Vitis Profiler and provides a summary of the execution.
        -> block.xmodel: The compiled model designed for FPGA inference.
        -> Other Generated Files: Various files generated during the profiling process.

2. Inference and shell script for batch profiling
