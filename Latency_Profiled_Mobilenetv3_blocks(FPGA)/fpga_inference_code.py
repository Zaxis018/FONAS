from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import pathlib
import xir
import os
import math
import threading
import time
import sys

current_directory = os.getcwd()

def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph
            is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]

def runEfficientnetv2(dpu_runner):
    """get dimension of input and output tensor"""
    inputTensor = dpu_runner.get_input_tensors()
    outputTensor = dpu_runner.get_output_tensors()

    input_ndim = tuple(inputTensor[0].dims)
    print("input dimension:",input_ndim)

    pre_output_size = int(outputTensor[0].get_data_size() / input_ndim[0])
    print("output dimension:",pre_output_size)
    output_ndim = tuple(outputTensor[0].dims)

    input_data = [np.empty(input_ndim, dtype=np.int8, order="C")]
    print("Shape of input data:", len(input_data))
   
    image_run = input_data[0]
    print("image_run:", image_run)
    print("image run shape:", image_run.shape)
    #image_run[0,...] = image.reshape(inputTensor[0].dims[1]
    # image_run[0,...] = image.reshape(inputTensor[0].dims[1],inputTensor[0].dims[2],inputTensor[0].dims[3])
    image_run[0,...] = np.random.randint(low=-128, high=127, size=input_ndim[0], dtype=np.int8)
    print("image_run shape after:", image_run.shape)
    print("image_run data:", image_run)
    
    output_data = [np.empty(output_ndim, dtype=np.int8, order="C")]
    print("Output data before:", output_data)

    print("Execute async")
    job_id = dpu_runner.execute_async(input_data, output_data)
    # print("job id:", job_id)
    print("Output data after:", output_data)
    dpu_runner.wait(job_id)
    print("Execution complete")

def runModel(runner: "Runner", cnt):
    """get tensor"""
    inputTensors = runner.get_input_tensors()
    outputTensors = runner.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    print("input_ndim[0]:{}".format(input_ndim))
    pre_output_size = int(outputTensors[0].get_data_size() / input_ndim[0])

    output_ndim = tuple(outputTensors[0].dims)

    count = 0
    run_time = []
    while count < cnt:
        runSize = input_ndim[0]
        """prepare batch input/output """
        inputData = [np.empty(input_ndim, dtype=np.int8, order="C")]
        outputData = [np.empty(output_ndim, dtype=np.int8, order="C")]

        """init input image to input buffer """
        for j in range(runSize):
            imageRun = inputData[0]
            
            imageRun[j, ...] = np.random.randint(low=-128, high=127, size=input_ndim[0], dtype=np.int8)
        """run with batch """

        time_start = time.time()
        job_id = runner.execute_async(inputData, outputData)
        runner.wait(job_id)
        time_end = time.time()

        inference_time = time_end - time_start
        run_time.append(inference_time)
        count = count + runSize
    print("Total runs:{}".format(count))
    print("Average run time:{} ms".format((np.sum(run_time)/len(run_time))*1000))

def main(argv):
    g = xir.Graph.deserialize(argv[1])
    subgraphs = get_child_subgraph_dpu(g)
    assert len(subgraphs) == 1

    dpu_runners = vart.Runner.create_runner(subgraphs[0], "run")
    print("DPU runner created")

    #runEfficientnetv2(dpu_runners)
    count=500

    #time_start = time.time()
    #runEfficientnetv2(dpu_runners)
    runModel(dpu_runners,count)
    del dpu_runners
    #time_end = time.time()
    #total_time = time_end - time_start
    #print("Avg inference time:",total_time/count)
    #print("FPS:{}".format(count/total_time))

if __name__=="__main__":
    if len(sys.argv) !=2:
        print("usage: python3 fpga_inference_code.py <xmodel location>")
    else:
        main(sys.argv)
