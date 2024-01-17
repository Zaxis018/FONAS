#!/bin/bash
# Cleaning up
sudo rm -r ./compiled_all_blocks/*
counter=0

declare -a models=("Conv_in-1x1x1152_out-1536_k-1_acth_swish.pth" 
                   "Conv_in-224x224x3_out-24_k-3_actrelu.pth"
                   "Linear_in-1536_out-1000.pth"
                   "Conv_in-7x7x192_out-1152_k-1_acth_swish.pth")

declare -a input_shapes=("1152,1,1"
                         "3,224,224"
                         "1536"  # Updated based on the actual output shape of the Linear layer
                         "192,7,7")
i
declare -a deploy_names=()

for ((i=0; i<${#models[@]}; i++))
do
    model=${models[i]}
    input_shape=${input_shapes[i]}
    
    file="./float/$model"
    if [ -f "$file" ]; then
        name=$(echo "$file" | sed 's|./float/||')

        echo -----------------------Starting Qt $name -----------------------------

        # To quantize a certain block of the model
        sudo /opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/python code/quantization.py --device "cpu" --quant_mode calib --subset_len 100 --model_name "$name"  --input_shape "[$input_shape]"
        echo Compilation
        sudo /opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/python ./code/quantization.py --quant_mode test --subset_len 1 --batch_size=1 --deploy --model_name "$name" --input_shape "[$input_shape]"

        # Construct the output compiled block name with proper shape included
        if [ "$i" -eq 2 ]; then
            # For Linear layer, no need to reverse the ordering of input shape
            deploy_name=$(echo "$name" | sed -E 's/\.pth$/_shape'"[$input_shape]_output100x2x2"'/')
        else
            # For Convolutional layers, reverse the ordering of input shape
            deploy_name=$(echo "$name" | sed -E 's/\.pth$/_shape'"[$(echo $input_shape | tr ',' 'x')]"'/')
        fi

        deploy_names+=("$deploy_name")

        # To deploy the model and get the quantized model in (onnx, pth, xmodel format)
        sudo -E PATH=$PATH:/opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin /opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/vai_c_xir -x ./quantize_result/*.xmodel -a arch.json -o ./compiled -n "$deploy_name"

        sudo mkdir ./compiled_all_blocks/"$deploy_name"
        sudo mv ./compiled/* ./compiled_all_blocks/"$deploy_name"
        # sudo rm -r ./compiled/*
        sudo rm -r ./quantized/*
        sudo rm -r ./quantize_result/*

        ((counter++))
        echo -e "\n\n"
        echo -e "---------------------------Done Block $counter-------------------------\n\n"
    fi
done

# Print the deploy names for reference
echo "Deploy Names:"
printf "%s\n" "${deploy_names[@]}"

