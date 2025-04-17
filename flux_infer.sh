export FLUX_DEV_FILL="/home/work/fuping-workspace/model_zoo/FLUX.1-Fill-dev/flux1-fill-dev.safetensors"
export AE="/home/work/fuping-workspace/model_zoo/FLUX.1-Fill-dev/ae.safetensors"

start_time=$(date +%s)  
echo "Starting at $(date +"%H:%M:%S")"  

input_image="./assets/cup.png"
input_mask="./assets/cup_mask.png"

# python -m src.flux.cli_fill \
#   --img_cond_path $input_image \
#   --img_mask_path $input_mask

python3 flux_fill_diffusers.py

# python3 flux_diffusers.py

end_time=$(date +%s)  
elapsed=$((end_time - start_time))  
echo "Completed at $(date +"%H:%M:%S")"  
echo "Total execution time: $elapsed seconds ($(($elapsed / 60)) minutes and $(($elapsed % 60)) seconds)"  