#!/bin/bash

# Check if model_id parameter is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_id>"
    exit 1
fi
model_id=$1
main_dir="scripts/run"
# Loop through all subdirectories in the main directory
for sub_dir in "$main_dir"/*; do
    if [ -d "$sub_dir" ]; then
        sub_dir_name=$(basename "$sub_dir")
        sub_dir_name_clean=$(echo "$sub_dir_name" | sed 's/_scripts//')
        echo "Entering directory: $sub_dir"

        # Loop through all .sh files in the subdirectory
        for sh_file in "$sub_dir"/*.sh; do
            if [ -f "$sh_file" ]; then
                sh_file_name=$(basename "$sh_file".sh)
                # sh_file_name_clean=$(echo "$sh_file_name" | sed 's/^f[0-9]-//')
                echo "Running experiment in $sub_dir_name_clean catagory of $sh_file_name_clean subset...."
                bash "$sh_file" "$model_id" "$sub_dir_name_clean"
            fi
        done
    fi
done

echo "All scripts have been run"
