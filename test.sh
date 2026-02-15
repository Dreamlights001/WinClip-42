#!/bin/bash

# WinClip Zero-shot Testing Script
# Random seed: 42

# Define datasets and their classes
declare -A DATASET_CLASSES

# MVTec
DATASET_CLASSES["mvtec"]="carpet grid leather tile wood bottle cable capsule hazelnut metal_nut pill screw toothbrush transistor zipper"

# MVTec2
DATASET_CLASSES["mvtec2"]="can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts"

# VisA
DATASET_CLASSES["visa"]="candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum"

# MPDD
DATASET_CLASSES["mpdd"]="bracket_black bracket_brown bracket_white connector metal_plate tubes"

# DAGM
DATASET_CLASSES["dagm"]="Class1 Class2 Class3 Class4 Class5 Class6 Class7 Class8 Class9 Class10"

# BTAD
DATASET_CLASSES["btad"]="01 02 03"

# DTD
DATASET_CLASSES["dtd"]="Blotchy_099 Fibrous_183 Marbled_078 Matted_069 Mesh_114 Perforated_037 Stratified_154 Woven_001 Woven_068 Woven_104 Woven_125 Woven_127"

# Medical datasets
DATASET_CLASSES["br35h"]="br35h"
DATASET_CLASSES["brainmri"]="brainmri"
DATASET_CLASSES["brain_tumor_mri"]="brain_tumor_mri"

# ISIC
DATASET_CLASSES["isic"]="isic2016"

# Colonoscopy
DATASET_CLASSES["clinicdb"]="clinicdb"
DATASET_CLASSES["colondb"]="colondb"

# Common parameters
K_SHOT=0
BATCH_SIZE=128
IMG_RESIZE=240
IMG_CROPSIZE=240
RESOLUTION=400
GPU_ID=0

# Output directory
ROOT_DIR="./result_winclip"

# Function to run test for a specific dataset and class
run_test() {
    local dataset=$1
    local class_name=$2
    
    echo "=========================================="
    echo "Testing: $dataset - $class_name"
    echo "=========================================="
    
    python eval_WinCLIP.py \
        --dataset $dataset \
        --class-name $class_name \
        --k-shot $K_SHOT \
        --batch-size $BATCH_SIZE \
        --img-resize $IMG_RESIZE \
        --img-cropsize $IMG_CROPSIZE \
        --resolution $RESOLUTION \
        --gpu-id $GPU_ID \
        --root-dir $ROOT_DIR \
        --vis False
    
    echo ""
}

# Main execution
echo "=================================================="
echo "WinClip Zero-shot Testing"
echo "Random Seed: 42"
echo "K-shot: $K_SHOT"
echo "=================================================="
echo ""

# Parse command line arguments
SELECTED_DATASETS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SELECTED_DATASETS+=("$1")
                shift
            done
            ;;
        --gpu)
            shift
            GPU_ID=$1
            shift
            ;;
        --k-shot)
            shift
            K_SHOT=$1
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --datasets DATASET1 DATASET2 ...  Specify datasets to test (default: all)"
            echo "  --gpu GPU_ID                      GPU ID to use (default: 0)"
            echo "  --k-shot K                        Number of shots (default: 0 for zero-shot)"
            echo "  --help                            Show this help message"
            echo ""
            echo "Available datasets:"
            for dataset in "${!DATASET_CLASSES[@]}"; do
                echo "  - $dataset"
            done
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# If no datasets specified, run all
if [ ${#SELECTED_DATASETS[@]} -eq 0 ]; then
    echo "No datasets specified. Running all datasets..."
    SELECTED_DATASETS=("${!DATASET_CLASSES[@]}")
fi

echo "Datasets to test: ${SELECTED_DATASETS[*]}"
echo ""

# Run tests
total_start_time=$(date +%s)

for dataset in "${SELECTED_DATASETS[@]}"; do
    if [ -z "${DATASET_CLASSES[$dataset]}" ]; then
        echo "Warning: Unknown dataset '$dataset', skipping..."
        continue
    fi
    
    classes=${DATASET_CLASSES[$dataset]}
    
    for class_name in $classes; do
        run_test "$dataset" "$class_name"
    done
done

total_end_time=$(date +%s)
total_duration=$((total_end_time - total_start_time))

echo "=================================================="
echo "All tests completed!"
echo "Total time: $((total_duration / 3600))h $(((total_duration % 3600) / 60))m $((total_duration % 60))s"
echo "=================================================="
