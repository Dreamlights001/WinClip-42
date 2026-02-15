#!/bin/bash

# WinClip Zero-shot Testing Script
# Random seed: 42

# Common parameters
K_SHOT=0
BATCH_SIZE=128
IMG_RESIZE=240
IMG_CROPSIZE=240
RESOLUTION=400
GPU_ID=0
ROOT_DIR="./result_winclip"

# Function to run test for a specific dataset and class
run_test() {
    local dataset=$1
    local class_name=$2
    
    echo "=========================================="
    echo "Testing: $dataset - $class_name"
    echo "=========================================="
    
    python eval_WinCLIP.py \
        --dataset "$dataset" \
        --class-name "$class_name" \
        --k-shot "$K_SHOT" \
        --batch-size "$BATCH_SIZE" \
        --img-resize "$IMG_RESIZE" \
        --img-cropsize "$IMG_CROPSIZE" \
        --resolution "$RESOLUTION" \
        --gpu-id "$GPU_ID" \
        --root-dir "$ROOT_DIR" \
        --vis False
    
    echo ""
}

# Function to run all classes for a dataset
run_dataset() {
    local dataset=$1
    shift
    local classes="$@"
    
    for class_name in $classes; do
        run_test "$dataset" "$class_name"
    done
}

# Main execution
echo "=================================================="
echo "WinClip Zero-shot Testing"
echo "Random Seed: 42"
echo "K-shot: $K_SHOT"
echo "=================================================="
echo ""

# Parse command line arguments
SELECTED_DATASETS=""

while [ $# -gt 0 ]; do
    case $1 in
        --datasets)
            shift
            while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do
                SELECTED_DATASETS="$SELECTED_DATASETS $1"
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
            echo "  - mvtec"
            echo "  - mvtec2"
            echo "  - visa"
            echo "  - mpdd"
            echo "  - dagm"
            echo "  - btad"
            echo "  - dtd"
            echo "  - br35h"
            echo "  - brainmri"
            echo "  - brain_tumor_mri"
            echo "  - isic"
            echo "  - clinicdb"
            echo "  - colondb"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# If no datasets specified, run all
if [ -z "$SELECTED_DATASETS" ]; then
    echo "No datasets specified. Running all datasets..."
    SELECTED_DATASETS="mvtec mvtec2 visa mpdd dagm btad dtd br35h brainmri brain_tumor_mri isic clinicdb colondb"
fi

echo "Datasets to test:$SELECTED_DATASETS"
echo ""

# Run tests
total_start_time=$(date +%s)

for dataset in $SELECTED_DATASETS; do
    case $dataset in
        mvtec)
            run_dataset mvtec carpet grid leather tile wood bottle cable capsule hazelnut metal_nut pill screw toothbrush transistor zipper
            ;;
        mvtec2)
            run_dataset mvtec2 can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts
            ;;
        visa)
            run_dataset visa candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum
            ;;
        mpdd)
            run_dataset mpdd bracket_black bracket_brown bracket_white connector metal_plate tubes
            ;;
        dagm)
            run_dataset dagm Class1 Class2 Class3 Class4 Class5 Class6 Class7 Class8 Class9 Class10
            ;;
        btad)
            run_dataset btad 01 02 03
            ;;
        dtd)
            run_dataset dtd Blotchy_099 Fibrous_183 Marbled_078 Matted_069 Mesh_114 Perforated_037 Stratified_154 Woven_001 Woven_068 Woven_104 Woven_125 Woven_127
            ;;
        br35h)
            run_dataset br35h br35h
            ;;
        brainmri)
            run_dataset brainmri brainmri
            ;;
        brain_tumor_mri)
            run_dataset brain_tumor_mri brain_tumor_mri
            ;;
        isic)
            run_dataset isic isic2016
            ;;
        clinicdb)
            run_dataset clinicdb clinicdb
            ;;
        colondb)
            run_dataset colondb colondb
            ;;
        *)
            echo "Warning: Unknown dataset '$dataset', skipping..."
            ;;
    esac
done

total_end_time=$(date +%s)
total_duration=$((total_end_time - total_start_time))

echo "=================================================="
echo "All tests completed!"
echo "Total time: $((total_duration / 3600))h $(((total_duration % 3600) / 60))m $((total_duration % 60))s"
echo "=================================================="
