#!/bin/bash

# Path to your Python script
PYTHON_SCRIPT="cnn_float32_inference_full.py"

# Total start time
TOTAL_START=$(date +%s)

for i in {1..10}
do
    echo "Run #$i"
    START=$(date +%s)

    python3 "$PYTHON_SCRIPT"

    END=$(date +%s)
    RUNTIME=$((END-START))
    echo "Run #$i took $RUNTIME seconds"
    echo "---------------------------"
done

# Total end time
TOTAL_END=$(date +%s)
TOTAL_RUNTIME=$((TOTAL_END-TOTAL_START))
echo "All runs completed in $TOTAL_RUNTIME seconds"
