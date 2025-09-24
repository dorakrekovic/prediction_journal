#!/bin/bash

#rnn_inference_simple.py
#rnn_inference_full.py
#cnn_float32_inference_full.py
#cnn_float32_inference_simple.py
PYTHON_SCRIPT="cnn_float32_inference_simple.py"

# Total start time
TOTAL_START_NS=$(date +%s%N)
TOTAL_START_HUMAN=$(date +%Y-%m-%dT%H:%M:%S.%3N)
echo "=== All runs started at $TOTAL_START_HUMAN ==="

for i in {1..10}
do
    echo "Run #$i"
    START_NS=$(date +%s%N)
    START_HUMAN=$(date +%Y-%m-%dT%H:%M:%S.%3N)
    echo "  Start: $START_HUMAN"

    python3 "$PYTHON_SCRIPT"

    END_NS=$(date +%s%N)
    END_HUMAN=$(date +%Y-%m-%dT%H:%M:%S.%3N)
    RUNTIME_MS=$(( (END_NS - START_NS)/1000000 ))
    echo "  End  : $END_HUMAN"
    echo "  Run #$i took ${RUNTIME_MS} ms"
    echo "---------------------------"
done

# Total end time
TOTAL_END_NS=$(date +%s%N)
TOTAL_END_HUMAN=$(date +%Y-%m-%dT%H:%M:%S.%3N)
TOTAL_RUNTIME_MS=$(( (TOTAL_END_NS - TOTAL_START_NS)/1000000 ))
echo "=== All runs finished at $TOTAL_END_HUMAN ==="
echo "All runs completed in ${TOTAL_RUNTIME_MS} ms"
