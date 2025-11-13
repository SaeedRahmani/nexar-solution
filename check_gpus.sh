#!/bin/bash
# Quick GPU checker - finds best GPU nodes

echo "=========================================="
echo "Checking GPU specifications on all nodes"
echo "=========================================="
echo ""

for node in gpu001 gpu002 gpu003 gpu004 gpu005 gpu006 gpu007 gpu008 gpu009 gpu010 gpu011 gpu012 gpu013; do
    echo "Node: $node"
    srun -w $node --gres=gpu:1 -p gpuloq -t 0:01:00 nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>/dev/null | head -1 || echo "  Not available"
    echo ""
done

echo "=========================================="
echo "Summary from scontrol:"
echo "=========================================="
for node in gpu001 gpu002 gpu003 gpu010 gpu011 gpu012 gpu013; do
    echo "=== $node ==="
    scontrol show node $node | grep -E "Gres=|RealMemory=" | head -2
done
