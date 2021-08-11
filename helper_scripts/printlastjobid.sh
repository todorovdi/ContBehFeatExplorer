squeue | grep todorov1 | tail -1 | awk '{print $1;}' | head -c 6
