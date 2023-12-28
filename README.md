# LargeScale 

Set `DATA_PATH`, `MULTITASK_DATA_PATH`, `CHECKPOINT_PATH` in `configs/glm-130b/glm-130b.sh` and `HOST_FILE_PATH` in `scripts/submit_gpu.sh`. Run the following scripts to reproduce GLM-130B's  training.

```
bash scripts/submit_gpu.sh configs/glm-130b/glm-130b.sh
```

At least 24 DGX-A100 (40G) is needed to lanuch training. A more detailed README will be released soon.

