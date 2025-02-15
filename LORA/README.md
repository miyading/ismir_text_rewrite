# Fine-Tuning with LoRA or QLoRA

This is an example of using MLX to fine-tune an LLM with low rank adaptation
(LoRA) for a target task.

## Contents
* [Fine-tune](#Fine-tune)
* [Evaluate](#Evaluate)
* [Generate](#Generate)
* [Fuse and Upload](#Fuse-and-Upload)


## Fine-tune

To fine-tune a model use:

```
mlx_lm.lora \
   --model <path_to_model: models/Llama-3.1-8B-Instruct> \
   --train \
   --data <path_to_data: data> \
   --fine-tune-type lora \
   --batch-size 4 \
   --iters 1500 \
   --num-layers 8

```



## Evaluate

To compute test set perplexity use:

```
mlx_lm.lora \
--model <path_to_model: models/Llama-3.1-8B-Instruct> \
--adapter-path <path_to_adapters: models/2.0/adapters> \
--data <path_to_data: data> \
--test
```

And use `LORA_Evaluate.py` for score matrix.
## Generate

For generation use `LORA_Generate.py`


## Fuse and Upload

You can generate a fused model with the low-rank adapters included using the
`fuse.py` script. 

To generate the fused model run:

```
python fuse.py

mlx_lm.fuse \
--model <path_to_model: models/Llama-3.1-8B-Instruct> \
--adapter-path <path_to_adapters: models/2.0/adapters> \
--save-path <save_path_to_fused_model: NEP2.0> \ \
--de-quantize

```

This will by default load the base model from `mlx_model/`, the adapters from
`models/2.0/adapters`,  and save the fused model in the path `NEP2.0/`. All
of these are configurable. You can see the list of options with:

```
python fuse.py --help
```


