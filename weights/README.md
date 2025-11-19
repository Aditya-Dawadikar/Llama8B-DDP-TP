This directory is intended for storing adapter checkpoint files (e.g. LoRA or
AdaLoRA weights) that have been trained via the scripts in this repository.
During training, the scripts save one subdirectory per epoch in the `--output_dir` and
write the adapter weights and tokenizer there.  Those checkpoints can be
moved or symlinked into this `weights/` directory if desired.  A small
placeholder file has been added to ensure the directory exists in version
control.