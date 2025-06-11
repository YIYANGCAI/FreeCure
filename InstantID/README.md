# Example of Integrating FreeCure into InstantID
This example demonstrate an example of how to integrate FreeCure in InstantID to improve its prompt-following performance. 

## Environment Preparation
Please following InstantID's original environment [setup](https://github.com/instantX-research/InstantID) to create the initial environment

Add following packages to enable Segment-Anything function.
```shell
pip install inference[yolo-world]==0.9.13
pip install onnxsim==0.4.35
pip install git+https://github.com/facebookresearch/segment-anything.git # sam
pip install timm # required
```

## Run the code
Follow the InstantID style prompt formula to set your prompt
```shell
python infer.py \
    --id-image ./examples/musk_resize.jpeg \
    --prompt "a man with blue eyes and blonde curly hair"
```
