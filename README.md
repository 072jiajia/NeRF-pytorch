# NeRF-pytorch

Here's two branches for implementation of NeRF<br>
[main](https://github.com/072jiajia/NeRF-pytorch) is the basic implementation (easy to read).<br>
[gradient-accumulation](https://github.com/072jiajia/NeRF-pytorch/tree/gradient-accumulation) uses gradient accumulation for larger image size and more sample point for each pixel


## Installation
All requirements should be detailed in requirements.txt. Using virtual environment is recommended.
```
conda create --name nerf python=3.8
source activate nerf
pip install -r requirements.txt
```

## Run
```
python main.py
```

