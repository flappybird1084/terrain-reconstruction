Terrain Reconstruction 

```bash
pyenv shell 3.11
python3 -m venv env
source env/bin/activate

pip install -r requirements.txt
mkdir models
cd models
mkdir terrain
cd ../../

python3 train_heightmap.py
python3 train_terrain.py
```

CUDA/MPS advised.
