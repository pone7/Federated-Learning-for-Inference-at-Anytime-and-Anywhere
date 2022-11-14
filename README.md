# distributedFL

## Setup with Poetry
#### install poetry 
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

#### install the packages in the poetry container
poetry install

#### activate the install poetry enviroment
poetry shell

#### pip install requirements
pip install -r requirements.txt


## Setup with Conda
```bash
# create conda env
conda create -n affl python=3.8 -y

# activate your environment
source activate affl

# install pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -y

# install other requirements
pip install -r requirements.txt
pip install ml_collections einops timm tqdm scikit-image wand opencv-python==4.6.0.66 tensorboard thop
```

## run the script to launch the experiment program
sh run_fl_simulation.sh