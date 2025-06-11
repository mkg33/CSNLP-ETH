
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda3


eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init

source ~/.bashrc


# once this runs and you have the .yml file, run this:

conda env create -f ~/csnlp_torch_env.yml

conda activate csnlp_torch
python3 - <<'EOF'
import torch, transformers, pandas as pd, spacy
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("pandas:", pd.__version__)
print("spacy:", spacy.__version__)
EOF
conda deactivate
