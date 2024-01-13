wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.10.0-1-Linux-x86_64.sh
bash Miniconda3-py310_23.10.0-1-Linux-x86_64.sh
source ~/.bashrc
conda create -n seldo python=3.10 py
conda activate seldo
conda install pytorch  pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install seldonian-experiments transformers
echo "Installed seldo conda environment successfully"
