conda create -n chattts-ollama python=3.9
source activate base
conda activate chattts-ollama
pip install -r requirements.txt

# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124