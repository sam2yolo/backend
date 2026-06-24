conda create -n sam python=3.12
conda activate sam

pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# if flash attenstion supports
pip install einops ninja && pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128
pip install git+https://github.com/ronghanghu/cc_torch.git

git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .

pip install -e ".[train,dev]"

gdown https://drive.google.com/file/d/1U_SBWxdyRFx-519v_UQZh48cm4y4qLVm/view?usp=drive_link
mkdir hf
unzip sam3.1.zip -d hf

python sam/remap.py sam3.1_multiplex.pt sam3.1_multiplex_remapped.pt

