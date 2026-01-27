PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
nvidia-smi
CUDA_VISIBLE_DEVICES=0 python train_spcl.py --config="/configs/spcl/scd3.yaml" --exp='SCD/spcl_3'