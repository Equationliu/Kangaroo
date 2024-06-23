import os, torch, shutil, glob, time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_home", type=str, default="/home/ma-user/work/Data/")
parser.add_argument("--start_epoch", type=int, default=0)
parser.add_argument("--end_epoch", type=int, default=20)
parser.add_argument("--bs", type=int, default=4)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--exit_layer", type=int, default=2)
parser.add_argument("--debug", action='store_true')
args = parser.parse_args()
print(args)

work_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(work_dir)

def data_moxing(src, dst):
    shutil.copytree(src, dst, dirs_exist_ok=True)

LOCAL_TRAIN_DATA = args.data_home
LOCAL_CKPT_DATA = "/cache/CKPT/EarlyExitAdapter/"

MASTER_ADDR = "172.16.2.146"
MASTER_PORT = 62275
rank = "0"
nnodes = 1
processes = int(nnodes) * torch.cuda.device_count()

for epoch in range(args.start_epoch, args.end_epoch):
    print("start epoch: ", epoch)
    dst_dir = LOCAL_TRAIN_DATA

    command = f"MKL_SERVICE_FORCE_INTEL=1 MKL_THREADING_LAYER=GNU accelerate launch --multi_gpu \
        --num_machines {nnodes} --num_processes {processes} --main_process_ip {MASTER_ADDR} \
            --main_process_port {MASTER_PORT+epoch} --machine_rank {rank} --mixed_precision=fp16 train.py \
                --start {epoch} --tmpdir {dst_dir} --cpdir {LOCAL_CKPT_DATA} \
                    --basepath /cache/CKPT/vicuna-7b-v1.3/ --configpath ./data/vicuna_7B_config.json \
                        --bs {args.bs} --lr {args.lr} --exit_layer {args.exit_layer}"

    print(command)
    os.system(command)