import argparse
parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--basepath', type=str, default='')
parser.add_argument('--configpath', type=str, default="config.json")
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--bs', type=int, default=4)
parser.add_argument("--exit_layer", type=str, default='2')
parser.add_argument('--gradient-accumulation-steps', type=int, default=8)
parser.add_argument('--tmpdir', type=str, default='0')
parser.add_argument('--outdir', type=str, default='0')
parser.add_argument('--cpdir', type=str, default='0')
parser.add_argument('--start', type=int, default=0)
args = parser.parse_args()

train_config = {
    "lr": args.lr,
    "bs": args.bs,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "datapath": f"{args.tmpdir}",
    "is_warmup": True,
    "num_epochs": 20,
    "num_warmup_steps": 2000,
    "total_steps": 800000,
    "num_workers": 8,
    "act": "No",
    "residual": "true,norm",
    "max_len": 2048,
    # During training, truncating the training sequences means that the larger the setting, the more training data is used, and the better the effect, but it also consumes more VRAM.
    "config_path": args.configpath,
    "b1": 0.9,
    "b2": 0.95,
    "grad_clip": 0.5,
    "save_freq": 2
}

import json
from safetensors import safe_open
import os
import torch

torch.backends.cuda.matmul.allow_tf32 = True
from accelerate import Accelerator
from accelerate.utils import set_seed

set_seed(0)
accelerator = Accelerator(mixed_precision='fp16',
                          gradient_accumulation_steps=train_config["gradient_accumulation_steps"])

from kangaroo.adapter import AdapterModel

from typing import Any, Dict, List

from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from transformers import get_linear_schedule_with_warmup, AutoConfig, get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

if accelerator.is_main_process:
    writer = SummaryWriter(os.path.join(args.cpdir, f"tensorboard"))
    writer.add_text('config', json.dumps(train_config))

baseconfig = AutoConfig.from_pretrained(args.basepath)

head = torch.nn.Linear(baseconfig.hidden_size, baseconfig.vocab_size, bias=False)

try:
    with open(os.path.join(args.basepath, "model.safetensors.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
    with safe_open(os.path.join(args.basepath, head_path),
                   framework="pt",
                   device="cpu") as f:
        tensor_slice = f.get_slice("lm_head.weight")
        vocab_size, hidden_dim = tensor_slice.get_shape()
        tensor = tensor_slice[:, :hidden_dim].float()
except:
    with open(os.path.join(args.basepath, "pytorch_model.bin.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
    weights = torch.load(os.path.join(args.basepath, head_path))
    tensor = weights["lm_head.weight"].float()

head.weight.data = tensor
head.eval()

for param in head.parameters():
    param.requires_grad = False

def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath


class CustomDataset(Dataset):
    def __init__(self, datapath, transform=None):
        self.data = datapath
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.load(self.data[index])
        new_data = {}
        hidden_state = data['hidden_state'][:train_config["max_len"]][None, :]
        input_ids = data['input_ids'][:train_config["max_len"]][None, :]
        loss_mask = data["loss_mask"][:train_config["max_len"]][None, :]
        
        exit_layer = args.exit_layer
        hidden_state_layer = data["hidden_state_layer{}".format(exit_layer)][:train_config["max_len"]][None, :]
        
        length = hidden_state.shape[1]
        attention_mask = [1] * length
        loss_mask = loss_mask[0].tolist()
        loss_mask[-1] = 0

        input_ids_target = input_ids[:, 1:]
        zeropadding = torch.tensor([[0]])
        input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

        target = hidden_state[:, 1:, :]
        zeropadding = torch.zeros(1, 1, target.shape[2])
        target = torch.cat((target, zeropadding), dim=1)
        
        hidden_state_layer = hidden_state_layer[:, 1:, :]
        zeropadding = torch.zeros(1, 1, target.shape[2])
        hidden_state_layer = torch.cat((hidden_state_layer, zeropadding), dim=1)
        
        loss_mask[-1] = 0
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state_big"] = hidden_state
        new_data["input_ids"] = input_ids_target
        
        new_data["hidden_state_early"] = hidden_state_layer
      
        if self.transform:
            new_data = self.transform(new_data)

        return new_data


class DataCollatorWithPadding:

    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['hidden_state_big'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_hidden_states_early = torch.cat([self.paddingtensor(item['hidden_state_early'], max_length) for item in features])
        batch_hidden_states = torch.cat([self.paddingtensor(item['hidden_state_big'], max_length) for item in features])
        batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])
        batch_loss_mask = torch.tensor(
            [item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
        batch_attention_mask = torch.tensor(
            [item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])
        # batch_loss_mask = torch.ones_like(batch_loss_mask)
        # batch_attention_mask=torch.ones_like(batch_attention_mask)
        batch = {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "hidden_states_early": batch_hidden_states_early,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch


def top_accuracy(output, target, topk=(1,)):
    # output.shape (bs, num_classes), target.shape (bs, )
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res
    

datapath = list_files(train_config["datapath"])
traindatapath = datapath[:int(len(datapath) * 0.95)]
testdatapath = datapath[int(len(datapath) * 0.95):]
# print('td',train_config["datapath"])
# print(datapath)
# exit()
traindataset = CustomDataset(traindatapath, transform=None)
testdataset = CustomDataset(testdatapath)
train_loader = DataLoader(traindataset, batch_size=train_config["bs"], shuffle=True,
                          collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"],
                          pin_memory=True)
test_loader = DataLoader(testdataset, batch_size=train_config["bs"], shuffle=False,
                         collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"], pin_memory=True)

if accelerator.is_main_process:
    if not os.path.exists(args.cpdir):
        os.makedirs(args.cpdir)

config = AutoConfig.from_pretrained(train_config["config_path"])
model = AdapterModel(config)

optimizer = optim.AdamW(model.parameters(), lr=train_config["lr"], betas=(train_config["b1"], train_config["b2"]))

num_epochs = train_config["num_epochs"]
num_warmup_steps = train_config["num_warmup_steps"]
total_steps = train_config["total_steps"]
is_warmup = train_config["is_warmup"]

if is_warmup:
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_steps)

    model, head, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, head, optimizer, train_loader, test_loader, scheduler
    )
else:
    model, head, optimizer, train_loader, test_loader = accelerator.prepare(
        model, head, optimizer, train_loader, test_loader
    )
if args.start > 0:
    accelerator.load_state(f"{args.cpdir}/state_{args.start-1}/")

log_steps = 20

for epoch in range(args.start, args.start + 1):
    print("start epoch: ", epoch)
    top_3acc = [0 for _ in range(3)]
    correct = 0
    total = 0
    epoch_loss = 0
    num_batches = 0
    model.train()
    for batch_idx, data in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
      
        from torch.autograd import Variable
        data["hidden_states_early"] = Variable(data["hidden_states_early"], requires_grad=True)
        predict = model(inputs_embeds=data["hidden_states_early"], attention_mask=data["attention_mask"])
        
        with torch.no_grad():
            target_head = head(data["target"]) # predict the feature after the RMS-Norm
            target_p = nn.Softmax(dim=2)(target_head)
            target_p = target_p.detach()

        out_head = head(predict)
        prob_exit = F.softmax(out_head, dim = 2)
        prob_last = F.softmax(target_head, dim = 2)
        prob_acc = torch.min(prob_last, prob_exit).sum(dim = 2)
        
        out_logp = nn.LogSoftmax(dim=2)(out_head)
        loss_mask = data["loss_mask"][:, :, None]
        plogp = target_p * out_logp
        loss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / loss_mask.sum()
        prob_acc = torch.sum(data["loss_mask"] * prob_acc) / data["loss_mask"].sum() 
      
        if accelerator.is_main_process and batch_idx % log_steps == 0:
            print(f"\nStep: {batch_idx}\tLR: {optimizer.optimizer.param_groups[0]['lr']}\tAccept: {prob_acc.item()}\tLoss: {loss.item()}\n")
        
        accelerator.backward(loss)
        accelerator.clip_grad_value_(model.parameters(), train_config["grad_clip"])
        optimizer.step()

        if loss != loss and accelerator.is_main_process:
            print(f"nan, Epoch {epoch}, batch id {batch_idx}")
            with open('nan.txt', 'w') as f:
                f.write(f"nan, Epoch {epoch}, batch id {batch_idx}")
                torch.save(data, 'nandata.ckpt')
            exit()

        if is_warmup:
            scheduler.step()

        with torch.no_grad():
            _, predicted = torch.max(out_head, 2)
            _, target = torch.max(target_head, 2)
            ct = loss_mask.sum().item()
            cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
            out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
            target = target.view(-1)[loss_mask.view(-1) == 1]
            topkacc = top_accuracy(out_head, target, (1, 2, 3))
            for top_i in range(len(topkacc)):
                top_3acc[top_i] += topkacc[top_i]
            total += ct
            correct += cc
        
        if accelerator.is_main_process and ct != 0:
            writer.add_scalar(f"train/lr", optimizer.optimizer.param_groups[0]["lr"], batch_idx+len(train_loader)*epoch)
            writer.add_scalar(f"train/loss", loss.item(), batch_idx+len(train_loader)*epoch)
            writer.add_scalar(f"train/prob_accept", prob_acc.item(), batch_idx+len(train_loader)*epoch)
            writer.add_scalar(f"train/accuracy", cc / ct, batch_idx+len(train_loader)*epoch)
            for id, i in enumerate(top_3acc):
                writer.add_scalar(f"Top_K/top_{id + 1}_acc", topkacc[id].item() / ct, batch_idx+len(train_loader)*epoch)

        epoch_loss += loss.item()
        num_batches += 1

    correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
    correct, total = accelerator.gather_for_metrics((correct, total))
    correct, total = correct.sum().item(), total.sum().item()
    epoch_loss /= num_batches
    top_3acc = accelerator.gather_for_metrics(top_3acc)
    if accelerator.is_local_main_process:
        for id, i in enumerate(top_3acc):
            writer.add_scalar(f"epoch/top_{id + 1}_acc", i.sum().item() / total, batch_idx+len(train_loader)*epoch)
    
    if accelerator.is_local_main_process:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        print('Train Accuracy: {:.2f}%'.format(100 * correct / total))

    accelerator.save_state(output_dir=f"{args.cpdir}/state_{epoch}")
    
