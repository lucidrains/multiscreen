# /// script
# dependencies = [
#   "torch",
#   "accelerate",
#   "tqdm",
#   "numpy",
#   "multiscreen",
#   "fire",
# ]
# ///

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import fire
import gzip
import random
import tqdm
import numpy as np

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator

from multiscreen.multiscreen import MultiScreen

# helpers

def exists(v):
    return v is not None

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

# training function

def train(
    num_batches = int(1e5),
    batch_size = 4,
    grad_accum_every = 4,
    learning_rate = 1e-4,
    validate_every = 100,
    seq_len = 128,
    dim_keys = 32,
    dim_values = 64,
    heads = 16,
    depth = 12,
    dim = 512,
    prime_length = 32,
    generate_length = 128,
    generate_every = 500,
    competitive = False,
    use_sugar = True
):
    # accelerators

    accelerator = Accelerator()

    print = accelerator.print

    # the multiscreen char language model

    model = MultiScreen(
        num_tokens = 256,
        dim = dim,
        heads = heads,
        depth = depth,
        dim_keys = dim_keys,
        dim_values = dim_values,
        competitive = competitive,
        use_sugar = use_sugar
    )

    # prepare enwik8 data

    with gzip.open("./data/enwik8.gz") as file:
        data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
        np_train, np_valid = np.split(data, [int(90e6)])
        data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

    class TextSamplerDataset(Dataset):
        def __init__(self, data, seq_len):
            super().__init__()
            self.data = data
            self.seq_len = seq_len

        def __len__(self):
            return self.data.size(0) // self.seq_len

        def __getitem__(self, index):
            rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
            full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
            return full_seq

    train_dataset = TextSamplerDataset(data_train, seq_len)
    val_dataset = TextSamplerDataset(data_val, seq_len)
    train_loader = DataLoader(train_dataset, batch_size = batch_size)
    val_loader = DataLoader(val_dataset, batch_size = batch_size)

    # optimizer

    optim = Adam(model.parameters(), lr = learning_rate)

    model, optim, train_loader, val_loader = accelerator.prepare(
        model, optim, train_loader, val_loader
    )

    train_loader = cycle(train_loader)
    val_loader = cycle(val_loader)

    # training

    for i in tqdm.tqdm(range(num_batches), mininterval = 10.0, desc = "training"):
        model.train()

        for _ in range(grad_accum_every):
            data = next(train_loader)

            loss = model(data, return_loss = True)

            accelerator.backward(loss / grad_accum_every)

        print(f"training loss: {loss.item():.3f}")

        accelerator.clip_grad_norm_(model.parameters(), 0.5)

        optim.step()
        optim.zero_grad()

        if i % validate_every == 0:
            model.eval()
            with torch.no_grad():
                valid_data = next(val_loader)

                loss = model(valid_data, return_loss = True)
                print(f"validation loss: {loss.item():.3f}")

        if i % generate_every == 0:
            model.eval()

            inp = random.choice(val_dataset)[:prime_length]
            inp = inp.to(accelerator.device)

            prime = decode_tokens(inp)
            print(f"\nINPUT: {prime}")

            prompt = inp[None, ...]

            unwrapped_model = accelerator.unwrap_model(model)
            sampled = unwrapped_model.generate(prompt, generate_length)

            base_decode_output = decode_tokens(sampled[0])

            print(f"\nOUTPUT: {base_decode_output}\n")

if __name__ == '__main__':
    fire.Fire(train)
