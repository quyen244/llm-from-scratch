import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from src.model import Tokenizer , Model
from src.utils.config import Config
from pathlib import Path

class CharDataset(Dataset):
    """
    KHÔNG cắt trước hàng triệu substring ra RAM (sketch gốc làm vậy -> rất tốn bộ nhớ
    và mỗi item vẫn là string thô, chưa encode). Lưu 1 tensor id duy nhất, cắt lát
    on-the-fly trong __getitem__.
    """

    def __init__(self, data_ids: torch.Tensor, block_size: int):
        self.data = data_ids
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + 1: idx + self.block_size + 1]
        return x, y


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root =  Path(__file__).parent.parent
    save_dir = root / 'models'

    train_text = open(root.parent / 'data' / 'train.txt' , "r", encoding="utf-8").read()
    val_text = open(root.parent / 'data' / 'val.txt', "r", encoding="utf-8").read()

    tokenizer = Tokenizer(corpus=train_text)  # xây vocab từ train, val DÙNG CHUNG vocab này

    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)

    train_dataset = CharDataset(train_ids, block_size=Config.block_size)
    val_dataset = CharDataset(val_ids, block_size=Config.block_size)

    dataloader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=2)

    model = Model(device=device)

    model.build(vocab_size=tokenizer.vocab_size,
            d_model=Config.d_model,
            num_heads=Config.num_heads,
            num_layers=Config.num_layers,
            block_size=Config.block_size)

    print('Initialized model !')

    optimizer = torch.optim.AdamW(model.model.parameters(), lr=Config.lr)  # sketch gốc gọi Adam() thiếu params, và tạo TRƯỚC model
    criterion = nn.CrossEntropyLoss(ignore_index=0)                   # sketch gốc: criterion = torch (vô nghĩa)

    model.train(val_dataloader=val_dataloader, dataloader=dataloader ,
                criterion=criterion , optimizer=optimizer, save_dir=save_dir)

    model.save(save_dir=save_dir, optimizer=optimizer, criterion=criterion, tokenizer=tokenizer)

    