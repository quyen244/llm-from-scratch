import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

# cho phép chạy trực tiếp `python src/scripts/train.py` lẫn `python -m src.scripts.train`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Console Windows mặc định là cp1252 -> print tiếng Việt/ký tự lạ sẽ UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr, sys.stdin):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from src.model import Tokenizer, Model
from src.utils.config import Config


class CharDataset(Dataset):
    """
    KHÔNG cắt trước hàng triệu substring ra RAM (sketch gốc làm vậy -> rất tốn bộ nhớ
    và mỗi item vẫn là string thô, chưa encode). Lưu 1 tensor id duy nhất, cắt lát
    on-the-fly trong __getitem__.
    """

    def __init__(self, data_ids: torch.Tensor, block_size: int):
        assert len(data_ids) > block_size + 1, (
            f"Dữ liệu quá ngắn ({len(data_ids)} token) so với block_size={block_size}"
        )
        self.data = data_ids
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + 1: idx + self.block_size + 1]
        return x, y


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu: {path}")
    return path.read_text(encoding="utf-8")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(__file__).resolve().parents[2]
    save_dir = root / "models"

    train_text = read_text(root / "data" / "train.txt")
    val_text = read_text(root / "data" / "val.txt")

    print(f"device={device} | train={len(train_text)} ký tự | val={len(val_text)} ký tự")
    if len(train_text) < len(val_text):
        print("CẢNH BÁO: train.txt NHỎ HƠN val.txt - hai file có thể đang bị đảo ngược.")

    # Vocab xây từ train + val: val vẫn là dữ liệu giữ riêng để đánh giá, nhưng nếu chỉ
    # lấy vocab từ train thì mọi ký tự lạ trong val thành id 0 (bị ignore_index bỏ qua)
    # -> val loss trông đẹp giả tạo.
    tokenizer = Tokenizer(corpus=train_text + val_text)
    print(f"vocab_size = {tokenizer.vocab_size}")

    train_ds = CharDataset(tokenizer.encode(train_text), block_size=Config.block_size)
    val_ds = CharDataset(tokenizer.encode(val_text), block_size=Config.block_size)
    print(f"train samples = {len(train_ds)} | val samples = {len(val_ds)}")

    # Windows spawn lại toàn bộ process cho mỗi worker -> với dataset in-memory này
    # num_workers=0 nhanh hơn hẳn.
    num_workers = 0 if os.name == "nt" else 2
    pin = device == "cuda"

    train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=Config.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)

    model = Model(device=device)
    model.build(vocab_size=tokenizer.vocab_size,
                d_model=Config.d_model,
                num_heads=Config.num_heads,
                num_layers=Config.num_layers,
                block_size=Config.block_size)

    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"Initialized model ! ({n_params/1e6:.2f}M params)")

    # sketch gốc gọi Adam() thiếu params, và tạo TRƯỚC model
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=Config.lr)
    # sketch gốc: criterion = torch (vô nghĩa). ignore_index=0 = token <unk>/padding.
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model.train(dataloader=train_loader, val_dataloader=val_loader,
                criterion=criterion, optimizer=optimizer,
                epochs=Config.epochs, log_every=Config.log_every)

    model.save(save_dir=save_dir, optimizer=optimizer, tokenizer=tokenizer)

    # sinh thử 1 đoạn ngay sau khi train để biết model có học được gì không
    print("\n--- sample ---")
    print(model.generate(tokenizer, prompt="KATHARINA:\n", max_new_tokens=200, temperature=0.8))


if __name__ == "__main__":
    main()
