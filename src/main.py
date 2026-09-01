import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Console Windows mặc định là cp1252 -> print tiếng Việt/ký tự lạ sẽ UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr, sys.stdin):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from src.model import Model
from src.utils.config import Config


def latest_checkpoint(models_dir: Path) -> Path:
    """Lấy checkpoint mới nhất thay vì hard-code đường dẫn tuyệt đối như bản cũ."""
    ckpts = sorted(models_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(
            f"Không có checkpoint nào trong {models_dir}. Chạy `python -m src.scripts.train` trước."
        )
    return ckpts[-1]


def parse_args(root: Path):
    parser = argparse.ArgumentParser(description="Chat với LLM char-level đã train")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Đường dẫn file .pth (mặc định: checkpoint mới nhất trong models/)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Sinh một lần với prompt này rồi thoát (không vào chat loop)")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    return parser.parse_args()


def main():
    root = Path(__file__).resolve().parents[1]
    args = parse_args(root)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.checkpoint or latest_checkpoint(root / "models")
    print(f"device={device} | checkpoint={ckpt_path}")

    model = Model(device=device)
    # load() tự build kiến trúc từ checkpoint -> KHÔNG cần build() thủ công nữa
    # (bản cũ gọi load() khi self.model vẫn là None -> AttributeError bị nuốt mất)
    tokenizer = model.load(save_path=ckpt_path, map_location=device)
    if tokenizer is None:
        print("Load thất bại: checkpoint không kèm tokenizer, không thể encode/decode.")
        return 1

    print(model.model)

    if args.prompt is not None:
        print(args.prompt, end="")
        print(model.generate(tokenizer, args.prompt,
                             max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature, top_p=args.top_p))
        return 0

    model.chat(tokenizer=tokenizer,
               max_new_tokens=args.max_new_tokens,
               temperature=args.temperature,
               top_p=args.top_p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
