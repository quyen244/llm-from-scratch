
import torch 
import torch.nn as nn
from typing import Dict
from pathlib import Path
from datetime import datetime
from src.modules.decoder import DecoderBlock
from src.utils.config import Config
from src.modules.normalization import RMSNorm


class Tokenizer:
    """Char-level, xây vocab ĐỘNG từ corpus thay vì hard-code 4 ký tự."""

    def __init__(self, corpus: str):
        chars = sorted(list(set(corpus)))
        # id 0 dành riêng cho ký tự lạ / không có trong vocab train
        self.stoi: Dict[str, int] = {ch: i + 1 for i, ch in enumerate(chars)}
        self.itos: Dict[int, str] = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi) + 1  # +1 cho id 0

    def encode(self, text: str) -> torch.Tensor:
        ids = [self.stoi.get(ch, 0) for ch in text]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, indices: torch.Tensor) -> str:
        return "".join(self.itos.get(int(i), "") for i in indices.tolist())


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, block_size: int):
        super().__init__()
        self.token_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(num_embeddings=block_size, embedding_dim=d_model)

    def forward(self, index: torch.Tensor) -> torch.Tensor:
        B, T = index.shape
        positions = torch.arange(T, device=index.device)
        # (B,T,d_model) + (T,d_model) -> broadcast đúng theo batch
        return self.token_emb(index) + self.pos_emb(positions)


class MyModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, block_size: int):
        super().__init__()
        self.embedding = Embedding(vocab_size=vocab_size, d_model=d_model, block_size=block_size)
        self.decoder = nn.ModuleList([  # sketch gốc dùng nn.Sequential(list) -> lỗi, Sequential không nhận list
            DecoderBlock(d_model=d_model, num_heads=num_heads, block_size=block_size)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(in_features=d_model, out_features=vocab_size)  # sketch gốc để out_features=num_heads (sai)
        self.lm_head.weight = self.embedding.token_emb.weight  # weight tying (mẹo chuẩn của GPT-2/nanoGPT)
        # Mặc định nn.Embedding init N(0,1) -> cộng với weight tying làm logits scale ~sqrt(d_model),
        # loss ban đầu ~86 thay vì ~ln(vocab_size). Init std=0.02 kiểu GPT-2 để loss xuất phát đúng.
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for block in self.decoder:
            x = block(x)
        x = self.final_norm(x)  # KHÔNG cắt out[-1] như sketch gốc (nó lấy nhầm sample cuối trong batch)
        return self.lm_head(x)   # (B, T, vocab_size) - logits ở MỌI vị trí, cần cho loss (B,T)

class Model:
    def __init__(self, device):
        self.device = device
        self.model = None
        self.optimizer = None
        self.config = None  # nhớ kiến trúc để save/load dựng lại được y hệt

    # sketch gốc TẠO model xong không bao giờ .to(device)
    def build(self, vocab_size, d_model, num_heads, num_layers, block_size):
        self.config = {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "block_size": block_size,
        }
        self.model = MyModel(**self.config).to(self.device)
        return self.model

    @staticmethod
    def infer_config(state_dict):
        """Suy ra kiến trúc từ state_dict (dùng cho checkpoint cũ không lưu config).

        num_heads KHÔNG suy ra được từ shape (qkv luôn là d_model*3) -> lấy từ Config.
        """
        vocab_size, d_model = state_dict["lm_head.weight"].shape
        block_size = state_dict["embedding.pos_emb.weight"].shape[0]
        num_layers = 1 + max(
            int(k.split(".")[1]) for k in state_dict if k.startswith("decoder.")
        )
        return {
            "vocab_size": int(vocab_size),
            "d_model": int(d_model),
            "num_heads": Config.num_heads,
            "num_layers": num_layers,
            "block_size": int(block_size),
        }

    def train(self, dataloader, val_dataloader, criterion, optimizer,
              epochs: int = None, log_every: int = None, grad_clip: float = 1.0):
        assert self.model is not None, "Gọi build() (hoặc load()) trước khi train()"
        self.optimizer = optimizer
        epochs = Config.epochs if epochs is None else epochs
        log_every = Config.log_every if log_every is None else log_every

        self.model.train()
        history = []

        for epoch in range(epochs):
            running_loss = 0.0
            for step, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad(set_to_none=True)                    # sketch gốc THIẾU dòng này
                logits = self.model(x)                                   # (B,T,vocab_size)
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                loss.backward()                                          # sketch gốc THIẾU dòng này
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                optimizer.step()                                         # sketch gốc THIẾU dòng này

                running_loss += loss.item()
                if step % log_every == 0:
                    print(f"epoch {epoch} step {step}/{len(dataloader)} "
                          f"loss {running_loss / (step + 1):.4f}", flush=True)

            train_loss = running_loss / max(len(dataloader), 1)
            val_loss = self.evaluate(val_dataloader, criterion)
            history.append((train_loss, val_loss))
            print(f"== epoch {epoch} xong | train loss {train_loss:.4f} | val loss {val_loss:.4f} ==",
                  flush=True)

        return history

    @torch.no_grad()
    def evaluate(self, dataloader, criterion):
        if dataloader is None or len(dataloader) == 0:
            return float("nan")
        was_training = self.model.training
        self.model.eval()
        total_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()
        self.model.train(was_training)
        return total_loss / len(dataloader)


    def save(self, save_dir, optimizer=None, tokenizer=None):
        # Tạo thư mục nếu chưa tồn tại
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Tạo tên file với timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = save_dir / f"model_{timestamp}.pth"
        
        # Tạo checkpoint
        checkpoint = {
            'model_state_dict': self.model.state_dict(),  # Lưu state_dict, không phải model
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'config': self.config,      # thiếu cái này thì load() không dựng lại được kiến trúc
            'tokenizer': tokenizer,
            'timestamp': timestamp
        }
        
        # Lưu
        torch.save(checkpoint, filepath)
        print(f'Model saved to: {filepath}')
        return filepath
    
    def load(self, save_path, map_location=None):
        """Load checkpoint. Tự build model nếu chưa build (kiến trúc lấy từ checkpoint).

        Trả về tokenizer đã lưu kèm, hoặc None nếu lỗi.
        """
        map_location = map_location or self.device
        save_path = Path(save_path)
        if not save_path.exists():
            print(f"Error: File not found at {save_path}")
            return None

        checkpoint = torch.load(save_path, map_location=map_location, weights_only=False)
        state_dict = checkpoint.get('model_state_dict') or checkpoint.get('model')
        if state_dict is None:
            print("Error: checkpoint không có 'model_state_dict'")
            return None

        # dựng lại kiến trúc: ưu tiên config trong checkpoint, không có thì suy từ state_dict
        if self.model is None:
            config = checkpoint.get('config') or self.infer_config(state_dict)
            print(f"Building model từ checkpoint config: {config}")
            self.build(**config)

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        print('Model loaded successfully')

        if self.optimizer is not None and checkpoint.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Optimizer loaded successfully')

        self.model.eval()  # inference mode (quan trọng!)

        tokenizer = checkpoint.get('tokenizer')
        if tokenizer is not None:
            print(f'Tokenizer loaded successfully (vocab_size={tokenizer.vocab_size})')
            if tokenizer.vocab_size != self.model.lm_head.out_features:
                print("CẢNH BÁO: vocab_size của tokenizer khác với lm_head của model!")
        return tokenizer

    @torch.no_grad()
    def generate(self, tokenizer: Tokenizer, prompt: str, max_new_tokens: int = 200,
                temperature: float = 1.0, top_p: float = 0.9) -> str:
        """
        Sinh văn bản autoregressive từ 1 prompt.
        - temperature: chia logits trước softmax. <1 -> phân phối "nhọn" hơn (ít ngẫu nhiên),
        >1 -> "phẳng" hơn (sáng tạo/loạn hơn).
        - top_p (nucleus sampling): chỉ giữ tập token nhỏ nhất có tổng xác suất >= top_p,
        cắt bỏ phần đuôi xác suất thấp trước khi sample.
        """
        assert self.model is not None, "Gọi build() hoặc load() trước khi generate()"
        self.model.eval()
        block_size = (self.config or {}).get("block_size", Config.block_size)

        if not prompt:
            prompt = "\n"  # model cần ít nhất 1 token để bắt đầu
        ids = tokenizer.encode(prompt).unsqueeze(0).to(self.device)  # (1, T)
        prompt_len = ids.size(1)

        for _ in range(max_new_tokens):
            # positional embedding chỉ học tới block_size vị trí -> phải crop context nếu chuỗi dài hơn
            ids_cond = ids[:, -block_size:]
            logits = self.model(ids_cond)                                    # (1, T, vocab)
            logits = logits[:, -1, :] / max(temperature, 1e-6)          # chỉ cần logits ở VỊ TRÍ CUỐI
            logits[:, 0] = float("-inf")   # id 0 = <unk>/pad, decode ra "" -> cấm sinh ra nó

            probs = torch.softmax(logits, dim=-1)                       # (1, vocab)

            # --- top-p / nucleus filtering ---
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)

            remove_mask = cum_probs > top_p
            remove_mask[..., 1:] = remove_mask[..., :-1].clone()  # dịch phải 1 -> luôn giữ token vừa vượt ngưỡng
            remove_mask[..., 0] = False                            # luôn giữ ít nhất 1 token
            sorted_probs[remove_mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            next_in_sorted = torch.multinomial(sorted_probs, num_samples=1)  # sample theo index đã sort
            next_id = sorted_idx.gather(-1, next_in_sorted)                   # map ngược lại token id gốc

            ids = torch.cat([ids, next_id], dim=1)

        # chỉ trả về phần MỚI sinh ra: cắt theo token id, không cắt theo len(prompt) của
        # chuỗi đã decode (ký tự lạ decode thành "" -> lệch độ dài -> cắt sai)
        return tokenizer.decode(ids[0, prompt_len:])


    def chat(self, tokenizer: Tokenizer, max_new_tokens: int = 200,
            temperature: float = 0.8, top_p: float = 0.9):
        """
        Lưu ý: đây là base LM char-level train bằng next-token prediction thuần trên Shakespeare,
        KHÔNG phải model đã instruction-tune -> "chat" ở đây nghĩa là đưa 1 đoạn mồi (prompt),
        model tiếp tục viết theo văn phong đã học, không phải hỏi-đáp thật.
        """
        assert tokenizer is not None, "Không có tokenizer (load() thất bại?)"
        print(f"Gõ 'exit' để thoát. (temperature={temperature}, top_p={top_p})")
        while True:
            try:
                prompt = input("You: ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if prompt.strip().lower() in ("exit", "quit"):
                break
            continuation = self.generate(tokenizer, prompt, max_new_tokens,
                                         temperature=temperature, top_p=top_p)
            print("Model:", continuation)









