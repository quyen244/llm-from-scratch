
import torch 
import torch.nn as nn
from typing import Dict
from torch.utils.data import Dataset
from datetime import datetime
from src.modules.decoder import DecoderBlock
from src.utils.config import Config
from src.modules.normalization import RMSNorm
import os 



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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for block in self.decoder:
            x = block(x)
        x = self.final_norm(x)  # KHÔNG cắt out[-1] như sketch gốc (nó lấy nhầm sample cuối trong batch)
        return self.lm_head(x)   # (B, T, vocab_size) - logits ở MỌI vị trí, cần cho loss (B,T)

class Model:
    def __init__(self , device):
        self.device = device
        self.model = None
         

    # sketch gốc TẠO model xong không bao giờ .to(device)
    def build(self, vocab_size, d_model , num_heads, num_layers, block_size):

        self.model = MyModel(
                    vocab_size=vocab_size,
                    d_model=d_model,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    block_size=block_size,
                ).to(self.device) 

    def train(self, val_dataloader , dataloader , criterion, optimizer, device , save_dir , log_every : int = 500):
        self.model.train() 

        for epoch in range(Config.epochs):
            running_loss = 0.0
            for step, (x, y) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()                                    # sketch gốc THIẾU dòng này
                logits = self.model(x)                                        # (B,T,vocab_size)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()                                          # sketch gốc THIẾU dòng này
                optimizer.step()                                         # sketch gốc THIẾU dòng này

                running_loss += loss.item()
                if step % log_every == 0:
                    print(f"epoch {epoch} step {step} loss {running_loss / (step + 1):.4f}")

            val_loss = self.evaluate(self.model, val_dataloader, criterion, device)
            print(f"== epoch {epoch} xong | train loss {running_loss/len(dataloader):.4f} | val loss {val_loss:.4f} ==")

    @torch.no_grad()
    def evaluate(model, dataloader, criterion, device):
        model.eval()
        total_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
        model.train()
        return total_loss / len(dataloader)


    def save(self , save_dir , optimizer = None , loss = None, tokenzer = None):
        os.mkdir(save_dir, exist_ok = True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filepath = f"model_{timestamp}.pth"

        checkpoint = {
            'model' : self.model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'loss' : loss,
            'tokenzer' : tokenzer
        }

        torch.save(checkpoint, filepath)
        print('Model saved to : ' ,filepath )
        return filepath

    def load(self, save_dir):
        try:
            checkpoint = torch.save(save_dir)
            self.model = checkpoint['model']
            print('Load model succesfully')
            
            return checkpoint['tokenzer']
        
        except Exception as e:
            print(f"error during loading model : {e}")
            return None

    
    @torch.no_grad()
    def generate(self, tokenizer: Tokenizer, prompt: str, max_new_tokens: int, device,
                temperature: float = 1.0, top_p: float = 0.9) -> str:
        """
        Sinh văn bản autoregressive từ 1 prompt.
        - temperature: chia logits trước softmax. <1 -> phân phối "nhọn" hơn (ít ngẫu nhiên),
        >1 -> "phẳng" hơn (sáng tạo/loạn hơn).
        - top_p (nucleus sampling): chỉ giữ tập token nhỏ nhất có tổng xác suất >= top_p,
        cắt bỏ phần đuôi xác suất thấp trước khi sample.
        """
        self.model.eval()
        ids = tokenizer.encode(prompt).unsqueeze(0).to(device)  # (1, T)

        for _ in range(max_new_tokens):
            # positional embedding chỉ học tới block_size vị trí -> phải crop context nếu chuỗi dài hơn
            ids_cond = ids[:, -Config.block_size:]
            logits = self.model(ids_cond)                                    # (1, T, vocab)
            logits = logits[:, -1, :] / max(temperature, 1e-6)          # chỉ cần logits ở VỊ TRÍ CUỐI

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

        self.train.train()
        return tokenizer.decode(ids[0])


    def chat(self, tokenizer: Tokenizer, device, max_new_tokens: int = 200,
            temperature: float = 0.8, top_p: float = 0.9):
        """
        Lưu ý: đây là base LM char-level train bằng next-token prediction thuần trên Shakespeare,
        KHÔNG phải model đã instruction-tune -> "chat" ở đây nghĩa là đưa 1 đoạn mồi (prompt),
        model tiếp tục viết theo văn phong đã học, không phải hỏi-đáp thật.
        """
        print(f"Gõ 'exit' để thoát. (temperature={temperature}, top_p={top_p})")
        while True:
            prompt = input("You: ")
            if prompt.strip().lower() == "exit":
                break
            output = self.generate(tokenizer, prompt, max_new_tokens, device,
                            temperature=temperature, top_p=top_p)
            continuation = output[len(prompt):]  # chỉ in phần model sinh thêm, bỏ lại prompt gốc
            print("Model:", continuation)









