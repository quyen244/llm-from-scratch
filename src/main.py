from src.model import  Model
from src.utils.config import Config
from pathlib import Path
import torch




if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = Path(__file__).parent.parent / 'models'

    model = Model()

    tokenizer = model.load(save_dir=save_dir)

    model.chat(tokenizer=tokenizer, device=device, temperature=0.2, float = 0.9)




   