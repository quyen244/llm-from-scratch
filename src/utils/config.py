class Config:
    # kiến trúc
    d_model = 128
    num_heads = 4
    num_layers = 4
    block_size = 128          # context length
    # training
    batch_size = 64
    lr = 3e-4
    epochs = 1
    log_every = 200

