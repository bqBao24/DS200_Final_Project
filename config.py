class Config:
    img_folder = '/content/drive/MyDrive/TransVG/train2014'
    train_label_path = '/content/drive/MyDrive/TransVG/labels_refcoco_train.pth' 
    val_label_path = '/content/drive/MyDrive/TransVG/labels_refcoco_val.pth' 
    # Model
    img_size = 224
    max_text_len = 16
    hidden_dim = 256
    text_encoder = 'bert-base-uncased'
    
    # Training
    batch_size = 32
    num_workers = 4
    num_epochs = 20
    learning_rate = 1e-3
    weight_decay = 0.05
    
    # Paths
    checkpoint_dir = './checkpoints'
    log_dir = './outputs/logs'