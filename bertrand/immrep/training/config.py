from transformers import BertConfig
from bertrand.model.tokenization import tokenizer

# BERT config, i.e. number of Transformer layers, attention heads, embedding dimension sizes
BERT_CONFIG = BertConfig(
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=64,
    type_vocab_size=2,
    num_attention_heads=8,
    num_hidden_layers=8,
    hidden_size=512,
    intermediate_size=2048,
)

BERT_CONFIG_PTCR = BertConfig(
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=70,
    type_vocab_size=3,
    num_attention_heads=8,
    num_hidden_layers=8,
    hidden_size=512,
    intermediate_size=2048,
)

# Training args for peptide:TCR binding prediction
SUPERVISED_TRAINING_ARGS = dict(
    num_train_epochs=20,  # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=256,  # batch size for evaluation
    warmup_ratio=0.1,  # number of warmup steps for learning rate scheduler
    learning_rate=1e-5,  # learning rate
    weight_decay=1e-5,  # strength of weight decay
    logging_dir="./logs",  # directory for storing logs
    logging_steps=10,  # logging every 10 steps
    evaluation_strategy="epoch",  # model is evaluated every epoch
    save_strategy="epoch",  # model is saved every epoch
)

# Training args for MLM
MLM_TRAINING_ARGS = dict(
    num_train_epochs=100,  # total number of training epochs
    per_device_train_batch_size=1024,  # batch size per device during training
    per_device_eval_batch_size=1024,  # batch size for evaluation
    warmup_steps=10000,  # number of warmup steps for learning rate scheduler
    learning_rate=5e-5,  # learning rate
    weight_decay=1e-4,  # strength of weight decay
    logging_dir="./logs",  # directory for storing logs
    logging_steps=100,  # log every 100 steps
    evaluation_strategy="epoch",  # model is evaluated every epoch
    save_strategy="epoch",  # model is saved every epoch
    # eval_steps=5,
)

# Training args for MLM
MLM_TRAINING_ARGS_PTCR = dict(
    num_train_epochs=100,  # total number of training epochs
    per_device_train_batch_size=512+256,  # batch size per device during training
    per_device_eval_batch_size=512+256,  # batch size for evaluation
    warmup_ratio=0.01,  # number of warmup steps for learning rate scheduler
    learning_rate=5e-5,  # learning rate
    weight_decay=1e-4,  # strength of weight decay
    logging_dir="./logs",  # directory for storing logs
    logging_steps=100,  # log every 100 steps
    evaluation_strategy="epoch",  # model is evaluated every epoch
    save_strategy="epoch",  # model is saved every epoch
    # eval_steps=5,
)
