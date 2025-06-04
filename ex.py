import sys
import torch
from miditok import Octuple
from miditoolkit import MidiFile
from itertools import chain

# ✅ 讓你可以 import musicbert_hf
sys.path.append(".")

from musicbert_hf.checkpoints import load_musicbert_from_fairseq_checkpoint


# === 參數設定 ===
MIDI_PATH = "../10013.mid"  # ← 改成你的 MIDI 檔案路徑
CKPT_PATH = "./checkpoint_last_musicbert_base.pt"  # 模型權重路徑

# === Step 1: 載入 tokenizer 並將 MIDI 轉為 tokens ===
tokenizer = Octuple()
midi = MidiFile(MIDI_PATH)
tokens = tokenizer(midi)[0].ids  # list[int]

print(f"[INFO] Token count: {len(tokens)}")

# === Step 2: 載入 MusicBERT 模型 ===

from musicbert_hf.checkpoints import load_musicbert_from_fairseq_checkpoint

model = load_musicbert_from_fairseq_checkpoint(CKPT_PATH)
# 設定為評估模式
model.eval()


# === Step 3: Token → Tensor → Embedding ===

tokens = list(chain.from_iterable(tokens))
seq_len = len(tokens)
tokens = tokens[: seq_len - (seq_len % 8)]

input_ids = torch.tensor(tokens).unsqueeze(0)  # shape: [1, seq_len]
attention_mask = torch.ones_like(input_ids)

print(f"tokens type: {type(tokens)}")
print(f"tokens[0] type: {type(tokens[0]) if isinstance(tokens, list) else 'N/A'}")
print(f"tokens[:10]: {tokens[:10]}")

# === Step 4: 將 tokens 餵進模型取得 embedding ===
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True
    )
    embeddings = outputs.hidden_states[-1]
    #cls_output = outputs.last_hidden_state[:, 0, :] # Extract the CLS vector
print(outputs.keys())
#print(outputs.hidden_states)  # [batch_size, seq_len, hidden_size]
#print(f'cls_output.shape: {cls_output.shape}')
print(f"[INFO] Embedding shape: {embeddings.shape}")
print(embeddings)
