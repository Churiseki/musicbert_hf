import sys
import os
import torch
from miditok import Octuple
from miditoolkit import MidiFile
from itertools import chain

# 讓你可以 import musicbert_hf
sys.path.append(".")
from musicbert_hf.checkpoints import load_musicbert_from_fairseq_checkpoint

# === 參數設定 ===
MIDI_DIR = "./midis"  # MIDI 資料夾路徑
CKPT_PATH = "./checkpoint_last_musicbert_base.pt"
SAVE_PATH = "./all_embeddings.pt"  # 輸出檔案

# 載入 tokenizer 和模型
tokenizer = Octuple()
model = load_musicbert_from_fairseq_checkpoint(CKPT_PATH)
model.eval()

# 取得所有 .mid / .midi 檔案
midi_files = [f for f in os.listdir(MIDI_DIR) if f.endswith((".mid", ".midi"))]
print(f"[INFO] Found {len(midi_files)} MIDI files.")

all_embeddings = {}

# 逐一處理每個 MIDI 檔
for fname in midi_files:
    fpath = os.path.join(MIDI_DIR, fname)
    try:
        midi = MidiFile(fpath)
        tokens = tokenizer(midi)[0].ids
        tokens = list(chain.from_iterable(tokens))
        seq_len = len(tokens)
        tokens = tokens[: seq_len - (seq_len % 8)]  # 確保長度是 8 的倍數

        if len(tokens) == 0:
            print(f"[WARN] Skipped {fname}: token length is 0 after truncation.")
            continue

        input_ids = torch.tensor(tokens).unsqueeze(0)  # [1, seq_len]
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            embedding = outputs.hidden_states[-1].squeeze(0)  # [seq_len, hidden_size]
            all_embeddings[fname] = embedding  # 存下來

            print(f"[OK] Processed {fname}, shape: {embedding.shape}")
    except Exception as e:
        print(f"[ERROR] Failed processing {fname}: {e}")

# 存成一個 .pt 檔案
torch.save(all_embeddings, SAVE_PATH)
print(f"[DONE] Saved all embeddings to {SAVE_PATH}")
