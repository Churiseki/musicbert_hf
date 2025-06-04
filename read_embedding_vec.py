import torch

# === 參數 ===
EMBEDDING_PATH = "./all_embeddings.pt"

# 載入整個 dict
all_embeddings = torch.load(EMBEDDING_PATH)

# 顯示有哪些 MIDI 檔案
print(f"[INFO] Loaded {len(all_embeddings)} embeddings:")
for fname in list(all_embeddings.keys())[:5]:
    print(f" - {fname}: shape = {all_embeddings[fname].shape}")

# ✅ 範例：取得某個 MIDI 檔的 embedding
filename = "your_file.mid"  # ← 替換成你想看的檔案名
if filename in all_embeddings:
    embedding = all_embeddings[filename]  # shape: [seq_len, hidden_size]
    print(f"[OK] Embedding shape for {filename}: {embedding.shape}")
else:
    print(f"[WARN] {filename} not found in embedding file.")
