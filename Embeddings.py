import sys
import torch
import pickle
from pathlib import Path, PureWindowsPath
from itertools import chain
from miditoolkit import MidiFile
from miditok import Octuple
from tqdm import tqdm

# 讓你可以 import musicbert_hf
sys.path.append(".")
from musicbert_hf.checkpoints import load_musicbert_from_fairseq_checkpoint


class EmbeddingGenerator:
    def __init__(self, ckpt_path: str, device: str = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[INFO] Using device: {self.device}")

        self.tokenizer = Octuple()
        self.model = load_musicbert_from_fairseq_checkpoint(ckpt_path)
        self.model.eval().to(self.device)

    def _load_midi_tokens(self, midi_path: Path):
        try:
            midi = MidiFile(str(midi_path))
            tokens = self.tokenizer(midi)[0].ids
            tokens = list(chain.from_iterable(tokens))
            seq_len = len(tokens)
            tokens = tokens[: seq_len - (seq_len % 8)]
            return tokens if tokens else None
        except Exception as e:
            print(f"[ERROR] Failed to tokenize MIDI: {midi_path} → {e}")
            return None

    def compute_embeddings(self, doc_pkl_path: str):
        with open(doc_pkl_path, "rb") as f:
            docs = pickle.load(f)

        all_embeddings = {}
        for track_id, info in tqdm(docs.items(), desc="Processing MIDI files"):
            midi_path = Path(PureWindowsPath(info["file_path"]))
            if not midi_path.exists():
                print(f"[WARN] MIDI file not found: {midi_path}")
                continue

            tokens = self._load_midi_tokens(midi_path)
            if tokens is None:
                print(f"[WARN] No valid tokens: {midi_path}")
                continue

            input_ids = torch.tensor(tokens).unsqueeze(0).to(self.device)
            attention_mask = torch.ones_like(input_ids).to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                embedding = outputs.hidden_states[-1][:, 0, :]  # shape: [1, 768]
                all_embeddings[info["track_id"]] = embedding.cpu()

        return all_embeddings

    def save_embeddings(self, embeddings: dict, output_path: str):
        torch.save(embeddings, output_path)
        print(f"[INFO] Saved {len(embeddings)} embeddings to {output_path}")


if __name__ == "__main__":
    DOCS_JSON = "./ppr4env_index/documents.pkl"
    CKPT_PATH = "./checkpoint_last_musicbert_base.pt"
    OUTPUT_PATH = "./all_embeddings.pt"

    embedder = EmbeddingGenerator(ckpt_path=CKPT_PATH)
    embeddings = embedder.compute_embeddings(DOCS_JSON)
    embedder.save_embeddings(embeddings, OUTPUT_PATH)
