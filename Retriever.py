import torch
import json
from pathlib import Path, PureWindowsPath
from miditok import Octuple
from miditoolkit import MidiFile
from itertools import chain
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F

from musicbert_hf.checkpoints import load_musicbert_from_fairseq_checkpoint


class Retriever:
    def __init__(self, query_json_path, all_embeddings_path, ckpt_path, top_k=5):
        self.query_json_path = Path(PureWindowsPath(query_json_path))  # convert Windows path
        self.all_embeddings_path = Path(all_embeddings_path)
        self.ckpt_path = Path(ckpt_path)
        self.top_k = top_k

        # Load tokenizer
        self.tokenizer = Octuple()

        # Load MusicBERT model
        self.model = load_musicbert_from_fairseq_checkpoint(str(self.ckpt_path))
        self.model.eval()

        # Load database embeddings
        self.all_embeddings = torch.load(self.all_embeddings_path)
        
        # Load query metadata
        with open(self.query_json_path, "r") as f:
            self.queries = json.load(f)["queries"]

    def _embed_midi(self, midi_path):
        try:
            midi = MidiFile(midi_path)
            tokens = self.tokenizer(midi)[0].ids  # list[int]
            tokens = list(chain.from_iterable(tokens))
            tokens = tokens[: len(tokens) - (len(tokens) % 8)]
            input_ids = torch.tensor(tokens).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                embedding = outputs.hidden_states[-1][:, 0, :]  # CLS token
            return embedding.squeeze(0)  # shape: [hidden_dim]
        except Exception as e:
            print(f"[ERROR] Failed to embed {midi_path}: {e}")
            return None

    def retrieve(self):
        results = {}

        for query in self.queries:
            query_id = query["query_id"]
            query_path = Path(PureWindowsPath(query["query_midi_path"]))  # convert Windows path
            if not query_path.exists():
                print(f"[WARN] Query MIDI not found: {query_path}")
                continue

            print(f"[INFO] Processing {query_id}...")
            query_embedding = self._embed_midi(query_path)
            if query_embedding is None:
                continue

            similarities = []
            for track_id, db_embedding in self.all_embeddings.items():
                #print("query_embedding shape:", query_embedding.shape)
                #print("db_embedding shape:", db_embedding.shape)
                sim = F.cosine_similarity(
                    query_embedding.unsqueeze(0),  # from [768] → [1, 768]
                    db_embedding,                  # already [1, 768]
                    dim=1  
                ).item()
                similarities.append((track_id, sim))
                #print((track_id, sim))

            # Sort and keep top_k
            top_k = sorted(similarities, key=lambda x: x[1], reverse=True)[:self.top_k]
            results[query_id] = top_k
            print(f"[RESULT] {query_id} → {[t[0] for t in top_k]}")
        with open("retrieval_results.json", "w") as f:
            json.dump(results, f, indent=4)

        return results


if __name__ == "__main__":
    retriever = Retriever(
        query_json_path="queries_metadata.json",
        all_embeddings_path="all_embeddings.pt",
        ckpt_path="checkpoint_last_musicbert_base.pt",
        top_k=5
    )
    retrieval_results = retriever.retrieve()
    
    # 儲存結果
    with open("retrieval_results.json", "w") as f:
        json.dump(retrieval_results, f, indent=2)
