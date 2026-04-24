"""
HisTagPred — extract ESM-2 terminal embeddings from protein sequences.

Usage:
    python extract_embeddings.py --input  data/final_dataset.csv
                                 --seq_col fasta_sequence
                                 --output  X_esm_float32.npy
"""

import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def load_esm2(model_name="esm2_t33_650M_UR50D"):
    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    print(f"ESM-2 loaded on {device}")
    return model, alphabet, device


def get_embedding(seq, model, alphabet, device, window=15, layer=33):
    seq     = str(seq).upper().strip()
    n_win   = seq[:window]
    c_win   = seq[-window:] if len(seq) >= window else seq
    embs    = []
    batch_converter = alphabet.get_batch_converter()
    for w in [n_win, c_win]:
        _, _, tokens = batch_converter([("p", w)])
        tokens = tokens.to(device)
        with torch.no_grad():
            out = model(tokens, repr_layers=[layer], return_contacts=False)
        emb = out["representations"][layer][0, 1:-1].mean(0).cpu().numpy()
        embs.append(emb)
    return np.concatenate(embs)


def main():
    parser = argparse.ArgumentParser(
        description="Extract ESM-2 terminal embeddings")
    parser.add_argument("--input",    type=str, required=True,
                        help="CSV file with sequences")
    parser.add_argument("--seq_col",  type=str, default="fasta_sequence",
                        help="Column name containing sequences")
    parser.add_argument("--output",   type=str, required=True,
                        help="Output .npy file path")
    parser.add_argument("--window",   type=int, default=15,
                        help="Terminal window size (default: 15)")
    parser.add_argument("--layer",    type=int, default=33,
                        help="ESM-2 layer to extract (default: 33)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    assert args.seq_col in df.columns,         f"Column '{args.seq_col}' not in {list(df.columns)}"
    print(f"Extracting embeddings for {len(df)} sequences...")

    model, alphabet, device = load_esm2()

    embeddings = []
    for seq in tqdm(df[args.seq_col], desc="Extracting"):
        emb = get_embedding(seq, model, alphabet, device,
                           window=args.window, layer=args.layer)
        embeddings.append(emb)

    X = np.array(embeddings, dtype=np.float32)
    np.save(args.output, X)
    print(f"\nSaved embeddings: {X.shape} → {args.output}")


if __name__ == "__main__":
    main()
