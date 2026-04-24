"""
HisTagPred — predict optimal His-tag placement from protein sequence.

Usage:
    python predict.py --sequence MKTAYIAKQRST...
    python predict.py --fasta proteins.fasta
    python predict.py --csv input.csv --seq_col sequence

Output:
    C_terminal — place His-tag at C-terminus
    N_terminal — place His-tag at N-terminus
"""

import argparse
import numpy as np
import pandas as pd
import joblib
import torch
import warnings
warnings.filterwarnings("ignore")


def load_esm2():
    """Load ESM-2 650M model."""
    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, alphabet, device


def extract_terminal_embedding(sequence, model, alphabet, device,
                                window=15, layer=33):
    """Extract concatenated N+C terminal ESM-2 embeddings."""
    seq = str(sequence).upper().strip()
    n_window = seq[:window]
    c_window = seq[-window:] if len(seq) >= window else seq

    embeddings = []
    for window_seq in [n_window, c_window]:
        batch_converter = alphabet.get_batch_converter()
        data = [("protein", window_seq)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)

        with torch.no_grad():
            results = model(tokens, repr_layers=[layer],
                          return_contacts=False)
        token_repr = results["representations"][layer]
        # Mean pool over sequence length (exclude BOS/EOS tokens)
        emb = token_repr[0, 1:-1].mean(0).cpu().numpy()
        embeddings.append(emb)

    return np.concatenate(embeddings)  # 2560-dim


def predict_single(sequence, model_pkl, scaler_pkl,
                   esm_model, alphabet, device):
    """Predict His-tag placement for a single sequence."""
    clf    = joblib.load(model_pkl)
    scaler = joblib.load(scaler_pkl)

    emb = extract_terminal_embedding(sequence, esm_model, alphabet, device)
    X   = scaler.transform(emb.reshape(1, -1))

    pred  = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]

    label = "C_terminal" if pred == 0 else "N_terminal"
    conf  = max(proba)

    return {
        "prediction"  : label,
        "confidence"  : round(float(conf), 4),
        "prob_C_term" : round(float(proba[0]), 4),
        "prob_N_term" : round(float(proba[1]), 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="HisTagPred: Predict optimal His-tag placement")
    parser.add_argument("--sequence", type=str,
                        help="Single protein sequence")
    parser.add_argument("--fasta",    type=str,
                        help="FASTA file with protein sequences")
    parser.add_argument("--csv",      type=str,
                        help="CSV file with sequences")
    parser.add_argument("--seq_col",  type=str, default="sequence",
                        help="Column name for sequences in CSV")
    parser.add_argument("--model",    type=str,
                        default="models/xgboost_final.pkl")
    parser.add_argument("--scaler",   type=str,
                        default="models/scaler_final.pkl")
    parser.add_argument("--output",   type=str, default=None,
                        help="Output CSV path (optional)")
    args = parser.parse_args()

    print("Loading ESM-2 650M model (this may take 1-2 minutes)...")
    esm_model, alphabet, device = load_esm2()
    print(f"Device: {device}")

    results = []

    if args.sequence:
        res = predict_single(args.sequence, args.model, args.scaler,
                             esm_model, alphabet, device)
        res["sequence"] = args.sequence
        results.append(res)
        print(f"\nPrediction : {res['prediction']}")
        print(f"Confidence : {res['confidence']:.4f}")
        print(f"P(C-term)  : {res['prob_C_term']:.4f}")
        print(f"P(N-term)  : {res['prob_N_term']:.4f}")

    elif args.fasta:
        from Bio import SeqIO
        records = list(SeqIO.parse(args.fasta, "fasta"))
        print(f"\nProcessing {len(records)} sequences...")
        for rec in records:
            res = predict_single(str(rec.seq), args.model, args.scaler,
                                esm_model, alphabet, device)
            res["id"]       = rec.id
            res["sequence"] = str(rec.seq)
            results.append(res)
            print(f"  {rec.id:30s}  {res['prediction']:12s}  "
                  f"conf={res['confidence']:.3f}")

    elif args.csv:
        df = pd.read_csv(args.csv)
        assert args.seq_col in df.columns,             f"Column '{args.seq_col}' not found in CSV"
        print(f"\nProcessing {len(df)} sequences...")
        for i, row in df.iterrows():
            res = predict_single(row[args.seq_col], args.model,
                                 args.scaler, esm_model, alphabet, device)
            for col in df.columns:
                res[col] = row[col]
            results.append(res)
            print(f"  [{i+1}/{len(df)}]  {res['prediction']:12s}  "
                  f"conf={res['confidence']:.3f}")

    else:
        parser.print_help()
        return

    if results and args.output:
        out_df = pd.DataFrame(results)
        out_df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")
        print(out_df[["prediction","confidence"]].value_counts())


if __name__ == "__main__":
    main()
