__author__ = "biringaChi (Chidera Biringa)"

import time
import pickle
from simpletransformers.language_representation import RepresentationModel


def serialize(features):
    return "".join([str(data) for data in features])

def prepare_sequence(features):
    stmts, exprs, decls = features
    stmts_sequence, exprs_sequence, decls_sequence = ([] for _ in range(3))
    for ftr_stmts, ftr_exprs, ftr_decls in zip(stmts, exprs, decls):
        stmts_sequence.append(serialize(ftr_stmts)), exprs_sequence.append(serialize(ftr_exprs)), decls_sequence.append(serialize(ftr_decls))
    sequence = []
    for stmts_seq, exprs_seq, decls_seq in zip(stmts_sequence, exprs_sequence, decls_sequence):
        sequence.append(stmts_seq + exprs_seq + decls_seq)
    return sequence

def _batch(sequence, nsteps = 1):
    for idx in range(0, len(sequence), nsteps):
        yield sequence[idx : min(idx + nsteps, len(sequence))]

def _gpt(sequence, name, batch_n = 32, cuda = True):
    vectors = []
    gpt = RepresentationModel(model_type = "gpt2", model_name = "gpt2", use_cuda = cuda)
    for x in _batch(sequence, batch_n):
        vectors.append(gpt.encode_sentences(x, combine_strategy = "mean", batch_size = len(x)))
    vectors = [i for vector in vectors for i in vector]
    pickle_data(vectors, name)
    print(len(vectors))

if __name__ == "__main__":
    sequence = unpickle_data("d2a_sequence.pkl")
    start = time.time()
    print(">_D2A Starts")
    _gpt(sequence, "d2a_embeddings.pkl")
    print(f">_D2A Ends: {time.time() - start}")