__author__ = "biringaChi (Chidera Biringa)"

import pickle
from simpletransformers.language_representation import RepresentationModel

def pickle_data(data, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)
        
def unpickle_data(data):
    with open(data, "rb") as file:
        loaded = pickle.load(file)
    return loaded

def prepare_sequence(features):
    stmts, exprs, decls = features
    stmts_sequence, exprs_sequence, decls_sequence = [], [], []
    for ftr_stmts, ftr_exprs, ftr_decls in zip(stmts, exprs, decls):
        stmts_sequence.append("".join([str(data) for data in ftr_stmts]))
        exprs_sequence.append("".join([str(data) for data in ftr_exprs]))
        decls_sequence.append("".join([str(data) for data in ftr_decls]))
    sequence = []
    for stmts_seq, exprs_seq, decls_seq in zip(stmts_sequence, exprs_sequence, decls_sequence):
        temp = stmts_seq + exprs_seq + decls_seq
        sequence.append(temp)
    return sequence

def batch(sequence, nsteps = 1):
    for idx in range(0, len(sequence), nsteps):
        yield sequence[idx : min(idx + nsteps, len(sequence))]

def _gpt(sequence, name, batch_n = 32, cuda = False):
    vectors = []
    gpt = RepresentationModel(model_type = "gpt2", model_name = "gpt2", use_cuda = cuda)
    for x in batch(sequence, batch_n):
        vectors.append(gpt.encode_sentences(x, combine_strategy = "mean", batch_size = len(x)))
    vectors = [i for vector in vectors for i in vector]
    pickle_data(vectors, name)
    print(len(vectors))

if __name__ == "__main__":
    sequence = unpickle_data("train_seqeunce.pkl")
    _gpt(sequence, "vectors.pkl")