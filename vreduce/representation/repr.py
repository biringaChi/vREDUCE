__author__ = "biringaChi (Chidera Biringa)" 

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