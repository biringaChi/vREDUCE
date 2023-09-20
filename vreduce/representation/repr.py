__author__ = "biringaChi (Chidera Biringa)" 

def prepare_sequence(sequence):
    d2a_stmts, d2a_exprs, d2a_decls = sequence
    d2a_stmts_sequence, d2a_exprs_sequence, d2a_decls_sequence = [], [], []
    for ftr_stmts, ftr_exprs, ftr_decls in zip(d2a_stmts, d2a_exprs, d2a_decls, ):
        d2a_stmts_sequence.append("".join([str(data) for data in ftr_stmts]))
        d2a_exprs_sequence.append("".join([str(data) for data in ftr_exprs]))
        d2a_decls_sequence.append("".join([str(data) for data in ftr_decls]))
    train_sequence = []
    for stmts_seq, exprs_seq, decls_seq in zip(d2a_stmts_sequence, d2a_exprs_sequence, d2a_decls_sequence):
        temp = stmts_seq + exprs_seq + decls_seq
        train_sequence.append(temp)
    return train_sequence


def batch(sequence, nsteps = 1):
    for idx in range(0, len(sequence), nsteps):
        yield sequence[idx : min(idx + nsteps, len(sequence))]