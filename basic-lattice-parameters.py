# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 14:10:09 2021

@author: Georgia Nixon
"""

import numpy as np
def GetEvalsAndEvecs(HF):
    """
    Get e-vals and e-vecs of Hamiltonian HF.
    Order Evals and correspoinding evecs by smallest eval first.
    Set the gauge for each evec; choosing the first non-zero element to be real and positive.
    Note that the gauge may be changed later by multiplying any vec arbitrarily by a phase. 
    """
    #order by evals, also order corresponding evecs
    evals, evecs = eig(HF)
    idx = np.real(evals).argsort()
    evals = evals[idx]
    evecs = evecs[:,idx]
    
    #make first element of evecs real and positive
    for vec in range(np.size(HF[0])):
        
        # Find first element of the first eigenvector that is not zero
        firstNonZero = (evecs[:,vec]!=0).argmax()
        #find the conjugate phase of this element
        conjugatePhase = np.conj(evecs[firstNonZero,vec])/np.abs(evecs[firstNonZero,vec])
        #multiply all elements by the conjugate phase
        evecs[:,vec] = conjugatePhase*evecs[:,vec]

    # check that the evals are real
    if np.all((np.round(np.imag(evals),7) == 0)) == True:
        return np.real(evals), evecs
    else:
        print('evals are imaginary!')
        return evals, evecs


def H(V0, q, lmax):
    matrix = np.diag([(2*i+q)**2 + V0/2 for i in range(-lmax, lmax+1)],0)         
    matrix = matrix + np.diag([-V0/4]*(2*lmax ), -1) + np.diag([-V0/4]*(2*lmax), 1)
    return matrix

# lattice depth /Er
V0 = 10
# quasimomentum
qlist = np.linspace(-1, 1, 0.01)
# max plane wave index, theoretically infinite
lmax = 6
