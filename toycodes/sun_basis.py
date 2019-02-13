import numpy as np

def sun_basis(basis_size):
    """
    Function that creates a basis for the generalized Pauli or Gell-Mann
    matrices. The collection of matrices are hermitian and trace-less by
    construction. They are orthogonal with respect to the inner product defined
    as: Tr(basis(i)@basis(j)) = 2delta_ij. Further they are aranged such that
    basis(0) = Identity, followed by the off-diagonal elements, then followed
    by the diagonal ones.
    ---------------------------------------------------------------------------
    Input:
        basis_size = int

    Output:
        np.array((basis_size**2,basis_size,basis_size),dtype=np.complex128)
    """
    basis = np.zeros((basis_size**2,basis_size,basis_size),dtype=np.complex128)

    # Identity
    basis[0] = np.identity(basis_size)*(np.sqrt(2/basis_size))

    # Off-diagonal elements
    index = 0
    for i in range(basis_size):
        for j in range(i+1,basis_size):
            index +=1
            #Symmetric case:
            basis[index,i,j] = 1.
            basis[index,j,i] = 1.

            index +=1
            #Antisymmetric case:
            basis[index,i,j] = -1.j
            basis[index,j,i] =  1.j

    # Other diagonal elements
    for d in range(1,basis_size):
        norm = np.sqrt(2.0/(d*(d+1)))
        index = basis_size**2-basis_size + d # to distiguish which element we write
        for j in range(d):
            basis[index,j,j] = norm
        basis[index,j+1,j+1] = -(j+1)*norm
    return basis

def pauli_basis():
    """
    Special case of sun_basis, where basis_size is set to n = 2. Returns the
    Pauli basis.
    ---------------------------------------------------------------------------
    Input:

    Output:
        np.array((4,2,2),dtype=np.complex128)
    """
    return sun_basis(2)


def spin_operators(S):
    """
    Create the spin operators for arbitrary spin.

    Input:
    S - integer

    Output:
        Eye,Sp,Sm,Sx,Sy,Sz
    """
    d = int(np.rint(2*S + 1))
    dz = np.zeros(d)
    mp = np.zeros(d-1)

    for n in range(d-1):
        dz[n] = S - n
        mp[n] = np.sqrt((2*S - n)*(n + 1))

    dz[d - 1] = -S
    Sp = np.diag(mp,1)
    Sm = np.diag(mp,-1)
    Sx = 0.5*(Sp + Sm)
    Sy = -0.5j*(Sp - Sm)
    Sz = np.diag(dz)
    Eye = np.diag(d*[1.])
    return Eye,Sp,Sm,Sx,Sy,Sz
