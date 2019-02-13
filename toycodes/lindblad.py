import numpy as np
import sun_basis
import sys
from scipy.linalg import expm
import time
# from tools import svd_robust as svd

def make_superoperator(A):
    """
    Function that transforms an operator A into a superoperator A^#. Essentially it
    implements A^#_ij = 1/2 Tr(sigma_i A sigma_j), where sigma_i is a generalized
    (and normalized, see module) Gell-Mann matrix (for local Hilbert space of
    d = 2, this will be the set of Pauli matrices).

    Note that this (and henceforth in this code) assumes left multiplication of
    the superoperator. One may use the function commutator to implement e.g. an
    operator that may be right multiplied, or the function Lindblad.
    """
    d = A.shape[0]
    S = sun_basis.sun_basis(d)
    A_super = 0.5*np.tensordot(S,np.tensordot(
                A,S,axes = ([1,1])),axes = ([2,1],[0,2]))
    return A_super

def make_superbasis(basis):
    """
    This is essentially a helper function that converts a basis obtained via the
    sun_basis module into a superoperator basis.
    """
    len = basis.shape[0]
    d = basis.shape[1]
    super_basis = np.zeros((len,d**2,d**2),dtype=np.complex128)
    for i in range(len):
        super_basis[i,:,:] = make_superoperator(basis[i])
    return super_basis

class MPS(object):
    """
    A basic MPS formulated in the usual TenPy language in order to accommodate
    density matrices. The usual B and s formalism is retained, but the physical
    indices do not any longer represent the state, but the coefficients of the
    representation of the density matrix in the space of (generalized) Pauli
    matrices.

    In other words, if rho = sum_i c_i sigma_i, then the physical indices are
    the c_i.

    The first set of methods is contraction methods on this MPS network. As we
    are now concerned with density matrices instead of states, this required
    slightly different contractions such as ...


    """
    def __init__(self):
        self.L = None
        self.chi = None
        self.d = None
        self.B = None
        self.s = None
        self.dtype = None

    # @classmethod
    # def magnetized_state(cls,p_state,dtype=float):
    #     """
    #     Method to initialise class with a magnetized state that is a propduct
    #     state of one-site density matrices for spin 1/2. (Maybe extend to larger
    #     spin later)
    #
    #     Input:
    #     p_state - type: list
    #         [1,1,0,0] corresponds to up, up, down, down
    #     """
    #     psi=cls()
    #     psi.d = 2
    #     psi.L = len(p_state)
    #     psi.dtype = dtype
    #
    #     psi.B = []
    #     psi.s = []
    #     for i in range(psi.L):
    #         B = np.zeros((4,1,1),dtype=dtype)
    #         B[0,0,0] = 1
    #         B[3,0,0] = (-1)**(p_state[i]-1)
    #         s = np.zeros(1)
    #         s[0] = 1.
    #         psi.B.append(B)
    #         psi.s.append(s)
    #     psi.s.append(s)
    #     psi.chi = (psi.L + 1) * [1]
    #     return psi

    @classmethod
    def mixed_state(cls,L,d,dtype=float):
        """
        Method to initialize MPS with product state of completeley mixed density
        matrices

        Input:
        L - length of the chain
        d - number of on-site basis states
        """
        psi=cls()
        psi.d = d
        psi.L = L
        psi.dtype = dtype

        psi.B = []
        psi.s = []
        for i in range(psi.L):
            B = np.zeros((d**2,1,1),dtype=dtype)
            B[0,0,0] = 1
            s = np.zeros(1)
            s[0] = 1.
            psi.B.append(B)
            psi.s.append(s)
        psi.s.append(s)
        psi.chi = (psi.L + 1) * [1]
        return psi

    def get_trace_all(self,B_list):
        """
        Method to calculate the total trace of the MPS. This uses the fact that
        there is only one basis state that is NOT traceless, the identity.

        This therefore consists of collapsing the first physical dimension of
        entire MPS.

        Input:
        List of B-matrices

        This currently takes an input, but I might want to change that in the
        future.
        """
        norm = B_list[0][0,:,:]
        for i in range(len(B_list)-1):
            norm = np.dot(norm,B_list[i + 1][0,:,:])
        return norm.flatten().real[0]

    def get_left_product(self,site):
        """
        Essentially tracing out the sites to the left of a specified site.
        """
        if site > 0:
            tmp = self.B[0][0,:,:]
            for i in range(site-1):
                tmp = np.dot(tmp,self.B[i+1][0,:,:])
        else:
            tmp = np.array([[1.]])
        return tmp

    def get_right_product(self,site):
        """
        Essentially tracing out the sites to the right of a specified site.
        """
        L = self.L
        if site < L-1:
            tmp = self.B[-1][0,:,:]
            for i in range(site+1,L-1)[::-1]:
                tmp = np.dot(self.B[i][0,:,:],tmp)
            tmp = np.dot(np.diag(self.s[site+1]),tmp)
        else:
            tmp = np.array([[1.]])
        return tmp

    def get_middle_product(self,sitea,siteb):
        """
        Essentially tracing out the middle between two specified sites.
        """
        if sitea - siteb > -1:
            raise ValueError("Wrong site indices, you idiot!")
        if sitea < 0 or siteb > self.L:
            raise ValueError("Wrong site indices, you idiot!")
        tmp = np.diag(self.s[sitea+1])
        for i in range(sitea+1,siteb,1):
            tmp = np.dot(tmp,self.B[i][0,:,:])
        return tmp

    def get_rho_onesite(self,site):
        """
        Method to obtain the one-site density matrix at a given site. For this
        we collapse the first dimension of the MPS to the left and right of the
        desired site. (i.e. tracing out the rest of the system)

        Input: site
        """
        rho_vec = np.zeros(self.d**2)
        left = self.get_left_product(site)
        right = self.get_right_product(site)
        onsite = self.B[site]*self.s[site+1]**(-1)
        for i in range(self.d**2):
            rho_vec[i]= np.real(np.dot(np.dot(left,onsite[i,:,:]),right)[0][0])
        return rho_vec

    def get_rho_twosite(self,sitea,siteb):
        """
        Method to obtain the two-site density matrix at two given sites. For this
        we collapse the first dimension of the MPS to the left, middle and right
        of the desired sites. (i.e. tracing out the rest of the system)

        Input: sitea, siteb
        """
        rho_vec = np.zeros((4,4),dtype=np.complex)
        left = self.get_left_product(sitea)
        middle = self.get_middle_product(sitea,siteb)
        right = self.get_right_product(siteb)
        sitel = self.B[sitea]*self.s[sitea+1]**(-1)
        siter = self.B[siteb]*self.s[siteb+1]**(-1)
        for i in range(4):
            for j in range(4):
                rho_vec[i,j] = np.real(np.dot(np.dot(np.dot(np.dot(left,sitel[i,:,:]),middle),siter[j,:,:]),right)[0][0])
        return rho_vec

    def act_operator_on_rho(self,operator,rho):
        """
        Function that applies superoperator to the wavefunction (B matrix).
        """
        return  np.tensordot(operator, rho, axes=(1,0))

    def get_expectation_n_site(self,mod):
        """
        Calculates the expectation value for a sequence of operators passed as
        a list of tuples: e.g.  [(1,O_1),...,(n,O_n)]:

            <O_1...O_n> = Tr(O_1...O_n rho)

        Returns the expectation value.
        """
        B_modified = [B for B in self.B]
        for site, op in mod:
            B_modified[site] = self.act_operator_on_rho(op, self.B[site])
        expectation = self.get_trace_all(B_modified)
        return expectation

class H_XYZ(object): #thing about making XYZ a derived class from H
    """
    Base class for the XYZ model for arbitrary spin
    """
    def __init__(self,H_params,Drive_params = None):
        self.verbose = H_params['verbose']
        self.U_list,self.H_list = [],[]
        self.H_params = H_params
        self.Drive_params = Drive_params
        self.L = H_params['L']
        self.S = H_params['S']
        self.basis = sun_basis.spin_operators(self.S)
        if Drive_params == None:
            print("There is no form of Lindblad driving used!")
            self.make_H_xyz()
        elif Drive_params['type'] == 'spin_bath':
            self.make_H_xyz_mu_bond()
            if self.verbose > 0:
                print("Spin-bath drive: The end fields are NOT set to zero.")
        else:
            raise ValueError("Drive not recognised, abort...")

    def lindblad(self,X):
        """
        The superoperators as created for this Hamiltonian presume left
        multiplication. One may use this function to create an operator that is
        implemented as a Lindblad 'jump' operator X, i.e. that acts as on rho as:

            [X rho,X^{\dagger}] + [X,rho X^{\dagger}]
        """
        Xp = np.transpose(np.conj(X))
        return 2.0*np.dot(X, np.transpose(Xp)) - np.dot(Xp,
                    X) - np.transpose(np.dot(Xp,X))

    def commutator(self,op):
        """
        The superoperators as created for this Hamiltonian presume left
        multiplication. One may use this function to create an operator that is
        implemented as a Lindblad 'jump' operator X, i.e. that acts as on rho as:

            [rho,op]

        such as found in the von-Neumann equation for the time-evolution of
        density matrices.
        """
        return op - np.transpose(op)

    def combine_to_H(self,o,L,Hl,Hr,Hlr):
        """
        For numerical stability it is advantageous to split on-site terms
        equally between neighbouring sites. This function takes care of that.
        """
        lfactor = .5
        rfactor = .5
        if o == 0:
            lfactor = 1.
        if o == L-2:
            rfactor = 1.
        EYE = np.identity(np.shape(Hl)[0])
        H = Hlr + lfactor * np.kron(Hl,EYE) + rfactor * np.kron(EYE,Hr)
        return H

    def make_H_xyz(self):
        """
        Create the bond evolution operator used by the TEBD algorithm. This takes
        the form

            H_XYZ = sum_i Jx sx_i sx_i+1 + Jy sy_i sy_i+1 + Jy sz_i sz_i+1
                        + h_i s_z_i

        """
        H_site = []
        L = self.L
        h  = self.H_params['h']
        Jx = self.H_params['Jx']
        Jy = self.H_params['Jy']
        Jz = self.H_params['Jz']
        EYE,SP,SM,SX,SY,SZ = tuple(make_superbasis(np.array(self.basis)))
        msg = ("Creating closed XYZ model with: L={L:d}, Jx={Jx:.2f}, Jy={Jy:.2f}, Jz={Jz:.2f}")
        if self.verbose > 1:
            print(msg.format(L=L,Jx=Jx,Jy=Jy,Jz=Jz))
            print('h: ',h)
            sys.stdout.flush()
        for i in range(L):
            H_site.append(1j * h[i] * (self.commutator(SZ)))
        H_pair = 1j * (Jx * self.commutator(np.kron(SX,SX))
            + Jy * self.commutator(np.kron(SY,SY))
            + Jz * self.commutator(np.kron(SZ,SZ)))
        for bond in range(L-1):
            i1 = bond
            i2 = bond + 1
            H = self.combine_to_H(i1,L,
                H_site[i1],H_site[i2],H_pair)
            self.H_list.append(H)
        if self.verbose > 0:
            print("Set up Hamiltonian...")

    def make_H_xyz_mu_bond(self):
        """
        Create the bond evolution operator used by the TEBD algorithm. This takes
        the form

            H_XYZ = sum_i Jx sx_i sx_i+1 + Jy sy_i sy_i+1 + Jy sz_i sz_i+1
                        + h_i s_z_i

        with additional Lindblad driving operators of the form:

            L_1 = sqrt(1+mu) s^+_1
            L_2 = sqrt(1-mu) s^-_1
            L_3 = sqrt(1-mu) s^+_L
            L_4 = sqrt(1+mu) s^-_L

        This non-unitary term is preceded by an overall factor kappa.

        """
        H_site = []
        L = self.L
        h  = self.H_params['h']
        mu = self.Drive_params['mu']
        kappa = self.Drive_params['kappa']
        Jx = self.H_params['Jx']
        Jy = self.H_params['Jy']
        Jz = self.H_params['Jz']
        EYE,SP,SM,SX,SY,SZ = tuple(make_superbasis(np.array(self.basis)))
        msg = ("Creating open XYZ model with: L={L:d}, Jx={Jx:.2f}, Jy={Jy:.2f}, Jz={Jz:.2f}, mu={mu:.4f}, kappa={kappa:.1f}")
        if self.verbose > 1:
            print(msg.format(L=L,Jx=Jx,Jy=Jy,Jz=Jz,mu=mu,kappa=kappa))
            print('h: ',h)
            sys.stdout.flush()
        for i in range(L):
            H_site.append(-1j * h[i] * (self.commutator(SZ)))
            if i == 0:
                H_site[-1] += kappa *((1+mu) * self.lindblad(SP)+((1-mu) *
                 self.lindblad(SM)))
            if i == L-1:
                H_site[-1] += kappa *((1-mu) * self.lindblad(SP)+((1+mu) *
                 self.lindblad(SM)))
        H_pair = -1j * (Jx * self.commutator(np.kron(SX,SX))
            + Jy * self.commutator(np.kron(SY,SY))
            + Jz * self.commutator(np.kron(SZ,SZ)))
        for bond in range(L-1):
            i1 = bond
            i2 = bond + 1
            H = self.combine_to_H(i1,L,
                H_site[i1],H_site[i2],H_pair)

            self.H_list.append(H)
        if self.verbose > 0:
            print("Set up Hamiltonian...")


    # Below are some expectations that one might want to calculate in the XYZ
    def magnetization(self,psi,site):
        EYE,SP,SM,SX,SY,SZ = tuple(make_superbasis(np.array(self.basis)))
        return psi.get_expectation_n_site([(site,SZ)])

    def currentS(self,psi,site):
        Jx = self.H_params['Jx']
        EYE,SP,SM,SX,SY,SZ = tuple(make_superbasis(np.array(self.basis)))
        return Jx * (psi.get_expectation_n_site([(site,SX),(site+1,SY)])
            - psi.get_expectation_n_site([(site,SY),(site+1,SX)]))

    def bondE(self,psi,site):
        L = psi.L
        lfactor = .5
        rfactor = .5
        Jx = self.H_params['Jx']
        Jy = self.H_params['Jy']
        Jz = self.H_params['Jz']
        h = self.H_params['h']
        EYE,SP,SM,SX,SY,SZ = tuple(make_superbasis(np.array(self.basis)))
        expectation = (
         + Jx * psi.get_expectation_n_site([(site,SX),(site + 1,SX)])
         + Jy * psi.get_expectation_n_site([(site,SY),(site+1,SY)])
         + Jz * psi.get_expectation_n_site([(site,SZ),(site+1,SZ)])
         + (lfactor * h[site] * psi.get_expectation_n_site([(site,SZ)])
               + rfactor * h[site + 1] * psi.get_expectation_n_site([(site +1,SZ)])))
        return expectation


class Engine_TEBD(object):
    def __init__(self, psi, model, TEBD_params):
        self.verbose = TEBD_params['verbose']
        self.TEBD_params = TEBD_params
        self.trunc_params = TEBD_params['trunc_params']
        self.psi = psi
        self.model = model
        self.evolved_time = TEBD_params['start_time']
        self.U_list = None
        self.U_param = {}
        self._update_index = None
        self.TEBD_stats = {}
        self.SS_conv = False
        self.SS_stats = {}

    def calc_U_bond(self,i_bond,dt):
        d = np.shape(self.psi.B[0])[0]
        H_to_exp =  dt * self.model.H_list[i_bond]
        H_exp = expm(H_to_exp)
        U = np.reshape(H_exp,(d,d,d,d))
        return U

    def calculate_U(self,order,delta_t):
        U_param = dict(order=order, delta_t=delta_t)
        self.U_list = []
        for dt in self.suzuki_trotter_time_steps(order):
            U_bond = [
                self.calc_U_bond(i_bond, dt * delta_t) for i_bond in range(self.psi.L-1)
            ]
            self.U_list.append(U_bond)
        self.U_param = U_param
        msg = ("--> Creating propagator with: TrotterOrder={order:d}, dt={delta_t:.4f}")
        if self.verbose > 1:
            print(msg.format(order=order,delta_t=delta_t))

    def update_bond(self,i_bond,U_bond):
        # print "Updating bond",i_bond
        B1 = self.psi.B[i_bond]
        B2 = self.psi.B[i_bond+1]
        sL = self.psi.s[i_bond]
        d = B1.shape[0]
        chi1 = B1.shape[1]
        chi3 = B2.shape[2]
        chi_max = self.trunc_params['chi_max']
        min_SV = self.trunc_params['min_SV']
        # print "B1 in",i_bond,B1.real.round(10)
        C = np.tensordot(B1,B2,axes=(2,1))
        C = np.tensordot(C,U_bond,axes=([0,2],[2,3]))
        theta = np.reshape(np.transpose(np.transpose(C)*sL,(1,3,0,2)),(d*chi1,d*chi3))
        C = np.reshape(np.transpose(C,(2,0,3,1)),(d*chi1,d*chi3))

        X, Y, Z = np.linalg.svd(theta)
        Z=Z.T

        W = np.dot(C,Z.conj()) # left s not yet included in c!
        chi2 = np.min([np.sum(Y>min_SV), chi_max])

        # Obtain the new values for B and s
        # However, the inverse of sL is problematic, as it might contain very small singular
        # values.  Instead, we calculate ``C == sL**-1 theta == sL**-1 U S V``,
        # such that we obtain ``B_L = sL**-1 U S = sL**-1 U S V V^dagger = C V^dagger``
        self.psi.s[i_bond + 1] = Y[:chi2]
        self.psi.B[i_bond] = np.reshape(W[:,:chi2],(d,chi1,chi2))
        self.psi.B[i_bond + 1] = np.transpose(np.reshape(Z[:,:chi2],
            (d,chi3,chi2)),(0,2,1))

    def update_step(self, U_idx_dt, odd):
        Us = self.U_list[U_idx_dt]
        for i_bond in np.arange(int(odd) % 2, self.psi.L-1, 2):
            self._update_index = (U_idx_dt, i_bond)
            self.update_bond(i_bond, Us[i_bond])
        self._update_index = None


    def update(self,N_steps):
        order = self.U_param['order']
        for U_idx_dt, odd in self.suzuki_trotter_decomposition(order, N_steps):
            t0 = time.time()
            self.update_step(U_idx_dt, odd)
        self.evolved_time = self.evolved_time + N_steps * self.U_param['delta_t']
        self.psi.chi = [len(i) for i in self.psi.s]
        #Normalisation
        trace = self.psi.get_trace_all(self.psi.B)
        for site in range(self.psi.L):
            self.psi.B[site] = self.psi.B[site]/(
                trace/1)**(1/np.float(self.psi.L))

    def measurement(self):
        L = self.psi.L
        mag = [self.model.magnetization(self.psi,site) for site in range(L)]
        bondE = [self.model.bondE(self.psi,bond) for bond in range(L-1)]
        currS = [self.model.currentS(self.psi,bond) for bond in range(L-1)]
        return mag,bondE,currS

    def run(self):
        delta_t = self.TEBD_params['dt']
        N_steps = self.TEBD_params['N_steps']
        TrotterOrder = self.TEBD_params['order']

        if self.evolved_time == 0:
            self.calculate_U(TrotterOrder,delta_t)
            self.TEBD_stats['mag'] = []
            self.TEBD_stats['bondE'] = []
            self.TEBD_stats['currS'] = []
            self.TEBD_stats['ts'] = []
            mag,bondE,currS = self.measurement()
            self.TEBD_stats['mag'].append(mag)
            self.TEBD_stats['bondE'].append(bondE)
            self.TEBD_stats['currS'].append(currS)
            self.TEBD_stats['ts'].append(self.evolved_time)

        mag_old,bondE_old,currS_old = self.measurement()
        start_time = time.time()

        self.update(N_steps)

        mag,bondE,currS = self.measurement()
        Delta_mag = np.average(np.abs(np.array(mag_old) - np.array(mag)))
        Delta_bondE = np.average(
            np.abs(np.array(bondE_old) - np.array(bondE)))
        Delta_currS = np.average(
            np.abs(np.array(currS_old) - np.array(currS)))
        step_time = time.time() - start_time
        msg = ("--> time={t:3.3f}, max_chi={chi:d}, D_mag={dm:.3e}, "
            + " D_bondE={dBE:.3e}, D_currS={dCS:.3e}, "
            + "since last update: {time:.1f} s")
        if self.verbose > 1.5:
            print(msg.format(t=self.evolved_time, chi=max(self.psi.chi),
            dm=Delta_mag, dBE=Delta_bondE,
            dCS=Delta_currS, time=step_time,))
            sys.stdout.flush()

        self.TEBD_stats['mag'].append(mag)
        self.TEBD_stats['bondE'].append(bondE)
        self.TEBD_stats['currS'].append(currS)
        self.TEBD_stats['ts'].append(self.evolved_time)
        return Delta_mag,Delta_bondE,Delta_currS,step_time

    @staticmethod
    def suzuki_trotter_time_steps(order):
        if order == 1:
            return [1.]
        elif order == 2:
            return [0.5, 1.]
        elif order == 4:
            t1 = 1. / (4. - 4.**(1 / 3.))
            t3 = 1. - 4. * t1
            return [t1 / 2., t1, (t1 + t3) / 2., t3]
        # else
        raise ValueError("Unknown order {0!r} for Suzuki Trotter decomposition".format(order))

    @staticmethod
    def suzuki_trotter_decomposition(order, N_steps):
        even, odd = 0, 1
        if N_steps == 0:
            return []
        if order == 1:
            a = (0, odd)
            b = (0, even)
            return [a, b] * N_steps
        elif order == 2:
            a = (0, odd)  # dt/2
            a2 = (1, odd)  # dt
            b = (1, even)  # dt
            # U = [a b a]*N
            #   = a b [a2 b]*(N-1) a
            return [a, b] + [a2, b] * (N_steps - 1) + [a]
        elif order == 4:
            a = (0, odd)  # t1/2
            a2 = (1, odd)  # t1
            b = (1, even)  # t1
            c = (2, odd)  # (t1 + t3) / 2 == (1 - 3 * t1)/2
            d = (3, even)  # t3 = 1 - 4 * t1
            # From Schollwoeck 2011 (arXiv:1008:3477):
            # U = U(t1) U(t2) U(t3) U(t2) U(t1)
            # with U(dt) = U(dt/2, odd) U(dt, even) U(dt/2, odd) and t1 == t2
            # Uusing above definitions, we arrive at:
            # U = [a b a2 b c d c b a2 b a] * N
            #   = [a b a2 b c d c b a2 b] + [a2 b a2 b c d c b a2 b a] * (N-1) + [a]
            steps = [a, b, a2, b, c, d, c, b, a2, b]
            steps = steps + [a2, b, a2, b, c, d, c, b, a2, b] * (N_steps - 1)
            steps = steps + [a]
            return steps
        # else
        raise ValueError("Unknown order {0!r} for Suzuki Trotter decomposition".format(order))
def Test_Lindblad(ts,L,hs,kappa,gamma1,gamma2):
    Id = QTP.qeye(2)
    Sx = QTP.sigmax()
    Sy = QTP.sigmay()
    Sz = QTP.sigmaz()
    Sp = QTP.sigmap()
    Sm = QTP.sigmam()

    Jx, Jy, Jz = 1.0, 1.0, 1.0
    mu = 0.001

    psi0 = QTP.tensor([QTP.qeye(2)/2.0]*L)

    #Make operators
    Sx_list = []
    Sy_list = []
    Sz_list = []

    for ii in range (0, L):
        op_list = []
        for jj in range (0, L):
            op_list.append(Id)
        op_list[ii] = Sx / 2.0
        Sx_list.append(QTP.tensor(op_list))
        op_list[ii] = Sy / 2.0
        Sy_list.append(QTP.tensor(op_list))
        op_list[ii] = Sz / 2.0
        Sz_list.append(QTP.tensor(op_list))

    #Make Hamiltonian
    Hbs, Ops = [], []
    Ops += Sz_list
    for n in range (0, L-1):
        Hb = hs[n] * Sz_list[n] * 0.5
        Hb += hs[n+1] * Sz_list[n+1] * 0.5
        Hb += Jx * Sx_list[n] * Sx_list[n+1]
        Hb += Jy * Sy_list[n] * Sy_list[n+1]
        Hb += Jz * Sz_list[n] * Sz_list[n+1]
        Hbs.append(Hb)
        'Current operators for each bond (XXZ model)'
        j = Sx_list[n] * Sy_list[n+1]
        j -= Sy_list[n] * Sx_list[n+1]
        Ops.append(j * Jx)
    H = np.sum(Hbs)
    H += 0.5 * (hs[0]*Sz_list[0] + hs[-1]*Sz_list[-1])


    #Make Lindblads
    L1 = QTP.tensor([np.sqrt(kappa*(1.0+mu)) * Sp] + [Id] * (L-1))
    L2 = QTP.tensor([np.sqrt(kappa*(1.0-mu)) * Sm] + [Id] * (L-1))
    L3 = QTP.tensor([Id] * (L-1) + [np.sqrt(kappa*(1.0-mu)) * Sp])
    L4 = QTP.tensor([Id] * (L-1) + [np.sqrt(kappa*(1.0+mu)) * Sm])
    L5 = np.sqrt(gamma1 / 2.0) * Sz_list[L//2-1]
    L6 = np.sqrt(gamma2 / 2.0) * Sz_list[L//2] #Yay this works
    LBs = [L1, L2, L3, L4, L5, L6]


    res = QTP.mesolve(H, psi0, ts, LBs, Ops, options = QTP.Options(rtol=1e-20))


    ii = 0
    Szt, JSzt = [], []
    for ii in range(0, 2*L-1):
        if ii < L: Szt.append(res.expect[ii])
        else: JSzt.append(res.expect[ii])

    return Szt,JSzt

def rho_maker_one_size(a):
    "Preliminary function to create rho on site!"
    d = int(np.sqrt(len(a)))
    tmp = np.zeros((d,d),dtype=np.complex128)
    for i in range(d**2):
        tmp += a[i]*basis[i]
    return tmp/np.trace(tmp)

if __name__ == '__main__':
    import qutip as QTP
    import matplotlib.pyplot as plt

    L = 6
    W = 0.5
    h = W * 2 *(np.random.rand((L))-0.5); #h[0]=1; h[-1] = 1
    max_time = 40
    ts = np.arange(0.0, max_time + 0.0001, 0.1)

    H_params = {
        'verbose':2.,
        'L':L,
        'Jx':1.,
        'Jy':1.,
        'Jz':1.,
        'h':h,
        'S':0.5}

    Drive_params = {
        'type':'spin_bath',
        'kappa':0.5,
        'mu':0.001}


    TEBD_params = {
        'verbose':2,
        'dt': 0.5,
        'order': 4,
        'start_time': 0,
        'N_steps': 2,
        'trunc_params': {'chi_max':50,'min_SV':10**(-8)}
    }

    # H = H_XYZ(H_params,None)

    H = H_XYZ(H_params,Drive_params)
    psi = MPS.mixed_state(L,int(np.rint(2*H_params['S'] + 1)))
    TEBD = Engine_TEBD(psi,H,TEBD_params)

    for i in range(max_time):
        TEBD.run()


    Szt,JSzt = Test_Lindblad(ts,L,h,1,0,0)



    for ii in range (0, L-1):
        plt.plot(TEBD.TEBD_stats['ts'],np.array(TEBD.TEBD_stats['currS'])[:,ii],
            label = 'TEBD bond %d-%d' %(ii+1,ii+2))
        plt.plot(ts, JSzt[ii], '--', label='bond %d-%d' %(ii+1,ii+2))
        plt.legend()
        plt.xlabel(r'$t$')
        plt.ylabel(r'$J_{S^z}$')

    plt.show()

    for ii in range (0, L):
        plt.plot(TEBD.TEBD_stats['ts'],np.array(TEBD.TEBD_stats['mag'])[:,ii],
            label = 'TEBD bond %d-%d' %(ii+1,ii+2))
        plt.plot(ts, Szt[ii], '--', label='bond %d-%d' %(ii+1,ii+2))
        plt.legend()
        plt.xlabel(r'$t$')
        plt.ylabel(r'$S^z$')

    plt.show()
