##########################################################################
## 					        DER HAMILTONIAN 							##
#------------------------------------------------------------------------#
# Hier wird der Hamiltonian zusammengebaut. Dank der "Quspin"-Bibliothek #
# wird dieser Hamiltonian nicht als Matrix sondern als Hamiltonian-      #
# Objekt zurückgegeben; also essenziell eine Liste mit Operatoren        #
# (Funktionsanweisungen). Das macht Operationen sehr viel einfacher.     #
##########################################################################
# Zum Erstellen solcher "Hamiltonian-Objekte"
from quspin.operators import hamiltonian
import quspin.basis as qb  # Die Basis
import math  # Für die Wurzel


def build(L, N_u, N_d, U=math.sqrt(2), J=1.0, basis_back=False):
    #########################################################################
    # 				Der Hamiltonian wird nun zusammengebaut.				#
    #########################################################################

    #### Die Basis definieren #####
    basis = qb.spinful_fermion_basis_1d(L=L, Nf=(N_u, N_d))

    onsite = [[U, i, i, ] for i in range(L)]			 # Coulomb Abstoßung
    hop_right = [[J, i, (i+1) % L] for i in range(L)]    # t-Term nach rechts
    hop_left = [[-J, i, (i+1) % L] for i in range(L)]	 # t-Term nach links

    static = [						# Zeitunabhängiger Teil des Hamiltonians
    		["+-|", hop_left],
    		["-+|", hop_right],
    		["|+-", hop_left],
    		["|-+", hop_right],
    		["n|n", onsite],
    		]
    dynamic = []					# Zeitabhängiger Teil

    ## Beides zusammensetzen zum Hamiltonian ##
    H = hamiltonian(static, dynamic, basis=basis,
                    check_symm=False, check_herm=False, check_pcon=False)
    if basis_back == True:
        return H, basis
    return H, basis.Ns
