##########################################################################
## 					PSEUDO-EIGENWERT-REDUZIERER							##
#------------------------------------------------------------------------#
# Dieses ist das "Main"-Programm.                                        #
##########################################################################
import numpy as np  # Mathematische Funktionen
import timeit  # Für die Berechnungszeit
import multiprocessing as mp  # Paralleles Rechnen
from build_hamiltonian import build  # Den Hamiltonian bauen
from lauf import Lauf  # Der tatsächliche Walk
from functions import Basisvektoren, print_exact, startpunkte
# Praktische selsbtdefinierte Funktionen


def alles(num_starting_points=None, hilf=[0.3, 0.5, 0], ortho_startp=True):
	#### Konstanten ####
	Ll = 4  # Länge des Systems
	N_u = Ll//2 + Ll % 2  # Elektronen mit Spin Up
	N_d = Ll//2  # Elektronen mit Spin Down
	# U = np.sqrt(2)
	# J = 1.0

	#### Variablen ####
	steps = 10000  # Die Schritte, die das Programm geht
	picks = 0  # Die Anzahl der Picks, siehe weiter unten
	prob = 250  # Die Wahrscheinlichkeit, energiereichere Zustände zu behalten
	# (In Promille)
	num_results = 1  # Anzahl Wiederholungen der Walks, damit das niedrigste
	# Ergebnis genommen werden kann
	smooth = 0  # Anzahl der "smooths", also "leerer Schritte" - siehe weiter unten

	H, basis_Ns = build(Ll, N_u, N_d)

	starting_points = startpunkte(
		basis_Ns, num_starting_points, ortho_startp=ortho_startp)

	# Hier wird immer das "Lauf"-Objekt vom besten Walk gespeichert.
	L_best = None

	# Von jedem defininerten Startpunkt wird ein Walk gestartet:
	for it in range(len(starting_points)):

		num_steps_fwd = 0
		num_steps_bck = 0

		# Nun startet ein Walk und wird in der Variable "L" gespeichert. #
		L = Lauf(starting_points[it], steps, H, basis_Ns, num_results, smooth,
		         picks, prob, hilf=hilf, all_steps_back=True, sim_ann=True)

		if L_best == None or L.eigenwert < L_best.eigenwert:
			# Der erste Walk oder der Beste von allen nachfolgenden wird in
			# "L_best" gespeichert. "Gut" heißt hierbei "geringer
			# Energiewert", da dies die interessante Größe ist.
			L_best = L

	##------------------------------------------------------------------------##
	## Je nachdem, was von Interesse ist, im folgenden Part auskommentiert    ##
	## lassen oder nicht:                                                     ##
	##------------------------------------------------------------------------##

	# print("steps = ", steps)
	# print("Erwartungswert = ", np.real(L_best.eigenwert), " für L = ", Ll)
	# print("Vektor dazu = ", L_best.eigenvektor)
	# print("Calculated Error = ", L_best.error)
	# print("Actual Error = ", np.linalg.norm(L_best.eigenvektor - v))
	# print("Gebrauchte Zeit: ", L_best.zeit[0], "Sekunden")
	# print("Average steps forward:", (L_best.num_steps_fwd/num_results)*100/steps, "%")
	# print("Average steps missed:", (L_best.num_steps_bck/num_results)*100/steps, "%")
	# print("Average of good steps:", (L_best.num_good_steps/num_results)*100/steps, "%")
	# kv, ke = print_exact(H)
	# print("Relative deviation: ", (np.real(L.eigenwert)-np.real(ke))/np.real(ke))
	print()

	return


alles(num_starting_points=5, ortho_startp=False)

################################################################################
# Sollte das von der Geschwindigkeit her nicht reichen, kann auch der folgende #
# Part benutzt werden, welcher das Programm parallel auf der CPU rechnen lässt.#
################################################################################
# pool = mp.Pool(mp.cpu_count())
# pool.map(alles, [i for i in range(3)])
# pool.close()
