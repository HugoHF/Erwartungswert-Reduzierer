##########################################################################
## 					NÜTZLICHE UND NÖTIGE FUNKTIONEN						##
##########################################################################
import numpy as np
from random import choice, randrange, gauss, random
from scipy.constants import Boltzmann, pi


def print_exact(H):
	##########################################################################
	# Hier wird der Hamitlonian diagonalisiert und der tiefste Eigenwert     #
	# ausgegeben. In dieser Form kommt die Funktion schnell an ihre Grenzen, #
	# doch durch Benutzun von z.B. scipy.sparse kann das stark verbessert    #
	# werden.                                                                #
	##########################################################################

	x = np.linalg.eig(H.toarray())
	e = np.amin(x[0])
	v = x[1][:, np.argmin(x[0])]

	np.set_printoptions(precision=4, suppress=True)
	print("---------------")
	print(e)
	print("---------------")

	return v, e


def Steps(psi, basis_Ns, n, stepnum=1, curve=None):
	#################################################
	# Hier wird nun ein richtiger Schritt begangen. #
	#################################################

	vec = psi.copy()

	# Richtung des Schrittes soll zällig sein, aber senkrecht zu Psi.
	# Zufällige Richtung:
	v_richtung = np.random.rand(basis_Ns)
	# Nun anhand einer zufälligen Stelle den Vektor senkrecht machen:
	index = randrange(0, basis_Ns)
	v_richtung[index] = 0
	v_richtung[index] = -v_richtung.dot(vec) / vec[index]

	# Zugrundeliegende Formel: Zentralwinkel = Großkreis-Abstand d / Radius r
	# (Großkreise sind hier immer gegeben, da Psi sozusagen vom Mittelpunkt
	# "ausgeht")
	# r = 1, daher theta = d (in Radians)
	# Zudem ist step_length = d = curve.current(n):
	theta = curve.current(n)

	# Nun den schritt auch gehen
	# Trigonometrie: tan(theta) = |vec_richtung|/|psi| , wobei |vec_richtung|
	# die gesuchte Größe ist und |Psi| = 1.
	# Psi_neu ergibt sich also als vec + v_richtung mal die gewollte Länge
	# von tan(Theta):
	vec += (np.tan(theta) * v_richtung / np.linalg.norm(v_richtung))

	n += 1  # Begangene Schritte mitzählen
	return vec/np.linalg.norm(vec), n


class curve:
	############################################################################
	# Ein "Curve"-Objekt hat eine Funktion, die die jeweilige Schrittlänge     #
	# nach n Schritten zurückgibt. Initialisiert wird die Kurve durch die      #
	# Schrittlänge n, die Amplitude amp und der Mittelwert med (nicht mehr     #
	# dem Durchschnitt). Der sogenannte "Activation Point" actp bleibt noch zu #
	# testen.                                                                  #
	############################################################################

	def __init__(self, n, med=0.5, amp=0, actp=0, prt=True):
		self.n = n
		self.med = med
		self.amp = amp
		self.actp = actp
		# prt gibt an, ob die Charakteristika der Kurve geprintet werden sollen
		# oder nicht.
		if prt:
			print("[", med, ",", amp, ",", actp, "],")

	def current(self, i):
		if self.amp == 0:
			return (self.med)
		x = 1 - pow((i)/(self.n), 3)
		x *= self.amp
		x += self.med - 0.5*self.amp
		return (x)  # x ist hier die jeweilige Schrittlänge.


def winkel(vec1, vec2):
	#########################################################
	# Zurückgegeben wird der Winkel zwischen vec1 und vec2. #
	#########################################################
	return(np.arccos(abs(vec1.T.dot(vec2))/(np.linalg.norm(vec1)*np.linalg.norm(vec2))))


# def Verteilung(psi):
# 	v=0
# 	l=0
# 	for i in range(len(psi)):
# 		v += psi[i]*Verteilung_zustand(i)
# 		l += psi[i]
# 	v/=l
# 	return(v)
#
#
# def Verteilung_zustand(i):
# 	v = 0
# 	for i in range(L):
# 		if ((basis[i]//2**L)//2**i)%2 + (basis[i]// 2**i)% 2  == 1:
# 			v += 1
# 	v/=L
# 	return(v)
#   #####################################################################
#   # Diese beiden Methoden sollen einem Zustand einen Wert zwischen 0  #
# 	# und 1 zuordnen, je nachdem wieviele einfache Besetzungen der      #
#   # Zustand hat, um größere Zustände anschaulich Charakterisieren zu  #
#   # können. Noch in der Entwicklung.                                  #
#   #####################################################################


def Temp(n):
	###########################
	# Die Temperaturfunktion. #
	###########################

	T_zero = 200  # Starttemperatur
	return T_zero * 0.99**n  # Exponentieller Abfall


def Prob(diff, n, sim_ann):
	##########################################################################
	# Anhand der Differenz der Energien (diff) und den schon begangenen      #
	# Schritten (entspricht dem "Zeitpunkt" des Laufs) gibt diese Funktion   #
	# die entsprechende Akzeptanzwahrscheinlichkeit dieser Energiedifferenz  #
	# zurück. Wenn die Temperaturabhängigkeit dieser Wahrscheinlichkeit      #
	# hingegen ausgestellt ist (durc h "sim_ann"), wird eine konstante       #
	# Wahrscheinlichkeit angenommen (0.27).                                  #
	##########################################################################

	if not sim_ann:
		return 0.27
	boltz = 100000
	return np.real(1 / (1 + np.exp((diff/(boltz * Temp(n))))))


def Smooth(smooth, psi, basis_Ns, n):
	#######################################################################
	# Diese Methode "rundet den Zustand ab"; sie geht Schritte, ohne zu   #
	# prüfen, ob die Richtung korrekt ist. Dadurch sollte man mit einem   #
	# möglichst durchmischten Zustand bereits beginnen. Durch das Gram-   #
	# Schmidt-Verfahren (siehe weiter unten) wird sie aber vorerst        #
	# überflüssig.                                                        #
	#######################################################################

	for i in range(smooth):
		psi[randrange(0, basis_Ns)] += 1/n
		psi /= np.linalg.norm(psi)
		n += 1
	return psi, n


def Pick(picks, psi, H, basis_Ns):
	#########################################################################
	# Zufälliges "Herumstochern" im Hilbertraum und auswählen des kleinsten #
	# Elements. In der Theorie soll dadurch erreicht werden, dass der Walk  #
	# bereits in einem möglichst tiefen Minimum beginnt. Um effektiv ge-    #
	# nutzt werden zu können bedarf sie aber noch Testung.                  #
	#########################################################################

	E = []
	V = []
	if picks == 0:
		return psi, psi.dot(H.dot(psi))
		# Dieser Teil verhindert bloß eine Fehlermeldung bei picks = 0.
	for i in range(picks):
		psi_new = 2 * np.random.rand(basis_Ns) - 1  # Neuer Zufallszustand
		psi_new /= np.linalg.norm(psi_new)  # Normieren
		E.append(psi_new.dot(H.dot(psi_new)))  # Energie ausrechnen
		V.append(psi_new)
		print(min(E))
	index = E.index(min(E))  # Geringste Energie heraussuchen und zurückgeben
	return V[index], E[index]


def gram_schmidt_vektor(L):
	############################################################################
	# Diese Methode liefert, nach dem Gram-Schmidt-Verfahren, auf eine Menge L #
	# von Vektoren einen weiteren, zu allen vorherigen Vektoren senkrechten    #
	# Vektor. Das klappt also nur, wenn L mindestens einen Vektor, aber        #
	# wenige Vektoren als die Länge der Vektoren selbst behinhaltet. Jeder     #
	# Vektor danach wird ein rein zufälliger Vektor.                           #
	############################################################################

	v = np.random.rand(len(L[0]))
	# Die nächste Zeile ist gufgrund eines Bugs in der Numpy-Version 1.19.0
	# vonnöten und in vielen der Methoden zu finden.
	u = v.copy()
	for i in L:
		# Alle von i abhängigen Anteile in u heraussubtrahieren:
		u -= (v.T.dot(i) / i.T.dot(i)) * i
	return u/np.linalg.norm(u)


def Basisvektoren(L, n):
	############################################################################
	# Startend von mindestens einem Vektor in L gibt diese Methode eine Liste  #
	# mit n zueinander senkrechten Vektoren zurück.                            #
	############################################################################

	for i in range(n-len(L)):
		L.append(gram_schmidt_vektor(L))
	return L


def gleichverteilt(Ns):
	#########################################################################
	# Diese Methode gibt auf einer Sphäre in Ns Dimensionen einen           #
	# gleichverteilten Vektor zurück. Dazu braucht man interessanterweise   #
	# normalverteilte Komponenten (nach: Marsaglia, G.;                     #
	# doi:10.1214/aoms/1177692644 ).                                         #
	#########################################################################

	vec = np.zeros(Ns)
	for i in range(Ns):
		vec[i] = gauss(0, 1)
	return vec/np.linalg.norm(vec)


def startpunkte(basis_Ns, n=None, ortho_startp=True):
	#######################################################################
	# Hier wird spezifiziert,ob die Startpunkte senkrecht zueinander oder #
	# bloß zufällig gewählt werden sollen.                                #
	#######################################################################

	if n == None:
		n = basis_Ns
		# n kann freigelassen werden, dann wird per Default angenommen, dass
		# der Hilbertraum vollständig ausgenutzt werden soll, also so viele
		# Vektoren wie möglich.

	if not ortho_startp:
		# Zufällige Startpunkte
		starting_points = [gleichverteilt(basis_Ns) for i in range(n)]
		return starting_points

	# Zueinander senkrechte Startpunkte, beginnend mit einem
	vec = gleichverteilt(basis_Ns)
	starting_points = Basisvektoren([vec], n)
	return starting_points
