##########################################################################
## 					    DER EIGENTLICHE WALK   							##
#------------------------------------------------------------------------#
# Hier wird ein vollständiger Walk gemacht. Zurückgegeben wird ein       #
# "Lauf"-Objekt mit allen Ergebnissen des Walks als Variablen.           #
##########################################################################
import numpy as np
from random import choice, random, gauss
import timeit
from functions import Steps, Pick, Smooth, winkel, curve, Prob


class Lauf:
    def __init__(self, starting_point, number_steps, H, basis_Ns, number_results=1, smooth=0, picks=0, prob=250, hilf=[], all_steps_back=False, sim_ann=True):
        #################################################
        #     VARIABLES
        #################################################
        ew2 = []   # Es gibt jeweils drei Stufen, in denen das Programm aussieben
        ew3 = []   # soll: nach jedem Schritt (ew1), nach jedem Walk(ew2) und
        # nach allen num_results Durchgängen. Das gleiche gilt für gs.
        gs2 = []
        gs3 = []   # gs = GroundStates. Speichert die energieärmsten Zustände.

        all_steps = []  # Hilfsliste mit allen Einzelschritten

        error = []  # Abweichung von einem Eigenwert
        s = []     # Zeit in Sekunden

        num_steps_bck = 0  # Alle Schritte, die abgelehnt wurden
        num_steps_fwd = 0  # Alle Schritte, die gemacht wurden
        num_good_steps = 0  # Alle Schritte mit DeltaE < 0

        start = timeit.default_timer()  # Startzeitpunkt

        for j in range(number_results):  # Der jeweilige Walk wird initialisiert: #
            psi = starting_point
            # Der erste Zustand wird initialisiert: zuerst einmal nur Nullen
            e1, ew1 = 0, 0  # Dies wird die Energie sein
            e2 = 0
            n = 1  # Die Schritte werden zurückgesetzt
            # Nun ist "Psi" ein gültiger Zustand

        #!!!      Der folgende Teil ist noch nicht genügend getestet.      !!!#
        #
        #     psi, ew1 = Pick(picks, psi, H, basis_Ns)
        #
    	#     # Zuerst die Picks, damit "Smooth" nicht überschrieben wird
    	#     # An dieser Stelle wird "Smooth" aufgerufen
        #     psi, n = Smooth(smooth,  psi, basis_Ns, n)
        #
    	# #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
    	# # Zurzeit ergibt es keinen Sinn, Smooth zu verwenden, wenn Picks verwendet #
    	# # wird. Der einzige Vorteil, Smooth zu verwenden, ist also um die Rechen-  #
    	# # zeit zu drücken, wenn (zu Testzwecken z.B.) die Genauigkeit der Ergeb-   #
    	# # nisse nicht wichtig ist.												 #
    	# #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#

            # "kurve" ist hier dann das "curve"-Objekt, welches später für die
            # Schrittlänge verantwortlich sein wird.
            kurve = curve(number_steps, hilf[0], hilf[1], hilf[2], prt=False)

            psi_new = gs1 = psi  # gs1, genauso wie ew1, werden initialisiert
            e1 = psi.T.dot(H.dot(psi))  # Energie des Zustands

            for i in range(number_steps):  # Der aktive Teil des Walks startet:

    	        # Durch einen Schritt veränderter Zustand
                psi_new, n = Steps(psi, basis_Ns, n, curve=kurve)
                # Auch von diesem die Energie nehmen
                e2 = psi_new.T.dot(H.dot(psi_new))

            	# Wenn die Energie nun größer ist:
            	# Mit abnehmender Wahrscheinlichkeit übernehmen
            	# Wenn die Energie nun kleiner ist:
            	# in jedem Fall übernehmen

                if e1 > e2 or random() < Prob(e2-e1, n, sim_ann):
                    psi = psi_new
                    if e1 > e2:
                        num_good_steps += 1
                    num_steps_fwd += 1
                    e1 = e2

                else:
                    n -= 1
                    num_steps_bck += 1

                if all_steps_back:
                    # Hier werden alle Erwartungswerte im Verlauf des Walks
                    # gespeichert. Interessant für Veranschaulichung und
                    # Optimierung.
                    all_steps.append(e1)

       		    ########################
       		    # Die Rückfallvariable #
       	        ########################
                if min(ew1, e1) == e1:
                    ew1 = e1
                    gs1 = psi
                    ###################################################################
                    # ew1 speichert so immer den geringsten Wert, der vorkam. Daher   #
                    # ist garantiert dass, falls ein Walk aus einem Minimum ins       #
                    # andere laufen und sich das als Fehler erweisen sollte, trotzdem #
                    # der tiefste Wert, der vorkam, gewählt wird.					  #
                    ###################################################################

            ###########################################################
            # Am Ende eines Walks die besten Erwartungswerte und die  #
            # dazugehörigen Zustände sammeln.                         #
            ###########################################################
            ew2.append(ew1)
            gs2.append(gs1)

        stop = timeit.default_timer()  # Endzeit
        s.append(stop-start)
        # Die Dauer der gesamten Rechnung wird hier gespeichert. #

        ############################################################
        # Am Ende eines Gesamtdurchlaufs die besten Ergebnisse der #
        # einzelnen Walks und die dazugehörigen Zustände sammeln.  #
        ############################################################
        index = ew2.index(min(ew2))
        ew3.append(ew2[index])
        gs3.append(gs2[index])

        error.append(winkel(gs2[index], H.dot(gs2[index])))
        # Mit ihm einen Richtwert zur Abweichung von einem Eigenwert errechnen.#

        self.error = error  # Der eben erwähnte Richtwert
        self.eigenwert = ew3  # Die errechnete Energie des Grundzustands
        self.eigenvektor = gs3[0]  # Der errechnete Grundzustand
        self.zeit = s  # Für den Gesamtdurchlauf gebrauchte Zeit in Sekunden
        self.num_steps_fwd = num_steps_fwd  # Alle Schritte, die gemacht wurden
        self.num_steps_bck = num_steps_bck  # Alle Schritte, die abgelehnt wurden
        self.num_good_steps = num_good_steps  # Alle Schritte mit DeltaE < 0
        self.all_steps = all_steps  # Alle Energien im Verlauf des Walks
        self.curve = kurve  # Die benutzten Schrittlängen im Verlauf des Walks
