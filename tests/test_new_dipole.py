from pysol.inductance import *

def run_ldx_tests():


    LC = {}
    LC["r1"] = .4-.2
    LC["r2"] = .4+.2
    LC["z1"] = .95-.05
    LC["z2"] = .95+.05
    LC["nt"] = 37
    LC["at"] = 37*1000
    LC["fil"] = FilamentCoil(LC, 16, 4)

    F1 = {}
    F1["r1"] = .35 - .1
    F1["r2"] = .35 + .1
    F1["z1"] = -.1
    F1["z2"] = +.1
    F1["nt"] = 1470
    F1["at"] = 1470 * 1000
    F1["fil"] = FilamentCoil(F1, 10, 10)


    FC = {}
    # FC["fil"] = np.concatenate((F1["fil"], F2["fil"], F3["fil"]))
    FC["fil"] = F1["fil"]
    FC["nt"] = F1["nt"] # + F2["nt"] + F3["nt"]
    FC["at"] = F1["at"] # + F2["at"] + F3["at"]

    print("Inductances of LC by approximations")
    print(LMaxwell(LC), LLyle4(LC), LLyle6(LC), LLyle6A(LC))

    print("Self-Inductance of FC by approximations and mutuals")
    LF1 = LLyle6(F1)
    #LF2 = LLyle6(F2)
    #LF3 = LLyle6(F3)
    #MF12 = TotalM0(F1, F2)
    #MF13 = TotalM0(F1, F3)
    #MF23 = TotalM0(F2, F3)
    #LFC = LF1 + LF2 + LF3 + 2 * MF12 + 2 * MF13 + 2 * MF23
    print("LFC %f H" % (LF1))

    MFL = TotalM0(FC, LC)
    #MFC = TotalM0(FC, CC)
    FFL = TotalFz(FC, LC)

    print("Mutual of F & L: %f mH" % (MFL * 1000))
    #print("Mutual of F & C: %f H" % (MFC))
    print("Force between F & L in %6.2f kg" % (FFL / 9.81))

if __name__ == "__main__":
    run_ldx_tests()
