from pysol.inductance import *


def run_ldx_tests():
    HTS_LC = {}
    HTS_LC["r1"] = 0.41 / 2
    HTS_LC["r2"] = 1.32 / 2
    HTS_LC["z1"] = 1.610 - 0.018 / 2
    HTS_LC["z2"] = 1.610 + 0.018 / 2
    HTS_LC["nt"] = 2796
    HTS_LC["at"] = 2796 * 105
    HTS_LC["fil"] = FilamentCoil(HTS_LC, 30, 2)

    LC = {}
    LC["r1"] = 0.246
    LC["r2"] = 0.70
    LC["z1"] = 1.525
    LC["z2"] = LC["z1"] + 0.1
    LC["nt"] = 80
    LC["at"] = 3500 * 80
    LC["fil"] = FilamentCoil(LC, 20, 4)

    CC = {}
    CC["r1"] = 0.645
    CC["r2"] = 0.787
    CC["z1"] = -0.002 - 0.750 / 2
    CC["z2"] = -0.002 + 0.750 / 2
    CC["nt"] = 8388
    CC["at"] = 8388 * 420
    CC["fil"] = FilamentCoil(CC, 3, 7)

    F1 = {}
    F1["r1"] = 0.2717 - 0.01152 / 2
    F1["r2"] = 0.2717 + 0.01152 / 2
    F1["z1"] = -0.0694 / 2
    F1["z2"] = +0.0694 / 2
    F1["nt"] = 26.6
    F1["at"] = 26.6 * 1629
    F1["fil"] = FilamentCoil(F1, 2, 4)

    F2 = {}
    F2["r1"] = 0.28504 - 0.01508 / 2
    F2["r2"] = 0.28504 + 0.01508 / 2
    F2["z1"] = -0.125 / 2
    F2["z2"] = +0.125 / 2
    F2["nt"] = 81.7
    F2["at"] = 81.7 * 1629
    F2["fil"] = FilamentCoil(F2, 3, 7)

    F3 = {}
    F3["r1"] = 0.33734 - 0.08936 / 2
    F3["r2"] = 0.33734 + 0.08936 / 2
    F3["z1"] = -0.1615 / 2
    F3["z2"] = +0.1615 / 2
    F3["nt"] = 607.7
    F3["at"] = 607.7 * 1629
    F3["fil"] = FilamentCoil(F3, 10, 15)

    FC = {}
    FC["fil"] = np.concatenate((F1["fil"], F2["fil"], F3["fil"]))
    FC["nt"] = F1["nt"] + F2["nt"] + F3["nt"]
    FC["at"] = F1["at"] + F2["at"] + F3["at"]

    print("Inductances of LC by approximations")
    print(LMaxwell(LC), LLyle4(LC), LLyle6(LC), LLyle6A(LC))
    print("Inductances of CC by approximations")
    print(LMaxwell(CC), LLyle4(CC), LLyle6(CC), LLyle6A(CC))

    print("Self-Inductance of FC by approximations and mutuals")
    LF1 = LLyle6(F1)
    LF2 = LLyle6(F2)
    LF3 = LLyle6(F3)
    MF12 = TotalM0(F1, F2)
    MF13 = TotalM0(F1, F3)
    MF23 = TotalM0(F2, F3)
    LFC = LF1 + LF2 + LF3 + 2 * MF12 + 2 * MF13 + 2 * MF23
    print("LFC %f H" % (LFC))

    MFL = TotalM0(FC, LC)
    MFC = TotalM0(FC, CC)
    FFL = TotalFz(FC, LC)

    print("Mutual of F & L: %f mH" % (MFL * 1000))
    print("Mutual of F & C: %f H" % (MFC))
    print("Force between F & L in %6.2f kg" % (FFL / 9.81))


if __name__ == "__main__":
    run_ldx_tests()
