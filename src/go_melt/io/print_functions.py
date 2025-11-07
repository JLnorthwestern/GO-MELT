import numpy as np
import sys


def printLevelMaxMin(Ls, Lnames):
    """
    Print the min and max temperatures for each level and check for invalid values.
    Terminates the program if any temperature is NaN, too low, or unreasonably high.
    """
    print("Temps:", end=" ")
    flag = False

    for i in range(1, len(Ls)):
        T = Ls[i]["T0"]
        Lmin = np.min(T)
        Lmax = np.max(T)

        # Use vectorized checks for invalid values
        if not np.isfinite(Lmax) or Lmax <= 0 or Lmax > 1e5:
            print(
                f"\nTerminating program: Lmax for {Lnames[i-1]} is NaN, 0, or invalid."
            )
            flag = True

        if not np.isfinite(Lmin) or Lmin <= 0 or Lmin > 1e5:
            print(
                f"\nTerminating program: Lmin for {Lnames[i-1]} is NaN, 0, or invalid."
            )
            flag = True

        print(f"{Lnames[i-1]}: [{Lmin:.2f}, {Lmax:.2f}]", end=" ")

    if flag:
        sys.exit(1)
    print("")
