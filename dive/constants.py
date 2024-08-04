from math import pi

# Fundamental constants (CODATA 2018)
NA = 6.02214076e23       # Avogadro constant, mol^-1
muB = 9.2740100783e-24   # Bohr magneton, J T^-1
mu0 = 1.25663706212e-6   # magnetic constant, N A^-2 = T^2 m^3 J^-1
h = 6.62607015e-34       # Planck constant, J Hz^-1
ge = 2.00231930436256    # free-electron g factor
hbar = h/2/pi            # reduced Planck constant, J (rad s^-1)^-1

# Dipolar constant
D = (mu0/4/pi)*(muB*ge)**2/hbar  # m^3 rad s^-1