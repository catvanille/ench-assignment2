import numpy as np
from dataclasses import dataclass

R = 8.314  # J/mol-K
class Fluid:
    """
    Class for Fluid properties. 
    """
    def __init__(self, name: str, mw: float, Tc: float, Pc: float, w: float, A: float, B: float, C: float):
        self.name = name
        self.mw = mw # Molecular weight in g/mol
        self.Tc = Tc # Critical temperature in K (Kelvin)
        self.Pc = Pc # Critical pressure in Pa
        self.w = w # Acentric factor
        self.A = A # Antoine coefficient A
        self.B = B  # Antoine coefficient B
        self.C = C  # Antoine coefficient C

    def calc_pvap(self, T: float, P: float) -> float:
        """
        Calculates vapor pressure of a given Fluid using the Peng-Robinson equation of state (PR-EoS).
        Args:
        T (float): Temperature in Kelvin.
        P (float): Pressure in Pa, guessed
        Returns:
        float: Compressibility factor.
        """
        Tc = self.Tc # K
        Pc = self.Pc # Pa
        w = self.w # Acentric factor
        RT = R * T
        # Reduced properties
        Pr = P / Pc
        Tr = T / Tc

        # PR-EoS parameters
        a0 = (0.45724 * (R * Tc) ** 2) / Pc
        b = 0.0778 * (R * Tc) / Pc
        kappa = 0.37464 + 1.54226 * w - 0.26992 * w ** 2
        alpha = (1 + kappa * (1 - np.sqrt(Tr))) ** 2
        a = a0 * alpha
        for i in range(100):
          # Parameters for Cubic EOS in terms of compressibility
          A1 = a * P / RT**2
          B1 = b * P / RT
          # Coefficients for the cubic EOS
          a1 = 1
          a2 = B1 - 1
          a3 = A1 - 3 * B1**2 - 2 * B1
          a4 = -A1 * B1 + B1**2 + B1**3
          aa = np.array([a1, a2, a3, a4])

          # Solve for compressibility factors
          zz = np.roots(aa)
          zz = zz[np.isreal(zz)==True].real  # real roots only
          Z = np.sort(zz)  # sort to identify liquid and vapor, ascending order

          # Fugacity coefficient calculation
          phi = []
          for j in range(3):
            term1 = Z[j] - 1
            term2 = np.log((Z[j] - B1))
            term3 = A1 / (2 * B1 * np.sqrt(2))
            term4 = Z[j] + 2.414 * B1
            term5 = Z[j] - 0.414 * B1
            phi.append(term1 - term2 - term3 * np.log(term4 / term5))

          phiv = np.exp(phi[2])
          phil = np.exp(phi[0])
          # Vapor fugacity
          Fug_v = P * phiv

          # Liquid fugacity
          Fug_l = P * phil

          # Convergence criterion
          error = abs(1 - Fug_v / Fug_l)
          if error < 1e-6:
            Pvap = P
            break

          # Update pressure guess
          P = P * (phil / phiv)

        return Pvap/100000, phiv

    def antoine(self, T: float) -> float:
        """
        Calculates vapor pressure of a given Fluid using the Antoine equation.
        Args:
        T (float): Temperature in Kelvin.
        Returns:
        float: Vapor pressure in bar.
        """
        A = self.A
        B = self.B
        C = self.C
        Psat = np.exp(A - B / (T + C))
        return Psat
class Water(Fluid):
    def __init__(self):
        super().__init__('Water', 18.015, 647.3, 220.48e5, 0.344, 11.6834, 3816.44, -46.13)

class Methanol(Fluid):
    def __init__(self):
        super().__init__('Methanol', 32.042, 512.6, 80.96e5, 0.559, 11.9673, 3626.55, -34.29)

def g_excess(T: float, xa: float, Aab: float, Aba: float) -> float:
  """
  Calculates excess Gibbs free energy of mixing for a binary mixture.
  Args:
  T (float): Temperature in Kelvin.
  xa (float): Mole fraction of component A.
  Aab (float): Interaction parameter for A-B pair.
  Aba (float): Interaction parameter for B-A pair.
  """
  return -R * T * (xa * np.log(xa + (1 - xa) * Aab) + (1 - xa) * np.log((1-xa) + Aba*xa))

def gamma(T: float, xa: float, Aab: float, Aba) -> float:
  """
  Calculates activity coefficient for a binary mixture.
  Args:
  T (float): Temperature in Kelvin.
  xa (float): Mole fraction of component A.
  Aab (float): Interaction parameter for A-B pair.
  Aba (float): Interaction parameter for B-A pair.
  """
  ln_gamma_A = -np.log(xa+ (1-xa)*Aab) + (1-xa)*(Aab/(xa + (1-xa)*Aab) - Aba/(1-xa + Aba*xa))
  return np.exp(ln_gamma_A)
