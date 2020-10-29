from sympy import *
import numpy as np

init_printing(use_unicode=True)
a3, a4, a5, x, y = symbols('a3 a4 a5 x y', real=True)
f_radial = a3*cos(a4*sqrt(x**2+y**2)+a5)
print("Partial Differientiation with a3", diff(f_radial,a3))
print("Partial Differientiation with a4", diff(f_radial,a4))
print("Partial Differientiation with a5", diff(f_radial,a5))


# Partial Differientiation with a3 cos(a4*sqrt(x**2 + y**2) + a5)
# Partial Differientiation with a4 -a3*sqrt(x**2 + y**2)*sin(a4*sqrt(x**2 + y**2) + a5)
# # Partial Differientiation with a5 -a3*sin(a4*sqrt(x**2 + y**2) + a5)
#
# a3 0.0007185921996176913
# a4 1.7891324575070937
# a5 1.0047564636509498
#
# a3 0.0070634974867294625
# a4 -0.18980829462496857
# a5 0.08866446848504962