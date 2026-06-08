# check a subpackage from a package
from scipy import optimize
print(optimize.__doc__)

# check a module inside a subpackage
import scipy.optimize._lbfgsb_py as m
print(m.__file__)
print(m._minimize_lbfgsb.__doc__)


# check default parameter number for some function inside a module
import inspect
from scipy.optimize import _lbfgsb_py
print(inspect.signature(optimize._lbfgsb_py._minimize_lbfgsb))