#   Authors: Christian Komo, Niels Bidault
from sympy import symbols, Matrix
import sympy as sp
import numpy as np
from scipy import interpolate
from scipy import optimize
import math
import torch

#IMPORTANT NOTES:
    #  by default every beam type is an electron beam type

    #  NOTE: default fringe fields for now is noted as [[x list], [y list]],
    #  ASSUMES that the measurement begins at 0
    #  ex. [[0.01,0.02,0.03,0.95,1],[1.6,0.7,0.2,0.01,1]]

#      getSymbolicMatrice() must use all sympy methods and functions, NOT numpy


class lattice:
    E = 45  # Kinetic energy (MeV/c^2)
    ## this should be passed from ebeam
    E0 = 0.51099 # Electron rest energy (MeV/c^2)
    Q = 1.60217663e-19  # (C)
    M = 9.1093837e-31  # (kg)
    C = 299792458  # Celerity (m/s)
    f = 2856 * (10 ** 6)  # RF frequency (Hz)

    M_AMU = 1.66053906892E-27  # Atomic mass unit (kg)
    k_MeV = 1e-6 / Q  # Conversion factor (MeV / J)
    m_p = 1.67262192595e-27  # Proton Mass (kg)

    #  Dictionary of predefined particle properties [Mass (kg), Charge (C), Rest Energy (MeV)]
    PARTICLES = {"electron": [M, Q, (M * C ** 2) * k_MeV],
                      "proton": [m_p, Q, (m_p * C ** 2) * k_MeV]}

    # Relativistic gamma and beta factors, calculated based on current kinetic energy and rest energy
    gamma = torch.tensor(1 + (E / E0))
    beta = torch.sqrt(1 - (1 / (gamma ** 2)))

    unitsF = 10 ** 6 # Units factor used for conversions from (keV) to (ns)
    color = 'none'  #Color of beamline element when graphed

    def __init__(self, length, fringeType = None):
        '''
        parent class for accelerator beamline segment object

        Parameters
        ----------
        length : float
            Sets the physical length of the beamline element in meters.
        fringeType : 
        '''
        self.fringeType = fringeType  # Each segment has no magnetic fringe by default
        self.startPos = None
        self.endPos = None

        # Validate and set the length of the beamline segment
        if not length <= 0:
            self.length = length
        else:
            raise ValueError("Invalid Parameter: Please enter a positive length parameter")

    def setE(self, E):
        '''
        Sets the kinetic energy (E) of the particle and updates dependent relativistic factors.

        Parameters
        ----------
        E : float
            New kinetic energy value (MeV/c^2).
        '''
        self.E = E
        self.gamma = (1 + (self.E/self.E0))
        self.beta = np.sqrt(1-(1/(self.gamma**2)))

    def setMQE(self, mass, charge, restE):
        '''
        Sets the mass, charge, and rest energy of the particle, and updates
        dependent relativistic factors.

        Parameters
        ----------
        mass : float
            The new mass of the particle in kg.
        charge : float
            The new charge of the particle in Coulombs.
        restE : float
            The new rest energy of the particle in MeV.
        '''
        self.M = mass
        self.Q = charge
        self.E0 = restE
        self.gamma = (1 + (self.E/self.E0))
        self.beta = torch.sqrt(1-(1/(self.gamma**2)))

    def changeBeamType(self, particleType, kineticE, beamSegments = None):
        '''
        Changes the type of particle being simulated (e.g., "electron", "proton", or isotope).
        Updates the mass, charge, rest energy, and kinetic energy for the current segment
        and optionally for a list of other beamline segments.

        Parameters
        ----------
        particleType : str
            The type of particle. Either a predefined string ("electron", "proton")
            or an isotope string in the format "(isotope number),(ion charge)" (e.g., "12,5" for C12 5+).
        kineticE : float
            The kinetic energy for the new particle type in MeV/c^2.
        beamSegments : list[lattice], optional
            A list of other beamline segment objects whose particle properties
            should also be updated.

        Returns
        -------
        list[lattice] or None
            If `beamSegments` is provided, returns the updated list of beam segments.
            Otherwise, returns None.

        Raises
        ------
        TypeError
            If the `particleType` is not recognized or in an invalid isotope format.
        '''
        try:
            particleData = self.PARTICLES[particleType]
            self.setMQE(particleData[0], particleData[1], particleData[2])
            self.setE(kineticE)
            if beamSegments is not None:
                for seg in beamSegments:
                    seg.setMQE(particleData[0], particleData[1], particleData[2])
                    seg.setE(kineticE)
                return beamSegments
        except KeyError:
            #  Try look for isotope particle format, format = "(isotope number),(ion charge)"
            #  ex. C12+5 (carbon 12, 5+ charge) = "12,5"
            try:
                isotopeData = particleType.split(",")
                A = int(isotopeData[0])
                Z = int(isotopeData[1])
                m_i = A * self.M_AMU
                q_i = Z * self.Q
                meV = (m_i * self.C ** 2) * self.k_MeV

                self.setMQE(m_i, q_i, meV)
                self.setE(kineticE)
                if beamSegments is not None:
                    for seg in beamSegments:
                        seg.setMQE(m_i, q_i, meV)
                        seg.setE(kineticE)
                    return beamSegments
            except:
                raise TypeError("Invalid particle beam type or isotope")

    def getSymbolicMatrice(self, **kwargs):
        '''
        Abstract method to be implemented by child classes. This method should
        return the symbolic transfer matrix for the beamline element.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters specific to the child class's matrix calculation.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the child class.
        '''
        raise NotImplementedError("getSymbolicMatrice not defined in child class")

    #unfortuately, cannot check whether the kwargs exist in the segments function or not already
    def useMatrice(self, val, **kwargs):
        ''''
        Simulates the movement of given particles through its child segment by
        applying the segment's transfer matrix with numeric parameters.

        Parameters
        ----------
        val tensor V: torch tensor, with the shape N*6, N is the number of particles, 6 is the phase space coordinates (x, x', y, y', z, z')
            A 2D NumPy array representing the particle states. Each row is a particle,
            and columns correspond to phase space coordinates (e.g., [x, x', y, y', z, z']).
        **kwargs : dict
            Other segment-specific numeric parameters (e.g., `length`, `current`)
            that might override the segment's default properties for this specific simulation.

        Returns
        -------
        list
            A 2D list with length N, where each inner list with size 6 represents the transformed state of a particle
            after passing through the segment.

        '''
        # transformation matrix M with the size 6*6, where the new particle P'=MP
        mat = self.getSymbolicMatrice(numeric = True, **kwargs)
        # transform into torch tensor
        mat_tensor = torch.tensor(np.array(mat).astype(np.float64),
                                  dtype=val.dtype,
                                  device=val.device)

        # parallelized matrix multiplication
        new_val_tensor = torch.matmul(val, mat_tensor.T)
        new_val_tensor = new_val_tensor.cpu().numpy().tolist()
        return new_val_tensor

class driftLattice(lattice):
    color = "white"
    def __init__(self, length: float):
        '''
        Represents a drift space (empty section) in the beamline.

        Parameters
        ----------
        length : float
            The length of the drift segment in meters.
        '''
        super().__init__(length)

    # note: unlike old usematrice, this func doesnt check for negative/zero parameter numbers,
    # Nor if length is actually the dtype numeric specifies
    # implement both useMatrice and symbolic matrice in this, have to delete both later
    def getSymbolicMatrice(self, numeric = False, length = None):
        '''
        Returns the 6x6 transfer matrix for a drift space.

        Parameters
        ----------
        numeric : bool, optional
            If True, returns a numerical matrix. If False, returns a symbolic matrix.
        length : float or str, optional
            If `numeric` is True, this should be a float representing the length. If false, symbolic string length
            If None, uses segment's length.

        Returns
        -------
        sympy.Matrix or np.ndarray
            The 6x6 transfer matrix for the drift segment.
        '''
        l = None
        if length is None:
            l = self.length
        else:
            if numeric: l = length  # length should be number
            else: l = symbols(length, real = True)  # length should be string

        M56 = -(l * self.f / (self.C * self.beta * self.gamma * (self.gamma + 1)))
        mat = Matrix([[1, l, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, l, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, M56],
                      [0, 0, 0, 0, 0, 1]])
        return torch.tensor(mat,dtype=torch.float32)

    def __str__(self):
        '''
        Returns a string representation of the drift lattice segment.

        Returns
        -------
        str
            A descriptive string
        '''
        return f"Drift beamline segment {self.length} m long"


class qpfLattice(lattice):
    #color = "cornflowerblue"
    color = "blue"
    G = 2.694  #  Quadruple focusing strength (T/A/m)
    def __init__(self, current: float, length: float = 0.0889, fringeType = 'decay'):
        '''
        Represents a quadrupole focusing magnet. This magnet focuses in the x plane
        and defocuses in the y plane

        Parameters
        ----------
        current : float
            The current supplied to the quadrupole in Amps.
        length : float, optional
            The effective length of the quadrupole magnet in meters.
        fringeType :

        '''
        super().__init__(length, fringeType)
        self.current = current #  Amps

    def getSymbolicMatrice(self, numeric = False, length = None, current = None):
        '''
        Returns the 6x6 transfer matrix for a quadrupole focusing magnet.

        Parameters
        ----------
        numeric : bool, optional
            If True, returns a numerical matrix. If False, returns a symbolic matrix.
        length : float or str, optional
            If `numeric` is True, this is the numerical length, if False, string symbolic length.
            If None, uses segment's length.
        current : float or str, optional
            If `numeric` is True, this is the numerical current, if False, string symbolic current.
            If None, uses segment's current.

        Returns
        -------
        sympy.Matrix or np.ndarray
            The 6x6 transfer matrix for the quadrupole focusing magnet.
        '''
        l = None
        I = None

        if length is None:
            l = self.length
        else:
            if numeric: l = length  # length should be number
            else: l = symbols(length, real = True)  # length should be string

        if current is None:
            I = self.current
        else:
            if numeric: I = current  # current should be number
            else: I = symbols(current, real = True)  # current should be string

        self.k = sp.Abs((self.Q*self.G*I)/(self.M*self.C*self.beta*self.gamma))
        self.theta = sp.sqrt(self.k)*l

        M11 = sp.cos(self.theta)
        M21 = (-(sp.sqrt(self.k)))*sp.sin(self.theta)
        M22 = sp.cos(self.theta)
        M33 = sp.cosh(self.theta)
        M43 = sp.sqrt(self.k)*sp.sinh(self.theta)
        M44 = sp.cosh(self.theta)
        M56 = -(l * self.f / (self.C * self.beta * self.gamma * (self.gamma + 1)))

        if I == 0:
            M12 = l
            M34 = l
        else:
            M34 = sp.sinh(self.theta)*(1/sp.sqrt(self.k))
            M12 = sp.sin(self.theta)*(1/sp.sqrt(self.k))

        mat =  Matrix([[M11, M12, 0, 0, 0, 0],
                        [M21, M22, 0, 0, 0, 0],
                        [0, 0, M33, M34, 0, 0],
                        [0, 0, M43, M44, 0, 0],
                        [0, 0, 0, 0, 1, M56],
                        [0, 0, 0, 0, 0, 1]])

        return torch.tensor(mat,dtype=torch.float32)

    def __str__(self):
        '''
        Returns a string representation of the quadrupole focusing lattice segment.

        Returns
        -------
        str
            A descriptive string
        '''
        return f"QPF beamline segment {self.length} m long and a current of {self.current} amps"


class qpdLattice(lattice):
    #color = "lightcoral"
    color = "red"
    G = 2.694  # Quadruple focusing strength (T/A/m)
    def __init__(self, current: float, length: float = 0.0889, fringeType = 'decay'):
        '''
        Represents a quadrupole defocusing magnet. This magnet defocuses in the x plane
        and focuses in the y plane

        Parameters
        ----------
        current : float
            The current supplied to the quadrupole in Amps.
        length : float, optional
            The effective length of the quadrupole magnet in meters.
        fringeType :
        '''
        super().__init__(length, fringeType)
        self.current = current # Amps

    def getSymbolicMatrice(self, numeric = False, length = None, current = None):
        '''
        Returns the 6x6 transfer matrix for a quadrupole defocusing magnet.

        Parameters
        ----------
        numeric : bool, optional
            If True, returns a numerical matrix. If False, returns a symbolic matrix.
        length : float or str, optional
            If `numeric` is True, this is the numerical length, if False, string symbolic length.
            If None, uses segment's length.
        current : float or str, optional
            If `numeric` is True, this is the numerical current, if False, string symbolic current.
            If None, uses segment's current.

        Returns
        -------
        sympy.Matrix or np.ndarray
            The 6x6 transfer matrix for the quadrupole defocusing magnet.
        '''
        l = None
        I = None

        if length is None:
            l = self.length
        else:
            if numeric: l = length  # length should be number
            else: l = symbols(length, real = True)  # length should be string

        if current is None:
            I = self.current
        else:
            if numeric: I = current  # current should be number
            else: I = symbols(current, real = True)  # current should be string

        self.k = sp.Abs((self.Q*self.G*I)/(self.M*self.C*self.beta*self.gamma))
        self.theta = sp.sqrt(self.k)*l

        M11 = sp.cosh(self.theta)
        M21 = sp.sqrt(self.k)*sp.sinh(self.theta)
        M22 = sp.cosh(self.theta)
        M33 = sp.cos(self.theta)
        M43 = (-(sp.sqrt(self.k)))*sp.sin(self.theta)
        M44 = sp.cos(self.theta)
        M56 = -l * self.f / (self.C * self.beta * self.gamma * (self.gamma + 1))

        if I == 0:
            M12 = l
            M34 = l
        else:
            M34 = sp.sin(self.theta)*(1/sp.sqrt(self.k))
            M12 = sp.sinh(self.theta)*(1/sp.sqrt(self.k))

        mat = Matrix([[M11, M12, 0, 0, 0, 0],
                        [M21, M22, 0, 0, 0, 0],
                        [0, 0, M33, M34, 0, 0],
                        [0, 0, M43, M44, 0, 0],
                        [0, 0, 0, 0, 1, M56],
                        [0, 0, 0, 0, 0, 1]])

        return torch.tensor(mat,dtype=torch.float32)

    def __str__(self):
        '''
        Returns a string representation of the quadrupole defocusing lattice segment.

        Returns
        -------
        str
            A descriptive string
        '''
        return f"QPD beamline segment {self.length} m long and a current of {self.current} amps"

class dipole(lattice):
    #color = "forestgreen"
    color = "green"
    def __init__(self, length: float = 0.0889, angle: float = 1.5, fringeType = 'decay'):
        '''
        Represents a dipole bending magnet, which bends the beam horizontally.

        Parameters
        ----------
        length : float, optional
            The effective length of the dipole magnet in meters
        angle : float, optional
            The bending angle of the dipole magnet in degrees.
        fringeType :
        '''
        super().__init__(length, fringeType)
        self.angle = angle  # degrees

    def getSymbolicMatrice(self, numeric = False, length = None, angle = None):
        '''
        Returns the 6x6 transfer matrix for a horizontal dipole bending magnet.

        Parameters
        ----------
        numeric : bool, optional
            If True, returns a numerical matrix. If False, returns a symbolic matrix.
        length : float or str, optional
            If `numeric` is True, this is the numerical length, i
            f False, string symbolic length.
            If None, uses segment's length.
        angle : float or str, optional
            If `numeric` is True, this is the numerical bending angle in degrees,
            if False, string symbolic angle.
            If None, uses segment's angle.

        Returns
        -------
        sympy.Matrix or np.ndarray
            The 6x6 transfer matrix for the dipole magnet.
        '''
        l = None
        a = None

        if length is None:
            l = self.length
        else:
            if numeric: l = length  # length should be number
            else: l = symbols(length, real = True)  # length should be string

        if angle is None:
            a = self.angle
        else:
            if numeric: a = angle  # angle should be number
            else: a = symbols(angle, real = True)  # angle should be string

        by = (self.M*self.C*self.beta*self.gamma / self.Q) * (a * sp.pi / 180 / self.length)
        rho = self.M*self.C*self.beta*self.gamma / (self.Q * by)
        theta = l / rho
        C = sp.cos(theta)
        S = sp.sin(theta)

        M16 = rho * (1 - C) * (self.gamma / (self.gamma + 1))
        M26 = S * (self.gamma / (self.gamma + 1))
        M51 = -self.f * S / (self.beta * self.C)
        M52 = -self.f * rho * (1 - C) / (self.beta * self.C)
        M56 = -self.f * (l - rho * S) / (self.C * self.beta * self.gamma * (self.gamma + 1))

        mat = Matrix([[C, rho * S, 0, 0, 0, M16],
                      [-S / rho, C, 0, 0, 0, M26],
                      [0, 0, 1, l, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [M51, M52, 0, 0, 1, M56],
                      [0, 0, 0, 0, 0, 1]])

        return torch.tensor(mat,dtype=torch.float32)

    def __str__(self):
        '''
        Returns a string representation of the dipole magnet segment.

        Returns
        -------
        str
            A descriptive string
        '''
        return f"Horizontal dipole magnet segment {self.length} m long (curvature) with an angle of {self.angle} degrees"

class dipole_wedge(lattice):
    color = "lightgreen"
    def __init__(self, length, angle: float = 1, dipole_length: float = 0.0889, dipole_angle: float = 1.5,
                 pole_gap = 0.014478, enge_fct = 0, fringeType = 'decay'):
        '''
        Represents a dipole magnet with wedge-shaped pole faces at its entrance and/or exit,
        which introduces a vertical focusing or defocusing effect. This class models the
        effect of these wedge angles, often found in spectrometer dipoles.

        Parameters
        ----------
        length : float
            The effective length of the wedge magnet segment in meters.
        angle : float, optional
            The wedge angle (half-angle) of the pole face in degrees. This angle
            contributes to the vertical focusing/defocusing.
        dipole_length : float, optional
            The physical length of the main dipole field region in meters.
            This is used to calculate the magnetic field strength based on the dipole_angle.
        dipole_angle : float, optional
            The total bending angle of the main dipole field region in degrees.
            Used to calculate the magnetic field strength.
        pole_gap : float, optional
            The gap between the dipole poles in meters. Used in the fringe field calculation.
        enge_fct : float, optional
            Placeholder for Enge function parameter, related to fringe field modeling.
        fringeType :
        '''
        super().__init__(length, fringeType)
        self.angle = angle
        self.dipole_length = dipole_length
        self.dipole_angle = dipole_angle
        #  self.pole_gap = 0.0127
        self.pole_gap = pole_gap

    def getSymbolicMatrice(self, numeric = False, length = None, angle = None):
        '''
        Returns the 6x6 transfer matrix for a dipole magnet with wedge pole faces.
        This matrix includes the effects of the wedge angle on transverse focusing.

        Parameters
        ----------
        numeric : bool, optional
            If True, returns a numerical matrix. If False, returns a symbolic matrix.
        length : float or str, optional
            If `numeric` is True, this is the numerical effective length of the wedge segment,
            if False, string symbolic length.
            If None, uses segment's length.
        angle : float or str, optional
            If `numeric` is True, this is the numerical wedge angle in degrees.
            if False, string symbolic angle.
            If None, uses segment's angle.

        Returns
        -------
        sympy.Matrix or np.ndarray
            The 6x6 transfer matrix for the wedge dipole magnet.
        '''
        l = None
        a = None

        if length is None:
            l = self.length
        else:
            if numeric: l = length  # length should be number
            else: l = symbols(length, real = True)  # length should be string

        if angle is None:
            a = self.angle
        else:
            if numeric: a = angle  # angle should be number
            else: a = symbols(angle, real = True)  # angle should be string

        dipole_angle = self.dipole_angle
        dipole_length = self.dipole_length

        # Hard edge model for the wedge magnets
        By = (self.M*self.C*self.beta*self.gamma / self.Q) * (dipole_angle * sp.pi / 180 / dipole_length)
        R = self.M*self.C*self.beta*self.gamma / (self.Q * By)
        eta = (a * sp.pi / 180) * l / self.length
        Tx = sp.tan(eta)

        '''
        https://www.slac.stanford.edu/cgi-bin/getdoc/slac-r-075.pdf page 49.
        
        phi = K * g * h  g * (1 + sin^2 a) / cos a. 
        
        Fringe field contribution:
        K = int( By(z) * (By_max - By(z)) / (g * By_max^2), dz ) 
        h = 1 / rho_0, dipole radius
        g, pole gap
        
        hard-edge model, phi = 0
        '''
        z = sp.symbols("z", real=True)
        g = self.pole_gap
        le = self.length
        # Triangle fringe field model: B(z) = Bmax * (z / le)
        Bz = By * (z / le)
        K_expr = sp.integrate((Bz * (By - Bz)) / (g * By ** 2), (z, 0, le))
        K_simplified = sp.simplify(K_expr)
        h = 1 / R
        phi = sp.simplify(K_simplified * g * h * (1 + sp.sin(eta) ** 2) / sp.cos(eta))
        Ty = sp.tan(eta - phi)
        M56 = -self.f * (l / (self.C * self.beta * self.gamma * (self.gamma + 1)))


        mat = Matrix([[1, 0, 0, 0, 0, 0],
                      [Tx / R, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, -Ty / R, 1, 0, 0],
                      [0, 0, 0, 0, 1, M56],
                      [0, 0, 0, 0, 0, 1]])

        return torch.tensor(mat,dtype=torch.float32)

    def __str__(self):
        '''
        Returns a string representation of the horizontal wedge dipole magnet segment.

        Returns
        -------
        str
            A descriptive string
        '''
        return f"Horizontal wedge dipole magnet segment {self.length} m long (curvature) with an angle of {self.angle} degrees"



#NOTE: getSymbolicMatrice() must use all sympy methods and functions, NOT numpy
class Beamline:
    class fringeField(lattice):
        B = 0 #  Teslas
        color = 'brown'

        def __init__(self, length, fieldStrength, current = 0):
            super().__init__(length)
            self.B = fieldStrength
            # self.current = current

        #temporarily use drift matrice for testing
        def getSymbolicMatrice(self, numeric = False, length = None, current = None):
            l = None
            I = None
            if length is None:
                l = self.length
            else:
                if numeric: l = length  # length should be number
                else: l = symbols(length, real = True)  # length should be string

            # if current is None:
            #     I = self.current
            # else:
            #     if numeric: I = current  # current should be number
            #     else: I = symbols(current, real = True)  # current should be string

            # self.k = sp.Abs((self.Q*self.G*I * self.fringeDecay )/(self.M*self.C*self.beta*self.gamma))
            # self.theta = sp.sqrt(self.k)*l

            # M11 = sp.cos(self.theta)
            # M21 = (-(sp.sqrt(self.k)))*sp.sin(self.theta)
            # M22 = sp.cos(self.theta)
            # M33 = sp.cosh(self.theta)
            # M43 = sp.sqrt(self.k)*sp.sinh(self.theta)
            # M44 = sp.cosh(self.theta)
            # M56 = -(l * self.f / (self.C * self.beta * self.gamma * (self.gamma + 1)))

            # if I == 0:
            #     M12 = l
            #     M34 = l
            # else:
            #     M34 = sp.sinh(self.theta)*(1/sp.sqrt(self.k))
            #     M12 = sp.sin(self.theta)*(1/sp.sqrt(self.k))

            # mat =  Matrix([[M11, M12, 0, 0, 0, 0],
            #                 [M21, M22, 0, 0, 0, 0],
            #                 [0, 0, M33, M34, 0, 0],
            #                 [0, 0, M43, M44, 0, 0],
            #                 [0, 0, 0, 0, 1, M56],
            #                 [0, 0, 0, 0, 0, 1]])

            # return mat

            M56 = (l * self.f / (self.C * self.beta * self.gamma * (self.gamma + 1)))
            mat = Matrix([[1, l, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 1, l, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, M56],
                        [0, 0, 0, 0, 0, 1]])

            return torch.tensor(mat,dtype=torch.float32)


        def __str__(self) -> str:
            return f"Fringe field segment {self.length} m long with a magnetic field of {self.B} teslas"

    def csvToBeamline(self, csv):
        with open(csv, newline='') as file:
            reader = csv.reader(file)


    def __init__(self, line = []):
        self.ORIGINFACTOR = torch.tensor(0.99)

        a = {'first order decay': (self._frontModel, self._endModel)} # must work on later
        self.FRINGEDELTAZ = torch.tensor(0.01)

        self.beamline = line
        self.totalLen = torch.zeros(1)
        self.defineEndFrontPos()

    #  Invariant, call at end of all functions
    def defineEndFrontPos(self):
        """
        Vectorized calculation of beamline positions using cumulative sums.
        Replaces the Python 'for' loop with a single PyTorch pass.
        """
        lengths = torch.tensor([seg.length for seg in self.beamline])


        end_positions = torch.cumsum(lengths, dim=0)


        start_positions = torch.cat([torch.tensor([0.0]), end_positions[:-1]])


        for i, seg in enumerate(self.beamline):
            seg.startPos = start_positions[i].item()
            seg.endPos = end_positions[i].item()


        self.totalLen = end_positions[-1].item() if len(end_positions) > 0 else 0

    def findSegmentAtPos(self, pos):
        for i in range(len(self.beamline)):
            seg = self.beamline[i]
            if (pos >= seg.startPos and pos <= seg.endPos):
                return i
        return -1

     #  may use other interpolation methods (cubic, spline, etc)
     # x = 0 = start of end of segment
    def interpolateData(self, xData, yData, interval):
        rbf = interpolate.Rbf(xData, yData)
        totalLen = xData[-1] - xData[0]
        xNew = torch.linspace(xData[0], xData[-1], math.ceil(totalLen/interval) + 1)
        yNew = rbf(xNew)
        return xNew, yNew

    # def _model(self, x, B0, a, origin):
    #     return B0 * (1 - ((x-origin)/a)**2) * (np.exp(-(((x-origin)/a)**2)))

    # def formulaFit(self, xData, yData, pos):
    #     endParams, _ = optimize.curve_fit(self._model, xData, yData, p0= [1,1, pos])
    #     return endParams

    def _testModeOrder2end(self, x, origin, B0, a1, a2):
        return B0/(1+torch.exp((a1*(x - origin)) + (a2*(x-origin)**2)))

    def testFrontFit(self, xData, yData, pos):
        endParams, _ = optimize.curve_fit(self._testModeOrder2front, xData, yData, p0= [pos, 1, 1, 1], maxfev=50000)
        return endParams

    def testendFit(self, xData, yData, pos):
        endParams, _ = optimize.curve_fit(self._testModeOrder2end, xData, yData, p0= [pos, 1, 1, 1], maxfev=50000)
        print(endParams)
        return endParams

    def _testModeOrder2front(self, x, origin, B0, a1, a2):
        return B0/(1+torch.exp((a1*(-x - origin)) + (a2*(-x-origin)**2)))


    def _endModel(self, x, origin, B0, strength):
        return (B0/(1+torch.exp((x-origin)*strength)))

    def _frontModel(self, x, origin, B0, strength):
        return (B0/(1+torch.exp((-x+origin)*strength)))

    def frontFit(self, xData, yData, pos):
        endParams, _ = optimize.curve_fit(self._frontModel, xData, yData, p0= [pos, 1, 1], maxfev=50000)
        return endParams

    def endFit(self, xData, yData, pos):
        endParams, _ = optimize.curve_fit(self._endModel, xData, yData, p0= [pos, 1, 1], maxfev=50000)
        return endParams

    '''
    ind: int
        The indice of the magnetic segment to create fringe
    '''
    # assumes the zlist doesnt start at 0, and that the first element is how far from the segment fringe measurement is
    def _addEnd(self, zList, magnetList, beamline, ind):
        #  Initialize variables, find total space available for plotting
        driftLen = 0
        ind2 = ind
        while (ind2 != 0 and isinstance(beamline[ind2 - 1], driftLattice)):
            driftLen = driftLen + beamline[ind2 - 1].length
            ind2 -= 1

        #  Create and add fringe fields to list based on input z and B values
        i = 1
        fringeTotalLen = 0
        zList.insert(0,0)
        while (i < len(zList) and fringeTotalLen <= driftLen):
            fringeLen = zList[i] - zList[i-1]
            fringeTotalLen += fringeLen
            if fringeTotalLen <= driftLen:
                fringeSeg = self.fringeField(fringeLen, magnetList[i-1])
                beamline.insert(ind, fringeSeg)
            i += 1

        #  Shorten/eliminate any drift segments overlapping with fringe fields already
        while (fringeTotalLen > 0 and isinstance(beamline[ind-1], driftLattice)):
            if (beamline[ind-1].length <= fringeTotalLen):
                fringeTotalLen -= beamline[ind-1].length
                beamline.pop(ind-1)
                ind -= 1
            else:
                beamline[ind-1].length -= fringeTotalLen
                fringeTotalLen -= fringeTotalLen

    # BEAMLINE OBJECT DOESNT CONTAIN THE BEAMLINE, ONLY TO PERFORM CALCULATIONS ON THE LINE

    # TEST: functiom ran on the class' beamline OBJECT, NOT beamline LIST

    #  Fringe fields can only exist overlapping drift segments for now
    #  assume all fringe fields overlap drift segments

    # issue: origin of equations may not match with beamline exactly if interval 
    #        doesnt match beamline interval
    def reconfigureLine(self, interval=None):
        """
        Reconfigures the beamline by discretizing drift spaces into fringe field segments.
        Uses PyTorch vectorized operations to calculate total magnetic field distribution.
        All logic is computed in tensors to avoid expensive Python loops.
        """
        if interval is None:
            interval = self.FRINGEDELTAZ

        # 1. Vectorized zLine generation using torch.linspace
        # Replacing: while i <= totalLen: zLine.append(i)
        num_points = int(torch.ceil(self.totalLen / interval)) + 1
        z_line = torch.linspace(0, self.totalLen, num_points, dtype=torch.float32)
        y_values = torch.zeros_like(z_line)

        # 2. Vectorized Field Superposition (End Model)
        # Replaces 'zeroTracker' while-loops with Boolean Masking
        for segment in self.beamline:
            if segment.fringeType == 'first order decay':
                b0 = 1.0 # Temporary constant
                strength = 1.0
                # Calculate the origin point for the decay model
                origin = segment.endPos - (torch.log((1 - self.ORIGINFACTOR) / self.ORIGINFACTOR) / strength)

                # Compute the entire field for all z points simultaneously
                y_field = b0 / (1 + torch.exp((z_line - origin) * strength))

                # Apply Mask: zero out fields before the segment end (Parallelized)
                y_field[z_line < segment.endPos] = 0
                y_values += y_field

        # 3. Vectorized Field Superposition (Front Model)
        for segment in self.beamline:
            if segment.fringeType == 'first order decay':
                b0 = 1.0
                strength = 5.0
                origin = segment.startPos + (torch.log((1 - self.ORIGINFACTOR) / self.ORIGINFACTOR) / strength)

                y_field = b0 / (1 + torch.exp((-z_line + origin) * strength))

                # Apply Mask: zero out fields after the segment start
                y_field[z_line > segment.startPos] = 0
                y_values += y_field

        # 4. Optimized Reconstruction (Building a new list instead of in-place insert)
        # Python list.insert() is O(N), repeatedly inserting makes it O(N^2)
        new_beamline = []
        for seg in self.beamline:
            if not isinstance(seg, driftLattice):
                new_beamline.append(seg)
            else:
                # Vectorized lookup of the drift's range within the z_line
                mask = (z_line >= seg.startPos) & (z_line <= seg.endPos)
                z_slice = z_line[mask]
                y_slice = y_values[mask]

                # Sub-divide the drift into fringeField micro-segments
                for j in range(1, len(z_slice)):
                    l_slice = z_slice[j] - z_slice[j-1]
                    # Creating segment objects
                    new_beamline.append(self.fringeField(l_slice.item(), y_slice[j].item()))

                # Handle floating point residual for the last segment of the drift
                remaining_l = seg.length - (z_slice[-1] - z_slice[0])
                if remaining_l > 1e-12:
                    new_beamline.append(self.fringeField(remaining_l.item(), y_slice[-1].item()))

        self.beamline = new_beamline
        self.defineEndFrontPos() # Refresh position metadata

        return z_line.numpy(), y_values.numpy()