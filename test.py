import numpy as np
import matplotlib.pyplot as plt

# Define constants
a = 175e-9  # base layer thickness in meters
p = 0  # layer thickness modulation value p
theta0 = 0 * np.pi / 180  # Angle of incidence in radiants
n0 = 1

# Wavelength range from 900 nm to 1700 nm
wavelengths = np.linspace(900e-9, 1700e-9, 1000)

# Refractive indices
n_Si = 3.575
#Source: 
# 1) C. Schinke, P. C. Peest, J. Schmidt, R. Brendel, K. Bothe, M. R. Vogt, I. Kröger, S. Winter, A. Schirmacher, S. Lim, H. T. Nguyen, D. MacDonald. Uncertainty analysis for the coefficient of band-to-band absorption of crystalline silicon. AIP Advances 5, 67168 (2015)
# 2) M. R. Vogt. Development of physical models for the simulation of optical properties of solar cell modules, PhD. Thesis (2015)
n_SiO2 = 1.45
#Source:
#1) I. H. Malitson. Interspecimen comparison of the refractive index of fused silica, J. Opt. Soc. Am. 55, 1205-1208 (1965)
#2) C. Z. Tan. Determination of refractive index of silica glass for infrared wavelengths by IR spectroscopy, J. Non-Cryst. Solids 223, 158-163 (1998)


# Define layer widths
layers = [(1 + np.sin(np.pi * p)) * (2/3) * a,          #L1(Si)
        (1 + np.sin(-0.25 * np.pi)) * (1/3) * a,        #L2(SiO2)
        (1 - np.sin(np.pi * p)) * (2/3) * a,            #L3(Si)
        (1 - np.sin(-0.25 * np.pi)) * (1/3) * a,        #L4(SiO2)
        (1 + np.sin(np.pi * p)) * (2/3) * a,            #L5(Si)
        (1 + np.sin(-0.125 * np.pi)) * (1/3) * a,       #L6(SiO2)
        (1 - np.sin(np.pi * p)) * (2/3) * a,            #L7(Si)
        (1 - np.sin(-0.125 * np.pi)) * (1/3) * a,   	#L8(SiO2)
        (1 + np.sin(np.pi * p)) * (2/3) * a,            #L9(Si)
        (1 + np.sin(0 * np.pi)) * (1/3) * a,            #L10(SiO2)
        (1 - np.sin(np.pi * p)) * (2/3) * a,            #L11(Si)
        (1 - np.sin(0 * np.pi)) * (1/3) * a,            #L12(SiO2)
        (1 + np.sin(np.pi * p)) * (2/3) * a,            #L13(Si)
        (1 + np.sin(0.125 * np.pi)) * (1/3) * a,        #L14(SiO2)
        (1 - np.sin(np.pi * p)) * (2/3) * a,            #L15(Si)
        (1 - np.sin(0.125 * np.pi)) * (1/3) * a,    	#L16(SiO2)
        (1 + np.sin(np.pi * p)) * (2/3) * a,            #L17(Si)
        (1 + np.sin(0.25 * np.pi)) * (1/3) * a,         #L18(SiO2)
        (1 - np.sin(np.pi * p)) * (2/3) * a,            #L19(Si)
        (1 - np.sin(0.25 * np.pi)) * (1/3) * a,     	#L20(SiO2)
        ]

# Create array with refractive indices of layers    
refractive_indices = [n_Si if i % 2 == 0 else n_SiO2 for i in range(20)]

# Calculate angle in each layer via Snell's law
def calc_theta(n0, theta0, n):
    return np.arcsin(n0 * np.sin(theta0) / n)

# Berechne die Interface-Matrix
def interface_matrix(n1, n2):
    I = np.array([[0.5 * (1 + n2 / n1), 0.5 * (1 - n2 / n1)],
                  [0.5 * (1 - n2 / n1), 0.5 * (1 + n2 / n1)]])
    return I

# Berechne die Propagation-Matrix
def propagation_matrix(d, n, theta, wavelength):
    k0 = 2 * np.pi / wavelength
    delta = k0 * n * d * np.cos(theta)
    P = np.array([[np.exp(-1j * delta), 0],
                  [0, np.exp(1j * delta)]])
    return P

# Transfermatrix Methode mit Einfallswinkel
def transfer_matrix(wavelength, layers, indices, theta0, n0):
    M = np.identity(2, dtype=complex)
    theta = theta0
    for i in range(len(layers)):
        n = indices[i]
        d = layers[i]
        next_n = indices[i+1] if i+1 < len(indices) else n0
        theta_next = calc_theta(n0, theta, next_n)
        P = propagation_matrix(d, n, theta, wavelength)
        I = interface_matrix(n, next_n)
        M = P @ I @ M
        theta = theta_next
    #add last interface here
    return M

# Reflexion und Transmission berechnen mit Einfallswinkel
def reflection_transmission(wavelengths, layers, indices, theta0, n0):
    R = np.zeros_like(wavelengths)
    T = np.zeros_like(wavelengths)
    for i, wl in enumerate(wavelengths):
        M = transfer_matrix(wl, layers, indices, theta0, n0)
        r = M[1,0] / M[0,0]  # Reflection coefficient
        t = 1 / M[0,0]       # Transmission coefficient
        R[i] = np.abs(r)**2  # Reflectance
        T[i] = np.abs(t)**2  # Transmittance
    return R, T

# Compute reflection and transmission spectra
R, T = reflection_transmission(wavelengths, layers, refractive_indices, theta0, n0)

# Plot the spectra
plt.figure(figsize=(10, 6))
plt.plot(wavelengths * 1e9, R, label='Reflection')
plt.plot(wavelengths * 1e9, T, label='Transmission')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity')
plt.title('Reflection and Transmission Spectra')
plt.legend()
plt.grid(True)
plt.show()