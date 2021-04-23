import numpy as np
import matplotlib.pyplot as plt
import mpmath as mpm
import cmath
import scipy.linalg

#Discretize figures
#D: Height, L: Length


def discretize_geometry(D, L):
	#Refine this to match every geometry
	Nside = int(D*10); Nbott = int(L*10)
	if L < 0.5:
		Nbott = 5
	#Total number of points
	N = Nside + Nbott + Nside

	dx = L/Nbott; dy = D/Nside

	#Discretizing x-points
	#Defining three parts of geometry and appending them to one array
	xp1 = np.empty(Nside)
	xp1.fill(-L/2)
	xm1 = np.empty(Nside)
	xm1.fill(-L/2)
	xp2 = np.linspace(-L/2+dx, L/2, Nbott)
	xm2 = np.linspace(-L/2, L/2 - dx, Nbott)
	xp3 = np.empty(Nside)
	xp3.fill(L/2)
	xm3 = np.empty(Nside)
	xm3.fill(L/2)

	#Appending to one array
	xp_append = np.append(xp1, xp2)
	xp = np.append(xp_append, xp3)
	xm_append = np.append(xm1, xm2)
	xm = np.append(xm_append, xm3)

	#Discretizing y-points
	yp1 = np.linspace(-dy, -D, Nside)
	ym1 = np.linspace(0, -D + dy, Nside)
	yp2 = np.empty(Nbott)
	yp2.fill(-D)
	ym2 = np.empty(Nbott)
	ym2.fill(-D)
	yp3 = np.linspace(-D + dy, 0, Nside)
	ym3 = np.linspace(-D, -dy, Nside)

	#Appending to one array
	yp_append = np.append(yp1, yp2)
	yp = np.append(yp_append, yp3)
	ym_append = np.append(ym1, ym2)
	ym = np.append(ym_append, ym3)
	xa = np.append(xm, xp[N-1])
	ya = np.append(ym, yp[N-1])
	
	#midtpoints
	xbar = 0.5*(xm +  xp)
	ybar = 0.5*(ym + yp)
	return xm, xp, ym, yp, xbar, ybar, D, L


def plot_geometry(D, L):
	xm, xp, ym, yp, xbar, ybar, D, L = discretize_geometry(D, L)
	Nside = int(D*10); Nbott = int(L*10)
	if L < 0.5:
		Nbott = 5
	N = Nside + Nbott + Nside

	dx = L/Nbott; dy = D/Nside
	xa = np.append(xm, xp[N-1])
	ya = np.append(ym, yp[N-1])


	plt.xlim(-L/2-dx, L/2+dx)
	plt.ylim(-D - dx, dx)
	plt.title(f'Discretization of geometry with D = {D} and L = {L}, L/D = {L/D}')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.scatter(xa, ya)
	plt.show()

def lhs_rhs(D, L):
	nu = 0.9
	#Here we change lenght and height!
	xm, xp, ym, yp, xbar, ybar, D, L = discretize_geometry(D, L)
	N = xp.shape[0]
	inn = np.linspace(0, N, N, endpoint=True)
	dx = xp - xm
	dy = yp - ym
	ds = np.sqrt(dx**2 + dy**2)
	n1 = -(yp-ym)/ds
	n2 = (xp-xm)/ds



	xg1 = -0.5*dx/np.sqrt(3) + xbar
	xg2 = 0.5*dx/np.sqrt(3) + xbar
	yg1 = -0.5*dy/np.sqrt(3) + ybar
	yg2 = 0.5*dy/np.sqrt(3) + ybar

	phi0 = np.exp(nu*(ybar - complex(0,1)*xbar))
	phi0n = nu*(n2 - complex(0, 1)*n1)*phi0


	gg = np.zeros((N, N), dtype=complex)
	ss = np.zeros((N, N), dtype=complex)
	#for i in range(N+1):
	for i in range(N):
		for j in range(N):
			xa1 = xg1[j] - xbar[i]
			xa2 = xg2[j] - xbar[i]
			ya1 = yg1[j] - ybar[i]
			ya2 = yg2[j] - ybar[i]
			ra1 = np.sqrt(xa1**2 + ya1**2)
			ra2 = np.sqrt(xa2**2 + ya2**2)
			g0 = 0.5*(np.log(ra1) + np.log(ra2))

			#midpoint rule
			xa = xbar[j] - xbar[i]
			yb = ybar[j] + ybar[i]
			rb = np.sqrt(xa**2 + yb**2)
			g1 = -np.log(rb)
			zz = nu*(yb - complex(0, 1)*xa) # (91)

			
			f1 = -2*np.exp(zz)*(mpm.e1(zz) + np.log(zz) - np.log(-zz)) #(88) from lec-notes
			f2 = 2*np.pi*np.exp(zz) #(90)
			g2 = f1.real + complex(0, 1)*f2.real #()
			gg[i, j] = (g0 + g1 + g2)*ds[j]


			arg0 = (np.log((xm[j] - xbar[i] + complex(0, 1)*(ym[j] - ybar[i]))/\
					(xp[j] - xbar[i] + complex(0, 1)*(yp[j] - ybar[i])))).imag	
			if j - i == 0:
				arg0 = np.pi


			arg1 = (np.log((xm[j] - xbar[i] + complex(0, 1)*(ym[j] + ybar[i]))/ \
					(xp[j] - xbar[i] + complex(0, 1)*(yp[j] + ybar[i])))).imag

			help1 = (n1[j]*(f1.imag + complex(0, 1)*f2.imag) + n2[j]* \
					(f1.real + complex(0, 1)*f2.real))*nu*ds[j]

			ss[i][j] = (arg0 + arg1 + help1)
	#print(ss)
	rhs = np.matmul(gg, phi0n)
	lhs = np.matmul(ss, phi0)

	return lhs, rhs, inn, D, L


def plot_lhs_rhs(D, L):
	
	lhs, rhs, inn, D, L = lhs_rhs(D, L)
	plt.plot(inn, rhs.real, label='rhs real')
	plt.plot(inn, lhs.real, 'gx', label = 'lhs real')
	plt.plot(inn, lhs.imag, 'bo', label = 'lhs imaginary')
	plt.plot(inn, rhs.imag, label='rhs imaginary')
	plt.legend()
	plt.title(f'Plot for L = {L} and D = {D}')
	plt.show()


def calculate_phi(D, L, nu):

	#Here we change lenght and height!
	xm, xp, ym, yp, xbar, ybar, D, L = discretize_geometry(D, L)
	N = xp.shape[0]
	inn = np.linspace(0, N, N, endpoint=True)
	dx = xp - xm
	dy = yp - ym
	ds = np.sqrt(dx**2 + dy**2)
	n1 = -(yp-ym)/ds
	n2 = (xp-xm)/ds



	xg1 = -0.5*dx/np.sqrt(3) + xbar
	xg2 = 0.5*dx/np.sqrt(3) + xbar
	yg1 = -0.5*dy/np.sqrt(3) + ybar
	yg2 = 0.5*dy/np.sqrt(3) + ybar

	phi0 = np.exp(nu*(ybar - complex(0,1)*xbar))
	phi0n = nu*(n2 - complex(0, 1)*n1)*phi0


	gg = np.zeros((N, N), dtype=complex)
	ss = np.zeros((N, N), dtype=complex)
	#for i in range(N+1):
	for i in range(N):
		for j in range(N):
			xa1 = xg1[j] - xbar[i]
			xa2 = xg2[j] - xbar[i]
			ya1 = yg1[j] - ybar[i]
			ya2 = yg2[j] - ybar[i]
			ra1 = np.sqrt(xa1**2 + ya1**2)
			ra2 = np.sqrt(xa2**2 + ya2**2)
			g0 = 0.5*(np.log(ra1) + np.log(ra2))

			#midpoint rule
			xa = xbar[j] - xbar[i]
			yb = ybar[j] + ybar[i]
			rb = np.sqrt(xa**2 + yb**2)
			g1 = -np.log(rb)
			zz = nu*(yb - complex(0, 1)*xa) # (91)

			
			f1 = -2*np.exp(zz)*(mpm.e1(zz) + np.log(zz) - np.log(-zz)) #(88) from lec-notes
			f2 = 2*np.pi*np.exp(zz) #(90)
			g2 = f1.real + complex(0, 1)*f2.real #()
			gg[i, j] = (g0 + g1 + g2)*ds[j]


			arg0 = (np.log((xm[j] - xbar[i] + complex(0, 1)*(ym[j] - ybar[i]))/\
					(xp[j] - xbar[i] + complex(0, 1)*(yp[j] - ybar[i])))).imag	
			if j - i == 0:
				arg0 = -np.pi


			arg1 = (np.log((xm[j] - xbar[i] + complex(0, 1)*(ym[j] + ybar[i]))/ \
					(xp[j] - xbar[i] + complex(0, 1)*(yp[j] + ybar[i])))).imag

			help1 = (n1[j]*(f1.imag + complex(0, 1)*f2.imag) + n2[j]* \
					(f1.real + complex(0, 1)*f2.real))*nu*ds[j]

			ss[i][j] = (arg0 + arg1 + help1)

	rhs = np.matmul(gg, n2)
	phi2 = np.linalg.solve(ss, rhs)
	ff22 = phi2*n2*ds
	sum_ff22 = sum(ff22)
	AM2 = complex(0, 1)*(phi2*(nu*n2 - nu*complex(0, 1)*n1) - n2)*phi0*ds
	AP2 = complex(0, 1)*(phi2*(nu*n2 + nu*complex(0, 1)*n1) - n2)*np.conj(phi0)*ds
	sum_AM2 = sum(AM2)
	sum_AP2 = sum(AP2)

	dampingb22 = 0.5*(sum_AM2*np.conj(sum_AM2) + sum_AP2*np.conj(sum_AP2))
	v1H = sum(-complex(0, 1)*(phi0*n2 - phi2*phi0n)*ds)

	return phi2, inn, sum_ff22, sum_AM2, sum_AP2, v1H, dampingb22


def energy_calculations(D, L, nu):
	#Here we change lenght and height!
	xm, xp, ym, yp, xbar, ybar, D, L = discretize_geometry(D, L)
	N = xp.shape[0]
	inn = np.linspace(0, N, N, endpoint=True)
	dx = xp - xm
	dy = yp - ym
	ds = np.sqrt(dx**2 + dy**2)
	n1 = -(yp-ym)/ds
	n2 = (xp-xm)/ds

	phi0 = np.exp(nu*(ybar - complex(0,1)*xbar))
	ss = np.zeros((N, N), dtype=complex)

	for i in range(N):
		for j in range(N):
			xa = xbar[j] - xbar[i]
			yb = ybar[j] + ybar[i]
			zz = nu*(yb - complex(0, 1)*xa)
			f1 = -2*np.exp(zz)*mpm.e1(zz) + np.log(zz)
			zz = nu*(yb - complex(0, 1)*xa) # (91)

			
			f1 = -2*np.exp(zz)*(mpm.e1(zz) + np.log(zz) - np.log(-zz)) #(88) from lec-notes
			f2 = 2*np.pi*np.exp(zz) #(90)
			g2 = f1.real + complex(0, 1)*f2.real #()

			arg0 = (np.log((xm[j] - xbar[i] + complex(0, 1)*(ym[j] - ybar[i]))/\
					(xp[j] - xbar[i] + complex(0, 1)*(yp[j] - ybar[i])))).imag	

			if j - i == 0:
				arg0 = -np.pi

			arg1 = (np.log((xm[j] - xbar[i] + complex(0, 1)*(ym[j] + ybar[i]))/ \
					(xp[j] - xbar[i] + complex(0, 1)*(yp[j] + ybar[i])))).imag
			help1 = (n1[j]*(f1.imag + complex(0, 1)*f2.imag) + n2[j]* \
					(f1.real + complex(0, 1)*f2.real))*nu*ds[j]

			ss[i, j] = arg0 + arg1 + help1
	rhsD = -2*np.pi*phi0
	phiD = np.linalg.solve(ss, rhsD)

	XX2 = phiD*n2*ds
	sXX2 = sum(XX2)
	X2 = -complex(0, 1)*sXX2

	return phiD, inn, X2


def plot_energy(inn, D, L):
	nu = np.linspace(0, 2, num=100, endpoint=False)[1:]
	sff2_list = []
	X2_ = []
	b223_energy = []
	for _nu_ in nu:
		a, b, sff2, AM2, AP2, e, dampingb22 = calculate_phi(D, L, _nu_)
		sff2_list.append(sff2)
		b223_energy.append(dampingb22)
		#a, b, X2 = energy_calculations(D, L, _nu_)
		#X2_.append(X2)
	
	plt.plot(nu, np.real(sff2_list), label=r'$\frac{a_{22}}{\rho D^2}$')
	plt.plot(nu, -np.imag(sff2_list),'*', label=r'$\frac{b_{22}}{\rho \omega D^2}$')
	plt.plot(nu, np.real(b223_energy), '--', label=r'$\frac{b_{22}}{\rho \omega D^2}$' + 'energy')
	plt.xlabel(r'$\frac{\omega^2 D}{g}$')
	plt.title(f'Added mass and damping for L = {L} and D = {D}')
	plt.legend()
	plt.show()

def plot_diffraction(inn, D, L):
	xm, xp, ym, yp, xbar, ybar, D, L = discretize_geometry(D, L)
	phiD, a, b = energy_calculations(D, L, 0.9)
	plt.plot(inn, phiD.real, label=r'$Re(\phi_D$')
	plt.plot(inn, phiD.imag, label=r'$Im(\phi_D)$')
	plt.ylabel(r'$\phi_D$')
	plt.xlabel('m')
	plt.title(f'Diffraction potential for L = {L} and D = {D}')
	plt.legend()
	plt.show()




def plot_exi_force(inn, D, L):
	nu = np.linspace(0, 2, num=100, endpoint=False)[1:]
	X2_list = []
	for _nu_ in nu:
		a, b, X2 = energy_calculations(D, L, _nu_)
		X2_list.append(X2)
	plt.plot(nu, 2*np.exp(-nu*D)*np.sin(nu*L/2)/nu, label=r'$\frac{\left|X_{2}\right|}{\rho g D}$')
	plt.xlabel(r'$\frac{\omega^2 D}{g}$')
	plt.title(f'Exciting force for L = {L}, D = {D}')
	plt.legend()
	plt.show()



def plot_phi(inn, phi2, D, L):
	plt.plot(inn, phi2.real, 'x', label='Real part')
	plt.plot(inn, phi2.imag, '-', label='Imaginary part')
	plt.xlabel('m')
	plt.ylabel(r'$\phi_2$')
	plt.title(f'Heave potential for geometry with D = {D}, L = {L}')
	plt.legend()
	plt.show()

def plot_haskind_FK_and_direct_pressure(inn, D, L):
	nu = np.linspace(0, 2, num=100, endpoint=False)[1:]
	X2 = []
	X2140 = []
	direct_pressure = []
	for _nu_ in nu:
		a, b, c, sum_AM2, d, v1H = calculate_phi(D, L, _nu_)
		a, b, dir_press = energy_calculations(D, L, _nu_)
		X2.append(sum_AM2)
		X2140.append(v1H)
		direct_pressure.append(dir_press)

	#Froude-Krylov	
	X_FK = 2*np.exp(-nu*D)*np.sin(nu*L/2)/nu

	plt.xlabel(r'$\frac{\omega^2 d}{g}$')
	plt.ylabel(r'$\frac{\left|X_{2}\right|}{\rho g D}$')
	plt.plot(nu, np.absolute(direct_pressure), '-', label='Direct Pressure')
	plt.plot(nu, np.absolute(X2140), '*', label='Haskind (158)')
	plt.plot(nu, np.absolute(X2), '^', label='Haskind ver 2 (162)')
	plt.plot(nu, np.absolute(X_FK), 'x', label='Froude-Krylov')
	plt.title(f'Plot for L = {L} and D = {D}')
	plt.legend()
	plt.show()


def plot_resonance_freq(inn, D, L):
	nu = np.linspace(0, 2, num=100, endpoint=False)[1:]
	sff2_list = []
	for _nu_ in nu:
		a, b, sff2, c, d, e = calculate_phi(D, L, _nu_)
		sff2_list.append(sff2)
	omega_n = np.sqrt(1/(D*(1+np.real(sff2_list)*D/L)))
	plt.plot(nu, omega_n, 'x')
	plt.xlabel(r'$\frac{\left|X_{2}\right|}{\rho g D}$')
	plt.ylabel(r'$\frac{\omega_n \sqrt{D}}{\sqrt{g}}$')
	plt.title(f'Resonance frequency for L = {L} and D = {D}')
	plt.show()


def plot_response_freq(inn, D, L):
	nu = np.linspace(0, 2, num=100, endpoint=False)[1:]
	sAM_list = []
	sff2_list = []
	for _nu_ in nu:
		a, b, sff2, sum_AM2, d, e = calculate_phi(D, L, _nu_)
		sAM_list.append(sum_AM2)
		sff2_list.append(sff2)

	xi_A1 = np.absolute(sAM_list/(L-nu*(L + np.real(sff2_list)*D) \
		+ complex(0, 1)*nu*D*np.imag(sff2_list)))

	plt.plot(nu, xi_A1, '*')
	plt.xlabel(r'$\frac{\omega^2 d}{g}$')
	plt.ylabel(r'$\frac{\left|\xi_2 \right|}{A}$')
	plt.title(f'Response as a function of freq, L = {L} and D = {D}')
	plt.legend()
	plt.show()
	
def plot_response_freq2(inn, D, L):
	nu = np.linspace(0, 2, num=100, endpoint=False)[1:]
	sAM_list = []
	sff2_list = []
	for _nu_ in nu:
		a, b, sff2, sum_AM2, d, e = calculate_phi(D, L, _nu_)
		sAM_list.append(sum_AM2)
		sff2_list.append(sff2)

	xi_A1 = np.absolute(sAM_list/(L-nu*(L + np.real(sff2_list)*D) \
		+ complex(0, 1)*nu*D*np.imag(sff2_list)))
	#term without added mass
	X_FK = 2*np.exp(-nu*D)*np.sin(nu*L/2)/nu
	xi_A1_WAM = np.absolute(X_FK/(L - nu*L + complex(0, 1)*nu*D*X_FK**2/D**2))
	plt.plot(nu, xi_A1, '-', label='All terms')
	plt.plot(nu, xi_A1_WAM, label='Neglected added mass')
	plt.xlabel(r'$\frac{\omega^2 d}{g}$')
	plt.ylabel(r'$\frac{\left|\xi_2 \right|}{A}$')
	plt.title(f'Plot to compare diff for neglected addded mass (L = {L}, D = {D})')
	plt.legend()
	plt.show()



def plot_response_freq3(inn, D, L):
	nu = np.linspace(0, 2, num=100, endpoint=False)[1:]
	sAM_list = []
	sff2_list = []
	for _nu_ in nu:
		a, b, sff2, sum_AM2, d, e = calculate_phi(D, L, _nu_)
		sAM_list.append(sum_AM2)
		sff2_list.append(sff2)

	xi_A1 = np.absolute(sAM_list/(L-nu*(L + np.real(sff2_list)*D) \
		+ complex(0, 1)*nu*D*np.imag(sff2_list)))
	#term without added mass
	X_FK = 2*np.exp(-nu*D)*np.sin(nu*L/2)/nu
	xi_A1_WAM = np.absolute(X_FK/(L - nu*L + complex(0, 1)*nu*D*X_FK**2/D**2))
	XIA_a22 = np.absolute(X_FK/(L - nu*(L + np.real(sff2_list)*D) + complex(0, 1)*nu*D*X_FK**2/D**2))
	plt.plot(nu, xi_A1, '-', label='All terms')
	plt.plot(nu, xi_A1_WAM, 'x', label='Haskind, neglected added mass')
	plt.plot(nu, XIA_a22, '^', label='Haskind, a22 included')
	plt.xlabel(r'$\frac{\omega^2 d}{g}$')
	plt.ylabel(r'$\frac{\left|\xi_2 \right|}{A}$')
	plt.title(f'Plot to compare diff for neglected addded mass (L = {L}, D = {D})')
	plt.legend()
	plt.show()

if __name__ == "__main__":

	#plot_lhs_rhs(1, 0.1)
	phi2, inn, sum_ff22, sum_AM2, sum_AP2, v1H, dampingb22 = calculate_phi(1, 0.1, 0.9)
	#plot_phi(inn, phi2, 1, 0.5)
	plot_energy(inn, 1, 0.1)
	#plot_exi_force(inn, 1, 0.1)
	#plot_diffraction(inn, 1, 0.1)
	#plot_haskind_FK_and_direct_pressure(inn, 1, 0.1)
	#plot_phi(inn, phi2, 1, 0.1)
	#plot_resonance_freq(inn, 1, 0.1)
	#plot_response_freq3(inn, 1, 0.1)
