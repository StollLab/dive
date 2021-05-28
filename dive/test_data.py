import deerlab as dl
import numpy as np

def generateSingleGauss(sigma = 0.01, r0 = 4, w = 0.4, nt = 150,  nr = 800, lamb = 0.5, k = 0.1, V0 = 1, seed = 0, r_edges = [1,10], t_edges = [-0.1,2.5]):
    t = np.linspace(t_edges[0],t_edges[1],nt)        # time axis, µs
    r = np.linspace(r_edges[0],r_edges[1],nr)      # distance axis, nm

    P = dl.dd_gauss(r,[r0,w])          # model distance distribution

    B = dl.bg_exp(t,k)         # background decay

    K = dl.dipolarkernel(t,r,integralop=True)    # kernel matrix
    K[:,0] = 2*K[:,0]
    K[:,-1] = 2*K[:,-1]

    Sm = K@P
    S = Sm + dl.whitegaussnoise(t,sigma,seed = seed)

    KB = dl.dipolarkernel(t,r,mod=lamb,bg=B,integralop=True)
    KB[:,0] = 2*KB[:,0]
    KB[:,-1] = 2*KB[:,-1]

    Vm = V0*KB@P
    V = Vm + dl.whitegaussnoise(t,sigma,seed = seed)

    pars = {'gaussian': [r0,w], 'lamb': lamb, 'k': k, 'V0': V0, 'sigma': sigma, 'seed': seed}
    data = {'t': t, 'V': V, 'S': S, 'r': r, 'P': P, 'V0': Vm, 'S0': Sm}
    return data, pars


def generateMultiGauss(sigma = 0.01, gausspars = [4, 0.3,0.6, 4.8, 0.5, 0.4], nt = 150, nr = 800, lamb = 0.5, k = 0.1, V0 = 1, seed = 0, r_edges = [1,10], t_edges = [-0.1,2.5]):
    t = np.linspace(t_edges[0],t_edges[1],nt)        # time axis, µs
    r = np.linspace(r_edges[0],r_edges[1],nr)      # distance axis, nm

    P = dl.dd_gauss2(r,gausspars)          # model distance distribution

    B = dl.bg_exp(t,k)         # background decay

    K = dl.dipolarkernel(t,r,integralop=True)    # kernel matrix
    K[:,0] = 2*K[:,0]
    K[:,-1] = 2*K[:,-1]

    Sm = K@P
    S = Sm + dl.whitegaussnoise(t,sigma,seed = seed)

    KB = dl.dipolarkernel(t,r,mod=lamb,bg=B,integralop=True)
    KB[:,0] = 2*KB[:,0]
    KB[:,-1] = 2*KB[:,-1]

    Vm = V0*KB@P
    V = Vm + + dl.whitegaussnoise(t,sigma,seed = seed)

    pars = {'gaussians': gausspars, 'lamb': lamb, 'k': k, 'V0': V0, 'sigma': sigma, 'seed': seed}
    data = {'t': t, 'V': V, 'S': S, 'r': r, 'P': P, 'V0': Vm, 'S0': Sm}
    return data, pars

def generateBiModalGauss(sigma = 0.01, gausspars = [4, 0.3,0.6, 6, 0.5, 0.4], nt = 150, nr = 800, lamb = 0.5, k = 0.1, V0 = 1, seed = 0, r_edges = [1,10], t_edges = [-0.1,2.5]):
    t = np.linspace(t_edges[0],t_edges[1],nt)        # time axis, µs
    r = np.linspace(r_edges[0],r_edges[1],nr)      # distance axis, nm


    P = dl.dd_gauss2(r,gausspars)          # model distance distribution

    B = dl.bg_exp(t,k)         # background decay

    K = dl.dipolarkernel(t,r,integralop=True)    # kernel matrix
    K[:,0] = 2*K[:,0]
    K[:,-1] = 2*K[:,-1]

    Sm = K@P
    S = Sm + dl.whitegaussnoise(t,sigma,seed = seed)

    KB = dl.dipolarkernel(t,r,mod=lamb,bg=B,integralop=True)
    KB[:,0] = 2*KB[:,0]
    KB[:,-1] = 2*KB[:,-1]

    Vm = V0*KB@P
    V = Vm + + dl.whitegaussnoise(t,sigma,seed = seed)

    pars = {'gaussians': gausspars, 'lamb': lamb, 'k': k, 'V0': V0, 'sigma': sigma, 'seed': seed}
    data = {'t': t, 'V': V, 'S': S, 'r': r, 'P': P, 'V0': Vm, 'S0': Sm}
    return data, pars