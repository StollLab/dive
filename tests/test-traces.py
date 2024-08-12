import dive
import pymc as pm
import deerlab as dl
import numpy as np

import pytest

@pytest.fixture
def t():
    return np.linspace(-0.1,5,300)

@pytest.fixture
def Vexp(t):
    # setting up data
    r = np.linspace(2,7,251) 
    P = dive.dd_gauss(r,5,0.8)

    lam = 0.4                       # modulation depth
    B = dl.bg_hom3d(t,0.2,lam)         # background decay
    K = dl.dipolarkernel(t,r,mod=lam,bg=B)  # kernel matrix

    return K@P + dl.whitegaussnoise(t,0.01,seed=10)

def test_running_regularization(t,Vexp):
    r = np.linspace(2,7,50)
    model = dive.model(t,Vexp,method="regularization",r=r)
    trace = dive.sample(model,draws=10,tune=20,chains=1,random_seed=100,
                        progressbar=False)
    lamb = trace.posterior.lamb[0][0].values
    assert lamb == 0.28143793404406964

def test_running_gaussian_1(t,Vexp):
    r = np.linspace(2,7,50)
    model = dive.model(t,Vexp,method="gaussian",r=r)
    trace = dive.sample(model,draws=10,tune=20,chains=1,random_seed=100,
                        progressbar=False)
    lamb = trace.posterior.lamb[0][0].values
    assert lamb == 0.35242692781607926

def test_running_gaussian_2(t,Vexp):
    r = np.linspace(2,7,50)
    model = dive.model(t,Vexp,method="gaussian",r=r,n_gauss=2)
    trace = dive.sample(model,draws=10,tune=20,chains=1,random_seed=100,
                        progressbar=False)
    lamb = trace.posterior.lamb[0][0].values
    assert lamb == 0.3931032862443532