% Sample P from full conditional Prob(P|delta,S,tau)
function P = randP(delta,tauKtK,tauKtS,LtL,nt)
    % based on:
    % J.M. Bardsley, C. Fox, An MCMC method for uncertainty quantification in
    % nonnegativity constrained inverse problems, Inverse Probl. Sci. Eng. 20 (2012)
    invSigma = tauKtK + delta*LtL;
    try
      % 'lower' syntax is faster for sparse matrices. Also matches convention in
      % Bardsley paper.
      C_L = chol(inv(invSigma),'lower');
    catch
      C_L = sqrtm(inv(invSigma));
    end
    v = randn(nt,1);
    w = C_L.'\v;
    P = fnnls(invSigma,tauKtS+w);
    end
    
    
    %-------------------------------------------------------------------------------
    % Sample delta from full conditional Prob(delta|P,S,tau)
    function delta = randdelta(P,a0,b0,nt,L)
    a_ = a0 + nt/2;
    b_ = b0 + (1/2)*norm(L*P)^2;
    % precision in distance domain %randraw uses the shape/scale paramaterization, but we use shape and rate.
    delta = randraw('gamma',[0,1/b_,a_],1);
    end