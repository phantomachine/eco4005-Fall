from scipy.optimize import fminbound
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.interpolate import pchip, Akima1DInterpolator
from scipy import interp
import scipy as sp
import scipy.stats as stats
import sys
import matplotlib.pyplot as plt
import warnings

class Bewley_Aiyagari(object):
    """Python class for solving a Bewley-Aiyagari incomplete markets 
    heterogeneous agent model. 
    (c) 2018, T. Kam (URL: phantomachine.github.io)"""
    
    def __init__(self, a_lb, a_ub, THETA=3.0, BETA=0.96, 
                 ALPHA=0.36, DELTA=0.08, RHO=0.8, SIGMA=0.005, 
                 NGRID_a = 40, NGRID_e = 3,
                 grid_method='linear', interp_method='slinear'):
        ## ------------------- Properties ----------------------------------
        self.GridMethod = grid_method
        self.InterpMethod = interp_method
                
        # Model parameters
        self.THETA = THETA # CRRA utility parameter
        self.ALPHA = ALPHA # Cobb-Douglas capital share
        self.BETA  = BETA  # Subjective discount factor
        self.DELTA = DELTA # Capital depreciation rate
        self.RHO = RHO     # Persistence of skill endowment (AR1 process)
        self.SIGMA = SIGMA # Volatility (std.dev.) of skill endowment shock            
        # Markov chain approximation of AR(1) shocks
        self.seed = True # fix seeding random number simulation (alt: None)
        self.NGRID_e = NGRID_e
        S, self.P = self.ar2mc()
        self.S = np.exp(S)
        # Solution space parameters
        self.r_lb = -DELTA   # (lower, upper) bounds on r
        self.r_ub = 1.0/BETA - 1.0
        self.a_lb = a_lb   # (lower, upper) bounds on a
        self.a_ub = a_ub       
        self.asset_grid = self.grid1d(a_lb, a_ub, Npoints=NGRID_a)
        self.NGRID_a = NGRID_a
        self.amat = np.tile(self.asset_grid, (NGRID_e, 1)).T
        self.emat = np.tile(self.S, (NGRID_a, 1))
        # Exogenous labor supply setting
        # Total labor supply - fixed (no endogenous labor supply)
        mu = self.ErgodistMC(self.P) # long run distro of skill levels
        self.N = mu @ self.S
        # Precision settings
        self.MAXITER_policy = 233
        self.MAXITER_value = 488
        self.MAXITER_distro = 49988
        self.MAXITER_price = 238
        self.TOL_policy = 1e-4      # stopping criterion: policy iteration
        self.TOL_value = 1e-4       # stopping criterion: value iteration
        self.TOL_distro = 1e-5      # stopping criterion: distribution iteration
        self.TOL_price = 1e-4       # stopping criterion: Walrasian pricing
        self.TOL_K = 1e-2           # stopping criterion: Walrasian excess demand
        # Secant algorithm settings
        self.SMOOTH = 0.56        
    
    ## ------------------- Methods -----------------------------------------
    def StatusBar(self, iteration, iteration_max, stats1, width=15):
        percent = float(iteration)/iteration_max
        sys.stdout.write("\r")
        progress = ""
        for i in range(width):
            if i <= int(width * percent):
                progress += "="
            else:
                progress += "-"
        sys.stdout.write(
            "[ %s ] %.2f%% %i/%i, error1 = %0.5f    "
            % (progress,percent*100,iteration,iteration_max,stats1)
            )
        sys.stdout.flush()
        
    def grid1d(self, xmin, xmax, Npoints=100):
        """Create 1D grid points: linear, inverse ratio scale, 
        or chebychev nodes. Default is linear: equally spaces gridpoints."""
        method = self.GridMethod
        if method=='linear':
            y = np.linspace(xmin, xmax, Npoints)
        elif method=='inverse_ratio':
            xmin_log = np.log(xmin - xmin + 1.0)/np.log(10.0)
            xmax_log = np.log(xmax - xmin + 1.0)/np.log(10.0)
            y = np.logspace(xmin_log, xmax_log, Npoints) + (xmin - 1.0)
        elif method=='chebychev':
            m = np.arange(1,Npoints+1)
            x = np.cos( (2.0*m - 1.0)*np.pi / (2.0*Npoints) )
            constant = 0.5*(xmin + xmax)
            slope = 0.5*(xmax - xmin)
            y = constant + slope*x
        return y
    
    def InterpFun1d(self, xdata, ydata):
        """ Interpolate 1D functions given data points (xdata, ydata). 
        Returns instance of class: funfit. 
        NOTE: funfit.derivative() will provide the derivative functions 
        of particular smooth approximant---depends on class of interpolating 
        function. See SCIPY.INTERPOLATE subclasses for more detail.   """
        if self.InterpMethod=='slinear':
            funfit = spline(xdata, ydata, k=1) # B-spline 1st order
        elif self.InterpMethod=='squadratic':
            funfit = spline(xdata, ydata,  k=2)# instantiate B-spline interp
        elif self.InterpMethod=='scubic':
            funfit = spline(xdata, ydata, k=3) # instantiate B-spline interp
        elif self.InterpMethod=='squartic':
            funfit = spline(xdata, ydata, k=4) # instantiate B-spline interp
        elif self.InterpMethod=='squintic':
            funfit = spline(xdata, ydata,  k=5) # instantiate B-spline interp
        elif self.InterpMethod=='pchip':
            # Shape preserving Piecewise Cubic Hermite Interp Polynomial splines
            funfit = pchip(xdata, ydata)
        elif self.InterpMethod=='akima':
            funfit = Akima1DInterpolator(xdata, ydata)
        return funfit # instance at m point(s)
    
    def supnorm(self, function1, function2):
        """Returns the absolute maximal (supremum-norm) distance between
        two arbitrary NumPy ND-arrays (function coordinates)"""
        return (np.abs(function1 - function2)).max()
    
    def ar2mc(self):
        """
        Approximate an AR1 model by a finite-state-space Markov Chain (MC)
        (Rouwenhorst 1995, Econometrica). This method outperforms earlier
        approximation schemes of Tauchen (1986) or Tauchen and Hussey (1991)
        when RHO is very close to 1: Kopecky and Suen (2010, RED).

        Input: AR(1) model parameters, y+ = RHO*y + SIGMA*u, u ~ Normal(0,1)
               N, desired cardinality of finite state space (Z) of approx. MC

        Output: (S, P), state space and Markov matrix
        """
        # Extract params from self object
        RHO, SIGMA, N = self.RHO, self.SIGMA, self.NGRID_e        
        # State space S
        bound = np.sqrt((N-1)/(1-RHO**2.0))*SIGMA
        S = np.linspace(-bound, bound, N)
        # Transition probabilities (N = 2). 
        p = (RHO + 1.0) / 2.0
        q = p
        # Initial P_temp is P for N = 2 case
        P_temp = np.array([[ p,  1-p ],
                           [ 1-q, q  ]])
        # Construct transition matrix P
        if N == 2:
            P =  P_temp 
        elif N > 2:
            # Recursive build of P for N > 2
            for n in range(3, N+1):
                block1 = np.zeros((n,n))
                block2 = block1.copy()
                block3 = block1.copy()
                block4 = block1.copy()
                # Fill with last iteration's P_temp
                block1[:-1,:-1] =  P_temp
                block2[:-1,1:] =  P_temp
                block3[1:,:-1] =  P_temp
                block4[1:,1:] =  P_temp
                # Update P_temp for next iteration
                P_temp = p*block1 + (1-p)*block2 + (1-q)*block3 + q*block4
                P_temp[1:-1,:] = P_temp[1:-1,:]/2
            # Final P for N > 2
            P = P_temp
        return S, P
    
    def SimulateMarkovChain(self, Z=None, P=None, mu=None, T=88888):
        """Simulate T-length observations of Markov chain (mu,P)"""
        """Note: Finite state space Z admit integers or reals"""
        if Z is None:
            Z = self.S
        if P is None:
            P = self.P
        if mu is None:
            if self.seed:
                # Fix a seeded random generator
                np.random.seed(52348282)
            # Define arbitrary initial uncond. distro over Z
            mu = np.random.rand(Z.size)
            mu = mu/mu.sum()
        data = np.empty(T)
        data[0] = np.random.choice(Z, replace=False, p = mu)
        for t in range(T-1):
            if self.seed:
                np.random.seed(t + 1234)
            # Find index/location of element in Z with value data[t]
            state = Z.tolist().index(data[t])
            # Given state index, draw new state from conditional distro
            data[t+1] = np.random.choice(Z, replace=False, p = P[state,:])         
        return data   

    def ErgodistMC(self, P):
        """Compute stationary distribution of an ergodic Markov Chain"""
        N_state = P.shape[0]
        z = np.zeros(N_state)
        # Normalization: right eigenvector (z) as prob. dist.
        z[-1] = 1.0
        # System of linear equations: find fixed point z
        PMI = P - np.eye(N_state)
        PMI[:,-1] = np.ones(N_state)
        lambda_inf = np.linalg.solve(PMI.T,z.T)
        return lambda_inf
    
    def U(self, c):
        """CRRA family of utility functions"""
        THETA = self.THETA
        if THETA == 1.0:
            ohjoy = np.log(c)
        elif THETA != 1.0 and THETA > 0.0:
            ohjoy = (c**(1.0 - THETA) - 1.0)/ (1.0 - THETA)        
        return ohjoy
    
    def CobbDouglas_mpk(self, k):
        """Marginal product of capital according to the world of Cobb-Douglas"""
        # given per-worker capital, deduce mpk or r
        mpk = self.ALPHA*k**(self.ALPHA - 1.0)
        return mpk
    
    def CobbDouglas_netmpk_inverse(self, r):
        """Inverse of (net) MPK function according to the world of Cobb-Douglas"""
        # per-worker capital demand
        k_at_r = ((r + self.DELTA)/self.ALPHA)**(1.0/(self.ALPHA-1.0))
        # total demand
        K_demand = k_at_r*self.N
        return K_demand
    
    def CobbDouglas_mpl(self, r):
        """Marginal product of labor according to the world of Cobb-Douglas"""
        K = self.CobbDouglas_netmpk_inverse(r)
        w = (1. - self.ALPHA)*(K/self.N)**self.ALPHA
        return w
    
    def EV(self, v, prob, anext):
        """Expected Continuation value, given:
            * current guess of value function, v (NGRID_a x NGRID_e)
            * prob = P[index_z,:], (1 x NGRID_e) array
            * anext (continuation state(s)), scalar or (NGRID_a x 1) array
        Interpolate over data point(s) knext, given function v(k,z) (array) 
        defined on tabular domain X x Z. Take expectations w.r.t. prob"""
        NGRID_e = self.NGRID_e
        N_a = np.asarray(anext).size # Could be 1 or NGRID_a!
        vinterp = np.empty((NGRID_e, N_a))
        # Interpolate over k_{+1} for each slice v(.,znext) 
        for index_enext in range(NGRID_e):
            # Instantiate interpolating class on v(.,znext) as V_fitted
            V_fitted = self.InterpFun1d(self.asset_grid, v[:,index_enext])
            # Evaluate V_fitted at point knext
            vinterp[index_enext, :] = V_fitted(anext)
        # Calculate expected value given k_{+1} conditional on index_z:
        return prob.dot(vinterp)
    
    def BudgetConstraint(self, action, state, params):
        """Aiyagari's household budget constraint"""
        # Enumerate states
        a, e = state
        # Prices
        r, w = params
        # budget constraint
        c = a*(1.0 + r) + w*e - action
        return c

    def TotalPayoff_scalar(self, action, state, v, params):
        """Objective function evaluated at current (action, state) pair, given
        aggregate relative price r, and, continuation value function v"""
        # Enumerate states
        a, e = state # (a,e) could be scalars or arrays!
        # Get index of e in array S
        index_e = (self.S).tolist().index(e)
        # Budget constraint
        # r, w = params
        # c = a*(1.0 + r) + w*e - action
        c = self.BudgetConstraint(action, state, params)

        # Map out forever-happiness consequence of current action
        if c <= 0.0:
            u_now = -np.inf   # not happy, Jane!
        else:
            u_now = self.U(c) # happiness is a warm gun
        vnext = self.EV(v, self.P[index_e,:], action)
        TotalExpectedUtility = u_now + self.BETA*vnext
        return -TotalExpectedUtility
    
    def TotalPayoff_array(self, anext, state, v, params):
        """Objective function evaluated at current (action, state) pair, given
        aggregate relative price r, and, continuation value function v"""
        # Enumerate states (a,e): could be scalars or arrays!
        a, e = state
        a = np.atleast_2d(a) # ensure indexing works even if (a,e) scalars
        e = np.atleast_2d(e)
        anext = np.atleast_2d(anext)
        # Budget constraint
        r, w = params
        # cmat = (1. + r)*a + w*e - anext
        all_state = [a, e]
        cmat = self.BudgetConstraint(anext, all_state, params)

        # Inada conditions to bound c(a,e) > 0 and is affordable a.s.
        umat = np.tile(-np.inf, cmat.shape)      # Worst payoffs (for possible c <= 0)
        umat[cmat > 0] = self.U(cmat[cmat > 0])  # Replace with U(c) for feasible c > 0  
        # Derive value function induced by fixed rule, at each (a,e) pair:
        # 1. Calculate continuation value function
        v_old = np.zeros((self.NGRID_a, self.NGRID_e))
        for idx_e, e in enumerate(self.S):
            prob = self.P[idx_e, :]
            v_old[:,idx_e] = self.EV(v, prob, anext[:, idx_e])
        # 2. Total expected payoff from following anext (action) forever
        v_new = umat + self.BETA*v_old
        return -v_new
    
    def ValueFixedPolicy(self, anext, all_states, v, params):
        """Evaluate value function v, supported by fixed rule anext"""
        for iter_value in range(self.MAXITER_value):
            # Evaluate total payoffs under anext, enforced by promises v
            v_update = -self.TotalPayoff_array(anext, all_states, v, params) 
            # Check if convergence attained
            gap_value = self.supnorm(v, v_update)
            v = v_update
            if gap_value < self.TOL_value:
                break
            if (iter_value == self.MAXITER_value) and (gap_value >= self.TOL_value):
                print("MAXITER_value reached. Increase this number.")
        return v
    
    def Bellman(self, v, params):
        """Bellman operator, given 
           * guess of value function, v
           * aggregate prices, params=(r,w)
        """
        anext_update = np.empty(v.shape)
        v_update = anext_update.copy()
        r, w = params
        for idx_a, a in enumerate(self.asset_grid):
            for idx_e, e in enumerate(self.S):
                # Feasibility - upper bound at current state (a,e)
                a_upper = (1. + r)*a + w*e
                a_max = min(a_upper, self.a_ub)
                # Get optimizer and value at state (a,e)
                states = [a,e]
                anext_star, v_star, ierr, numfunc = fminbound(
                                                    self.TotalPayoff_scalar, 
                                                    self.a_lb, a_max, 
                                                    args=(states, v, params),
                                                    full_output=True
                                                    )
                # Store them to build up policy and value functions' updates
                anext_update[idx_a, idx_e] = anext_star
                v_update[idx_a, idx_e] = -v_star
        return anext_update, v_update
    
    def Howard(self, r, anext=None, v=None, display_howard=False): 
        """Given current aggregate price r, and last guess of policy, 
        get agent's updated best response. Uses Howard's policy improvement 
        algorithm: see LS, Chapter 4.4.
        """
        # (Noobs: anext and v are NGRID_a x NGRID_e arrays)
        if anext is None:
            anext = np.zeros((self.amat.shape)) 
        if v is None:
            v = np.zeros((self.amat.shape))
        # Current (guess) relative wage as function of r
        w = self.CobbDouglas_mpl(r)
        # 1. Current states: All pairs of (a,e) 
        all_states = [self.amat, self.emat]
        for iter_policy in range(self.MAXITER_policy):
            # 2. For fixed anext, compute its induced value function v
            v = self.ValueFixedPolicy(anext, all_states, v, params=[r,w])
            # 3. One-shot deviation principle: if exists optimal deviation,
            #    then anext_update would be different to and replaces anext
            anext_update, v_update = self.Bellman(v, params=[r,w])
            # 4. Measure how different they are
            gap_policy = self.supnorm(anext, anext_update)
            if display_howard is True:
                self.StatusBar(iter_policy,self.MAXITER_policy, gap_policy)
            # 5. Now update guesses on v and anext
            anext = anext_update
            v = v_update
            # 6. Stopping rule (otherwise repeat and rinse)
            #   Convergence in anext, and hence v, mean there exists   
            #   no profitable one-shot deviation :=: optimal policy and value found
            if gap_policy < self.TOL_policy:
                if display_howard is True:
                    self.StatusBar(iter_policy,iter_policy, gap_policy)
                break
            if (iter_policy == self.MAXITER_policy) and (gap_policy >= self.TOL_policy):
                print("MAXITER_policy reached. Increase this number.")
        return v, anext
    
    def MonteCarloSimulation(self, policy, a_init=None, e_path=None, T_sim=None):
        """Simulate random path of an agent given agent's optimal policy"""
        if T_sim is None:
            T_sim = self.MAXITER_distro
        # Simulate Markov chain realizations {e(t)}
        if e_path is None:
            e_path = self.SimulateMarkovChain(T=T_sim)
        # Construct sequence of equilibrium best response outcomes
        # a_init = self.asset_grid.min() # arbitrary start from poorest guy
        if a_init is None:
            a_init = self.asset_grid.min()
        a_path = [ a_init ]
        for t, e in enumerate(e_path):
            # Get index of e in S
            idx_e = self.S.tolist().index(e)
            # Bounds checking and updating path of a(t)
            if a_path[t] <= self.a_lb:
                # Po'-rest boy
                a_path.append(policy[0, idx_e])
            elif a_path[t] >= self.a_ub:
                # Rich-est daddy-O
                a_path.append(policy[-1, idx_e])
            else:
                # Typical in-betweeners
                astar = np.interp(a_path[t], self.asset_grid, policy[:, idx_e])
                a_path.append(astar)
        return np.asarray(a_path), np.asarray(e_path) 
    
    def AssetDistributionStats(self, a_path):
        """First four moments of a distribution for data: a_path"""
        mean = np.mean(a_path)
        std = np.std(a_path)
        skew = stats.skew(a_path)
        kurt = stats.kurtosis(a_path)
        return [mean, std, skew, kurt]
    
    def Tatonnement(self, K, r):
        """A Bisection Method algorithm representing tâtonnement: 
        Historically, an imagined process of finding a Walrasian 
        equilibrium pricing vector. Here sufficient to solve for r
        to get vector (1, r, w(r))"""
        # Given last guess of equilibrium (K, r) get new r_update
        r_update = self.CobbDouglas_mpk(K/self.N) - self.DELTA
        # Smoothing parameters (step-size in bisection search)
        smooth = self.SMOOTH
        # Initial brackets
        r_lb, r_ub = self.r_lb, self.r_ub
        # Squeeze-update brackets
        if r_update < r:
            if r_update < r_ub:
                r_update = smooth*r + (1. - smooth)*r_lb
            else:
                r_update = smooth*r + (1. - smooth)*r_update
                r_lb = r_update
            r_ub = r
        else:
            if r_update > r_ub:
                r_update = smooth*r + (1. - smooth)*r_ub
            else:
                r_update = smooth*r + (1. - smooth)*r_update
                r_lb = r_update
            r_lb = r
        # Auctioneer checks if r_update is no different from r
        error = np.absolute(r_update - r)
        
        return r_update, error      
    
    def ShowDistro(self, a_path):
        """Plot histogram of asset distribution"""
        freq, bins, patches = plt.hist(a_path, 600, density=True, align='mid',
                                       facecolor='green', alpha=0.5)
        plt.xlabel('Asset Position')
        plt.ylabel('Probability')
        plt.axis([0., 1.2*bins.max(), 0., freq.max()])
        plt.grid(True)
        plt.show()
        return freq, bins
    
    def Gini(self, data_array):
        """Gini coefficient of empirical distribution of data_array
        See: https://en.wikipedia.org/wiki/Gini_coefficient#Alternate_expressions"""
        # sort ascending
        sorted_array = data_array.copy()
        sorted_array.sort()
        T = data_array.size
        # Gini formula
        # 
        weight = 2.0 / T
        constant = (T + 1.0) / T
        weighted_sum = sum([(i+1)*yi for i, yi in enumerate(sorted_array)])
        return weight*weighted_sum/(sorted_array.sum()) - constant
    
    def LorenzCurve(self, data_array):
        """Plot Lorenz Curve from data_array"""
        # sort ascending
        data_array = data_array.copy()
        data_array.sort()
        # Normalized cumulative sum of x-axis
        y_lorenz = data_array.cumsum() / data_array.sum()
        y_lorenz = np.insert(y_lorenz, 0, 0) 
        # Plot Lorenz as scatter points
        fig, ax = plt.subplots(figsize=[6,6])
        plt.scatter(np.arange(y_lorenz.size)/(y_lorenz.size-1), y_lorenz, 
                   marker='.', color='darkgreen', s=0.05)
        # Line of perfect equality
        plt.plot([0,1], [0,1], '--k', alpha=0.5)
        plt.xlabel("Cumulative share of people from lowest to highest wealth")
        plt.ylabel("Cumulative share of wealth owned")
        plt.show()