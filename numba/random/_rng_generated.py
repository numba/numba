
rk_randomseed = mtrand.rk_randomseed
rk_randomseed.restype = rk_error
rk_randomseed.argtypes = [rk_address]

def randomseed():
    return rk_randomseed(state_p)

rk_random = mtrand.rk_random
rk_random.restype = ct.c_ulong
rk_random.argtypes = [rk_address]

def random():
    return rk_random(state_p)

rk_long = mtrand.rk_long
rk_long.restype = ct.c_long
rk_long.argtypes = [rk_address]

def long():
    return rk_long(state_p)

rk_ulong = mtrand.rk_ulong
rk_ulong.restype = ct.c_ulong
rk_ulong.argtypes = [rk_address]

def ulong():
    return rk_ulong(state_p)

rk_double = mtrand.rk_double
rk_double.restype = ct.c_double
rk_double.argtypes = [rk_address]

def double():
    return rk_double(state_p)

rk_gauss = mtrand.rk_gauss
rk_gauss.restype = ct.c_double
rk_gauss.argtypes = [rk_address]

def gauss():
    return rk_gauss(state_p)

rk_normal = mtrand.rk_normal
rk_normal.restype = ct.c_double
rk_normal.argtypes = [rk_address,ct.c_double,ct.c_double]

def normal(loc,scale):
    return rk_normal(state_p,loc,scale)

rk_standard_exponential = mtrand.rk_standard_exponential
rk_standard_exponential.restype = ct.c_double
rk_standard_exponential.argtypes = [rk_address]

def standard_exponential():
    return rk_standard_exponential(state_p)

rk_exponential = mtrand.rk_exponential
rk_exponential.restype = ct.c_double
rk_exponential.argtypes = [rk_address,ct.c_double]

def exponential(scale):
    return rk_exponential(state_p,scale)

rk_uniform = mtrand.rk_uniform
rk_uniform.restype = ct.c_double
rk_uniform.argtypes = [rk_address,ct.c_double,ct.c_double]

def uniform(loc,scale):
    return rk_uniform(state_p,loc,scale)

rk_standard_gamma = mtrand.rk_standard_gamma
rk_standard_gamma.restype = ct.c_double
rk_standard_gamma.argtypes = [rk_address,ct.c_double]

def standard_gamma(shape):
    return rk_standard_gamma(state_p,shape)

rk_gamma = mtrand.rk_gamma
rk_gamma.restype = ct.c_double
rk_gamma.argtypes = [rk_address,ct.c_double,ct.c_double]

def gamma(shape,scale):
    return rk_gamma(state_p,shape,scale)

rk_beta = mtrand.rk_beta
rk_beta.restype = ct.c_double
rk_beta.argtypes = [rk_address,ct.c_double,ct.c_double]

def beta(a,b):
    return rk_beta(state_p,a,b)

rk_chisquare = mtrand.rk_chisquare
rk_chisquare.restype = ct.c_double
rk_chisquare.argtypes = [rk_address,ct.c_double]

def chisquare(df):
    return rk_chisquare(state_p,df)

rk_noncentral_chisquare = mtrand.rk_noncentral_chisquare
rk_noncentral_chisquare.restype = ct.c_double
rk_noncentral_chisquare.argtypes = [rk_address,ct.c_double,ct.c_double]

def noncentral_chisquare(df,nonc):
    return rk_noncentral_chisquare(state_p,df,nonc)

rk_f = mtrand.rk_f
rk_f.restype = ct.c_double
rk_f.argtypes = [rk_address,ct.c_double,ct.c_double]

def f(dfnum,dfden):
    return rk_f(state_p,dfnum,dfden)

rk_noncentral_f = mtrand.rk_noncentral_f
rk_noncentral_f.restype = ct.c_double
rk_noncentral_f.argtypes = [rk_address,ct.c_double,ct.c_double,ct.c_double]

def noncentral_f(dfnum,dfden,nonc):
    return rk_noncentral_f(state_p,dfnum,dfden,nonc)

rk_binomial = mtrand.rk_binomial
rk_binomial.restype = ct.c_long
rk_binomial.argtypes = [rk_address,ct.c_long,ct.c_double]

def binomial(n,p):
    return rk_binomial(state_p,n,p)

rk_binomial_btpe = mtrand.rk_binomial_btpe
rk_binomial_btpe.restype = ct.c_long
rk_binomial_btpe.argtypes = [rk_address,ct.c_long,ct.c_double]

def binomial_btpe(n,p):
    return rk_binomial_btpe(state_p,n,p)

rk_binomial_inversion = mtrand.rk_binomial_inversion
rk_binomial_inversion.restype = ct.c_long
rk_binomial_inversion.argtypes = [rk_address,ct.c_long,ct.c_double]

def binomial_inversion(n,p):
    return rk_binomial_inversion(state_p,n,p)

rk_negative_binomial = mtrand.rk_negative_binomial
rk_negative_binomial.restype = ct.c_long
rk_negative_binomial.argtypes = [rk_address,ct.c_double,ct.c_double]

def negative_binomial(n,p):
    return rk_negative_binomial(state_p,n,p)

rk_poisson = mtrand.rk_poisson
rk_poisson.restype = ct.c_long
rk_poisson.argtypes = [rk_address,ct.c_double]

def poisson(lam):
    return rk_poisson(state_p,lam)

rk_poisson_mult = mtrand.rk_poisson_mult
rk_poisson_mult.restype = ct.c_long
rk_poisson_mult.argtypes = [rk_address,ct.c_double]

def poisson_mult(lam):
    return rk_poisson_mult(state_p,lam)

rk_poisson_ptrs = mtrand.rk_poisson_ptrs
rk_poisson_ptrs.restype = ct.c_long
rk_poisson_ptrs.argtypes = [rk_address,ct.c_double]

def poisson_ptrs(lam):
    return rk_poisson_ptrs(state_p,lam)

rk_standard_cauchy = mtrand.rk_standard_cauchy
rk_standard_cauchy.restype = ct.c_double
rk_standard_cauchy.argtypes = [rk_address]

def standard_cauchy():
    return rk_standard_cauchy(state_p)

rk_standard_t = mtrand.rk_standard_t
rk_standard_t.restype = ct.c_double
rk_standard_t.argtypes = [rk_address,ct.c_double]

def standard_t(df):
    return rk_standard_t(state_p,df)

rk_vonmises = mtrand.rk_vonmises
rk_vonmises.restype = ct.c_double
rk_vonmises.argtypes = [rk_address,ct.c_double,ct.c_double]

def vonmises(mu,kappa):
    return rk_vonmises(state_p,mu,kappa)

rk_pareto = mtrand.rk_pareto
rk_pareto.restype = ct.c_double
rk_pareto.argtypes = [rk_address,ct.c_double]

def pareto(a):
    return rk_pareto(state_p,a)

rk_weibull = mtrand.rk_weibull
rk_weibull.restype = ct.c_double
rk_weibull.argtypes = [rk_address,ct.c_double]

def weibull(a):
    return rk_weibull(state_p,a)

rk_power = mtrand.rk_power
rk_power.restype = ct.c_double
rk_power.argtypes = [rk_address,ct.c_double]

def power(a):
    return rk_power(state_p,a)

rk_laplace = mtrand.rk_laplace
rk_laplace.restype = ct.c_double
rk_laplace.argtypes = [rk_address,ct.c_double,ct.c_double]

def laplace(loc,scale):
    return rk_laplace(state_p,loc,scale)

rk_gumbel = mtrand.rk_gumbel
rk_gumbel.restype = ct.c_double
rk_gumbel.argtypes = [rk_address,ct.c_double,ct.c_double]

def gumbel(loc,scale):
    return rk_gumbel(state_p,loc,scale)

rk_logistic = mtrand.rk_logistic
rk_logistic.restype = ct.c_double
rk_logistic.argtypes = [rk_address,ct.c_double,ct.c_double]

def logistic(loc,scale):
    return rk_logistic(state_p,loc,scale)

rk_lognormal = mtrand.rk_lognormal
rk_lognormal.restype = ct.c_double
rk_lognormal.argtypes = [rk_address,ct.c_double,ct.c_double]

def lognormal(mean,sigma):
    return rk_lognormal(state_p,mean,sigma)

rk_rayleigh = mtrand.rk_rayleigh
rk_rayleigh.restype = ct.c_double
rk_rayleigh.argtypes = [rk_address,ct.c_double]

def rayleigh(mode):
    return rk_rayleigh(state_p,mode)

rk_wald = mtrand.rk_wald
rk_wald.restype = ct.c_double
rk_wald.argtypes = [rk_address,ct.c_double,ct.c_double]

def wald(mean,scale):
    return rk_wald(state_p,mean,scale)

rk_zipf = mtrand.rk_zipf
rk_zipf.restype = ct.c_long
rk_zipf.argtypes = [rk_address,ct.c_double]

def zipf(a):
    return rk_zipf(state_p,a)

rk_geometric = mtrand.rk_geometric
rk_geometric.restype = ct.c_long
rk_geometric.argtypes = [rk_address,ct.c_double]

def geometric(p):
    return rk_geometric(state_p,p)

rk_geometric_search = mtrand.rk_geometric_search
rk_geometric_search.restype = ct.c_long
rk_geometric_search.argtypes = [rk_address,ct.c_double]

def geometric_search(p):
    return rk_geometric_search(state_p,p)

rk_geometric_inversion = mtrand.rk_geometric_inversion
rk_geometric_inversion.restype = ct.c_long
rk_geometric_inversion.argtypes = [rk_address,ct.c_double]

def geometric_inversion(p):
    return rk_geometric_inversion(state_p,p)

rk_hypergeometric = mtrand.rk_hypergeometric
rk_hypergeometric.restype = ct.c_long
rk_hypergeometric.argtypes = [rk_address,ct.c_long,ct.c_long,ct.c_long]

def hypergeometric(good,bad,sample):
    return rk_hypergeometric(state_p,good,bad,sample)

rk_hypergeometric_hyp = mtrand.rk_hypergeometric_hyp
rk_hypergeometric_hyp.restype = ct.c_long
rk_hypergeometric_hyp.argtypes = [rk_address,ct.c_long,ct.c_long,ct.c_long]

def hypergeometric_hyp(good,bad,sample):
    return rk_hypergeometric_hyp(state_p,good,bad,sample)

rk_hypergeometric_hrua = mtrand.rk_hypergeometric_hrua
rk_hypergeometric_hrua.restype = ct.c_long
rk_hypergeometric_hrua.argtypes = [rk_address,ct.c_long,ct.c_long,ct.c_long]

def hypergeometric_hrua(good,bad,sample):
    return rk_hypergeometric_hrua(state_p,good,bad,sample)

rk_triangular = mtrand.rk_triangular
rk_triangular.restype = ct.c_double
rk_triangular.argtypes = [rk_address,ct.c_double,ct.c_double,ct.c_double]

def triangular(left,mode,right):
    return rk_triangular(state_p,left,mode,right)

rk_logseries = mtrand.rk_logseries
rk_logseries.restype = ct.c_long
rk_logseries.argtypes = [rk_address,ct.c_double]

def logseries(p):
    return rk_logseries(state_p,p)
