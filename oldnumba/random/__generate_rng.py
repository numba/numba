def _groupn(iter, N):
    k = 0
    ret = ()
    for item in iter:
        k = k + 1
        ret += (item,)
        if k == N:
            yield ret
            k = 0
            ret = ()

_toreplace = [('unsigned long ', 'ct.c_ulong '),
              ('long ', 'ct.c_long '),
              ('double ', 'ct.c_double '),
              ('rk_state *', 'rk_address '),
              ('void *', 'ct.c_void_p '),
              ('void', 'None')]
        
# Return [name, restype, and argtypes, argnames]
def parse_header(header):
    mystr = '@@@_%d_@@@'
    for i, (old, new) in enumerate(_toreplace):
        header = header.replace(old, mystr % i)
    for i, (old, new) in enumerate(_toreplace):
        header = header.replace(mystr % i, new)
    funcs = [val.split(';')[0].strip() for indx, val in enumerate(header.split('extern')) if indx > 0]
    result = []
    for func in funcs:
        temp = func.split()
        restype = temp[0]
        name, arg0type = temp[1].split('(')
        argtypes = [arg0type]
        argnames = [temp[2].strip(')')]
        for atype, aname in _groupn(temp[3:], 2):
            argnames.append(aname.strip(')'))
            argtypes.append(atype)
        result.append([name, restype, argtypes, argnames])
    return result

header1 = """
/*
 * Initialize the RNG state using the given seed.
 */

/*
 * Initialize the RNG state using a random seed.
 * Uses /dev/random or, when unavailable, the clock (see randomkit.c).
 * Returns RK_NOERR when no errors occurs.
 * Returns RK_ENODEV when the use of RK_DEV_RANDOM failed (for example because
 * there is no such device). In this case, the RNG was initialized using the
 * clock.
 */
extern rk_error rk_randomseed(rk_state *state);

/*
 * Returns a random unsigned long between 0 and RK_MAX inclusive
 */
extern unsigned long rk_random(rk_state *state);

/*
 * Returns a random long between 0 and LONG_MAX inclusive
 */
extern long rk_long(rk_state *state);

/*
 * Returns a random unsigned long between 0 and ULONG_MAX inclusive
 */
extern unsigned long rk_ulong(rk_state *state);


/*
 * Returns a random double between 0.0 and 1.0, 1.0 excluded.
 */
extern double rk_double(rk_state *state);

/*
 * return a random gaussian deviate with variance unity and zero mean.
 */
extern double rk_gauss(rk_state *state);
"""

header2 = """ 
/* Normal distribution with mean=loc and standard deviation=scale. */
extern double rk_normal(rk_state *state, double loc, double scale);

/* Standard exponential distribution (mean=1) computed by inversion of the 
 * CDF. */
extern double rk_standard_exponential(rk_state *state);

/* Exponential distribution with mean=scale. */
extern double rk_exponential(rk_state *state, double scale);

/* Uniform distribution on interval [loc, loc+scale). */
extern double rk_uniform(rk_state *state, double loc, double scale);

/* Standard gamma distribution with shape parameter. 
 * When shape < 1, the algorithm given by (Devroye p. 304) is used.
 * When shape == 1, a Exponential variate is generated.
 * When shape > 1, the small and fast method of (Marsaglia and Tsang 2000) 
 * is used.
 */
extern double rk_standard_gamma(rk_state *state, double shape);

/* Gamma distribution with shape and scale. */
extern double rk_gamma(rk_state *state, double shape, double scale);

/* Beta distribution computed by combining two gamma variates (Devroye p. 432).
 */
extern double rk_beta(rk_state *state, double a, double b);

/* Chi^2 distribution computed by transforming a gamma variate (it being a
 * special case Gamma(df/2, 2)). */
extern double rk_chisquare(rk_state *state, double df);

/* Noncentral Chi^2 distribution computed by modifying a Chi^2 variate. */
extern double rk_noncentral_chisquare(rk_state *state, double df, double nonc);

/* F distribution computed by taking the ratio of two Chi^2 variates. */
extern double rk_f(rk_state *state, double dfnum, double dfden);

/* Noncentral F distribution computed by taking the ratio of a noncentral Chi^2
 * and a Chi^2 variate. */
extern double rk_noncentral_f(rk_state *state, double dfnum, double dfden, double nonc);

/* Binomial distribution with n Bernoulli trials with success probability p.
 * When n*p <= 30, the "Second waiting time method" given by (Devroye p. 525) is
 * used. Otherwise, the BTPE algorithm of (Kachitvichyanukul and Schmeiser 1988)
 * is used. */
extern long rk_binomial(rk_state *state, long n, double p);

/* Binomial distribution using BTPE. */
extern long rk_binomial_btpe(rk_state *state, long n, double p);

/* Binomial distribution using inversion and chop-down */
extern long rk_binomial_inversion(rk_state *state, long n, double p);

/* Negative binomial distribution computed by generating a Gamma(n, (1-p)/p)
 * variate Y and returning a Poisson(Y) variate (Devroye p. 543). */
extern long rk_negative_binomial(rk_state *state, double n, double p);

/* Poisson distribution with mean=lam.
 * When lam < 10, a basic algorithm using repeated multiplications of uniform
 * variates is used (Devroye p. 504).
 * When lam >= 10, algorithm PTRS from (Hoermann 1992) is used.
 */
extern long rk_poisson(rk_state *state, double lam);

/* Poisson distribution computed by repeated multiplication of uniform variates.
 */
extern long rk_poisson_mult(rk_state *state, double lam);

/* Poisson distribution computer by the PTRS algorithm. */
extern long rk_poisson_ptrs(rk_state *state, double lam);

/* Standard Cauchy distribution computed by dividing standard gaussians 
 * (Devroye p. 451). */
extern double rk_standard_cauchy(rk_state *state);

/* Standard t-distribution with df degrees of freedom (Devroye p. 445 as
 * corrected in the Errata). */
extern double rk_standard_t(rk_state *state, double df);

/* von Mises circular distribution with center mu and shape kappa on [-pi,pi]
 * (Devroye p. 476 as corrected in the Errata). */
extern double rk_vonmises(rk_state *state, double mu, double kappa);

/* Pareto distribution via inversion (Devroye p. 262) */
extern double rk_pareto(rk_state *state, double a);

/* Weibull distribution via inversion (Devroye p. 262) */
extern double rk_weibull(rk_state *state, double a);

/* Power distribution via inversion (Devroye p. 262) */
extern double rk_power(rk_state *state, double a);

/* Laplace distribution */
extern double rk_laplace(rk_state *state, double loc, double scale);

/* Gumbel distribution */
extern double rk_gumbel(rk_state *state, double loc, double scale);

/* Logistic distribution */
extern double rk_logistic(rk_state *state, double loc, double scale);

/* Log-normal distribution */
extern double rk_lognormal(rk_state *state, double mean, double sigma);

/* Rayleigh distribution */
extern double rk_rayleigh(rk_state *state, double mode);

/* Wald distribution */
extern double rk_wald(rk_state *state, double mean, double scale);

/* Zipf distribution */
extern long rk_zipf(rk_state *state, double a);

/* Geometric distribution */
extern long rk_geometric(rk_state *state, double p);
extern long rk_geometric_search(rk_state *state, double p);
extern long rk_geometric_inversion(rk_state *state, double p);

/* Hypergeometric distribution */
extern long rk_hypergeometric(rk_state *state, long good, long bad, long sample);
extern long rk_hypergeometric_hyp(rk_state *state, long good, long bad, long sample);
extern long rk_hypergeometric_hrua(rk_state *state, long good, long bad, long sample);

/* Triangular distribution */
extern double rk_triangular(rk_state *state, double left, double mode, double right);

/* Logarithmic series distribution */
extern long rk_logseries(rk_state *state, double p);
"""

#Create function and name
ctypes_template = """
{func_name} = mtrand.{func_name}
{func_name}.restype = {restype}
{func_name}.argtypes = {argtypes}
"""

func_template = """
def {pyfunc_name}({args}):
    return {func_name}({plus_args})
"""

_result1 = parse_header(header1)
_result2 = parse_header(header2)
_result = _result1 + _result2

afile = open('_rng_generated.py','w')

for func in _result:
    argstr = "[" + ",".join(func[2]) + "]"
    afile.write(ctypes_template.format(func_name = func[0],
                                restype = func[1], argtypes = argstr))
    args = [x.strip(',') for x in func[3][1:]]
    paramstr = ",".join(args)
    allparams = ",".join(['state_p']+args)
    afile.write(func_template.format(func_name = func[0], pyfunc_name = func[0][3:],
                                     args = paramstr, plus_args=allparams))

