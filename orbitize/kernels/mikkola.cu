#ifndef M_PI
#define M_PI           3.14159265358979323846  /* pi */
#endif

__global__ void mikkola_gpu(const double *manom, const double *ecc, double *eanom){
    /*
    Vectorized C Analtyical Mikkola solver for the eccentric anomaly.
    See: S. Mikkola. 1987. Celestial Mechanics, 40, 329-334.
    Adapted from IDL routine keplereq.pro by Rob De Rosa http://www.lpl.arizona.edu/~bjackson/idl_code/keplereq.pro
    Args:
        manom (double[]): mean anomaly, must be between 0 and pi.
        ecc (double[]): eccentricity
        eanom0 (double[]): array for eccentric anomaly
    Return:
        None: eanom (double[]): is changed by reference
    Written: Devin Cody, 2019
    */

    int i = threadIdx.x + blockIdx.x*blockDim.x;
    double alpha, beta, aux, z, s0, s1, se0, ce0;
    double f, f1, f2, f3, f4, u1, u2, u3;

    alpha = (1.0 - ecc[i]) / ((4.0 * ecc[i]) + 0.5);
    beta = (0.5 * manom[i]) / ((4.0 * ecc[i]) + 0.5);

    aux = sqrt(beta*beta + alpha*alpha*alpha);
    z = pow(fabs(beta + aux), (1.0/3.0));

    s0 = z - (alpha/z);
    s1 = s0 - (0.078*(pow(s0, 5))) / (1.0 + ecc[i]);
    eanom[i] = manom[i] + (ecc[i] * (3.0*s1 - 4.0*(s1*s1*s1)));

    se0=sin(eanom[i]);
    ce0=cos(eanom[i]);

    f  = eanom[i]-ecc[i]*se0-manom[i];
    f1 = 1.0-ecc[i]*ce0;
    f2 = ecc[i]*se0;
    f3 = ecc[i]*ce0;
    f4 = -f2;
    u1 = -f/f1;
    u2 = -f/(f1+0.5*f2*u1);
    u3 = -f/(f1+0.5*f2*u2+(1.0/6.0)*f3*u2*u2);
    eanom[i] += -f/(f1+0.5*f2*u3+(1.0/6.0)*f3*u3*u3+(1.0/24.0)*f4*(u3*u3*u3));
}