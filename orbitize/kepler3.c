#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI           3.14159265358979323846  /* pi */
#endif

#define PERIOD_CONVERSION 2105.6131276363703
#define G 6.6743e-11
#define KV_CONVERSION 2.742898587511721e-07

double tau_to_manom(
    const double epoch,
    const double sma,
    const double mtot,
    const double tau,
    const double tau_ref_epoch) {
        double period, frac_date, mean_anom;
        period = sqrt(
            4 * pow(M_PI, 2.0) * pow(sma, 3.0) / (G * mtot)
        ) / KV_CONVERSION;
        frac_date = fmod((epoch - tau_ref_epoch)/period, 1.0);
        mean_anom = fmod((frac_date - tau), 1.0) * 2 * M_PI;
        return mean_anom;
}

void calc_orbit(
    const int n_orbits,
    const int n_epochs,
    const double epochs[],
    const double sma[],
    const double ecc[],
    const double inc[],
    const double aop[],
    const double pan[],
    const double tau[],
    const double plx[],
    const double mtot[],
    const double mass_for_Kamp[],
    const double tau_ref_epoch,
    const double tolerance,
    const int max_iter,
    double raoff[],
    double deoff[],
    double vz[]){
    int i, j, k;
    double manom, eanom, tanom, radius, c2i2, s2i2, arg1, arg2, c1, c2, s1, s2, Kv;
    for (i = 0; i < n_orbits; i ++) {
        for (j = 0; j < n_epochs; j ++) {
            k = i * n_epochs + j;
            manom = tau_to_manom(epochs[j], sma[i], mtot[i], tau[i], tau_ref_epoch);
            raoff[k] = epochs[j] + sma[i] + ecc[i] + inc[i] + aop[i] + pan[i] + tau[i] + plx[i] + mtot[i] + mass_for_Kamp[i] + tau_ref_epoch;
            deoff[k] = epochs[j] + sma[i] + ecc[i] + inc[i] + aop[i] + pan[i] + tau[i] + plx[i] + mtot[i] + mass_for_Kamp[i] + tau_ref_epoch;
            vz[k] = epochs[j] + sma[i] + ecc[i] + inc[i] + aop[i] + pan[i] + tau[i] + plx[i] + mtot[i] + mass_for_Kamp[i] + tau_ref_epoch;
        }
    }
}

void newton_array(const int n_elements,
                    const double manom[], 
                    const double ecc[], 
                    const double tol, 
                    const int max_iter, 
                    double eanom[]){
    /* 
    Vectorized C Newton-Raphson solver for eccentric anomaly.

    Args:
        manom (double[]): array of mean anomalies
        ecc (double[]): array of eccentricities
        eanom0 (double[]): array of first guess for eccentric anomaly, same shape as manom (optional)
    Return:
        None: eanom (double[]): is changed by reference

    Written: Devin Cody, 2018
    */
    int i;
    for (i = 0; i < n_elements; i ++){
        double diff;
        int niter = 0;
        int half_max = max_iter/2.0; // divide max_iter by 2 using bit shift
        
        // Let's do one iteration to start with
        eanom[i] -= (eanom[i] - (ecc[i] * sin(eanom[i])) - manom[i]) / (1.0 - (ecc[i] * cos(eanom[i])));
        diff = (eanom[i] - (ecc[i] * sin(eanom[i])) - manom[i]) / (1.0 - (ecc[i] * cos(eanom[i])));

        while ((fabs(diff) > tol) && (niter <= max_iter)){
            eanom[i] -= diff;

            // If it hasn't converged after half the iterations are done, try starting from pi
            if (niter == half_max) {
                eanom[i] = M_PI;
            }

            diff = (eanom[i] - (ecc[i] * sin(eanom[i])) - manom[i]) / (1.0 - (ecc[i] * cos(eanom[i])));
            niter += 1;
        }

        // If it has not converged, set eccentricity to -1 to signal that it needs to be
        // solved using the analytical version. Note this behavior is a bit different from the 
        // numpy implementation
        if (niter >= max_iter){
            printf("%f %f %f %f >= %d iter\n", manom[i], eanom[i], diff, ecc[i], max_iter);
            eanom[i] = -1;
        }
    }   
}


void mikkola_array(const int n_elements, const double manom[], const double ecc[], double eanom[]){
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

    int i;
    double alpha, beta, aux, z, s0, s1, se0, ce0;
    double f, f1, f2, f3, f4, u1, u2, u3;

    for (i = 0; i < n_elements; i++){
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
}


int main(void){
    // Test functions with a small program

    // Define variables for newton array method
    double m[] = {.5, 1, 1.5};
    double ecc[] = {.25, .75, .83};
    double tol = 1e-9;
    int mi = 100;
    double eanom[] = {0, 0, 0};

    // test newton_array
    // Answer should be: [ 0.65161852,  1.73936894,  2.18046524])

    newton_array(3,  m, ecc, tol, mi, eanom);
    int i;
    for (i = 0; i < 3; i++){
        printf("eanom[%d] = %f\n", i, eanom[i]);
        eanom[i] = 0;
    }

    // test mikkola_array
    // Answer should be: [ 0.65161852,  1.73936894,  2.18046524])

    mikkola_array(3, m, ecc, eanom);
    for (i = 0; i < 3; i++){
        printf("eanom[%d] = %f\n", i, eanom[i]);
    }

    return 0;
}