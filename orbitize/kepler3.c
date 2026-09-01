#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI           3.14159265358979323846  /* pi */
#endif

#define PER_CONST 365.2568983840419
#define KV_CONVERSION 29.7846918319 // sqrt(mu/a) where mu = G [in km3/Msun/s2], a = 149597870.700 km
#define MIKKOLA_THRESHOLD 0.4

double positive_mod1(
    double x) {
        return x - floor(x);
        // double value = fmod(x,1.0);
        // if (value < 0.) {
        //     value += 1.0;
        // }
        // return value;
}

double tau_to_manom(
    const double epoch,
    const double period,
    const double tau,
    const double tau_ref_epoch) {
    /*
    Converts epoch and tau orbital parameter to mean anomaly

    Args:
        epoch (double): epoch in the same units as tau, period, and tau_ref_epoch
        period (double): orbital period
        tau (double): fraction of the orbit which is elapsed at the epoch of
            periastron passage relative to the reference epoch (in the range [0,1))
        tau_ref_epoch (double): reference epoch
    
    Return:
        (double): mean anomaly
    */
    double frac_date, mean_anom;
    frac_date = positive_mod1((epoch - tau_ref_epoch)/period);
    mean_anom = positive_mod1((frac_date - tau)) * 2 * M_PI;
    return mean_anom;
}

double newton_solver(
    /* 
    Newton-Raphson solver for eccentric anomaly.

    Args:
        manom (double): mean anomaly
        ecc (double): eccentricity
        tol (double): absolute tolerance at which to stop
        max_iter (int): maximum number of iterations to try
    Return:
        (double): eccentric anomaly or -1.0 if not converged after maximum number of iterations

    Written: Devin Cody, 2018
    */
    const double manom, 
    const double ecc, 
    const double tol, 
    const int max_iter) {
    double diff;
    int niter = 0;
    double eanom = manom;
    
    // Let's do one iteration to start with
    eanom -= (eanom - (ecc * sin(eanom)) - manom) / (1.0 - (ecc * cos(eanom)));
    diff = (eanom - (ecc * sin(eanom)) - manom) / (1.0 - (ecc * cos(eanom)));

    while ((fabs(diff) > tol) && (niter <= max_iter)){
        eanom -= diff;
        diff = (eanom - (ecc * sin(eanom)) - manom) / (1.0 - (ecc * cos(eanom)));
        niter += 1;
    }

    // If it has not converged, set eccentricity to -1 to signal that it needs to be
    // solved using the analytical version. Note this behavior is a bit different from the 
    // numpy implementation
    if (niter >= max_iter){
        // printf("%f %f %f %f >= %d iter\n", manom, eanom, diff, ecc, max_iter);
        eanom = -1.0;
    }
    return eanom;
}

double mikkola_solver(const double manom, const double ecc) {
    /*
    Analtyical Mikkola solver for the eccentric anomaly.
    See: S. Mikkola. 1987. Celestial Mechanics, 40, 329-334.
    Adapted from IDL routine keplereq.pro by Rob De Rosa http://www.lpl.arizona.edu/~bjackson/idl_code/keplereq.pro

    Args:
        manom (double): mean anomaly, must be between 0 and pi.
        ecc (double): eccentricity
    Return:
        (double): eccentric anomaly

    Written: Devin Cody, 2019
    */
    double eanom;
    double alpha, beta, aux, z, s0, s1, se0, ce0;
    double f, f1, f2, f3, f4, u1, u2, u3;

    alpha = (1.0 - ecc) / ((4.0 * ecc) + 0.5);
    beta = (0.5 * manom) / ((4.0 * ecc) + 0.5);

    aux = sqrt(beta*beta + alpha*alpha*alpha);
    z = cbrt(fabs(beta + aux));

    s0 = z - (alpha/z);
    s1 = s0 - (0.078*(pow(s0, 5))) / (1.0 + ecc);
    eanom = manom + (ecc * (3.0*s1 - 4.0*(s1*s1*s1)));

    se0=sin(eanom);
    ce0=cos(eanom);

    f  = eanom-ecc*se0-manom;
    f1 = 1.0-ecc*ce0;
    f2 = ecc*se0;
    f3 = ecc*ce0;
    f4 = -f2;
    u1 = -f/f1;
    u2 = -f/(f1+0.5*f2*u1);
    u3 = -f/(f1+0.5*f2*u2+(1.0/6.0)*f3*u2*u2);
    eanom += -f/(f1+0.5*f2*u3+(1.0/6.0)*f3*u3*u3+(1.0/24.0)*f4*(u3*u3*u3));
    
    return eanom;
}

double calc_ecc_anom(
    const double manom,
    const double ecc,
    const double tol,
    const int max_iter) {
    /*
    Computes the eccentric anomaly from the mean anomaly.
    e < MIKKOLA_THRESHOLD: use Newton solver, e >= MIKKOLA_THRESHOLD: use Mikkola solver

    Args:
        manom (double): mean anomaly
        ecc (double): eccentricity
        tol (double): absolute tolerance of iterative computation
        max_iter (int): maximum number of iterations before switching
    
    Return:
        eanom (double): eccentric anomaly
    
    Written: Eshel Dror, 2026 (based on Python implementation by Jason Wang, 2018)
    */
    double eanom = 0.0;
    if (ecc == 0.0) {
        return manom;
    }
    if (ecc < MIKKOLA_THRESHOLD) {
        eanom = newton_solver(manom, ecc, tol, max_iter);
    }
    if (ecc >= MIKKOLA_THRESHOLD || eanom == -1.0) {
        if (manom > M_PI) {
            eanom = 2. * M_PI - mikkola_solver(2. * M_PI - manom, ecc);
        } else {
            eanom = mikkola_solver(manom, ecc);
        }
    }
    return eanom;
}

void calc_ecc_anom_array(
    const int size,
    const double manom[],
    const double ecc[],
    const double tol,
    const int max_iter,
    double eanom[]) {
    /*
    Computes an array of eccentric anomalies from the mean anomalies.
    e < MIKKOLA_THRESHOLD: use Newton solver, e >= MIKKOLA_THRESHOLD: use Mikkola solver

    Args:
        size (int): size of manom and ecc
        manom (double[]): mean anomalies
        ecc (double[]): eccentricities
        tol (double): absolute tolerance of iterative computation
        max_iter (int): maximum number of iterations before switching
        eanom (double[]): array to update with eccentric anomalies
    
    Return:
        None (updates eanom with eccentric anomalies)
    */
    int i;
    for (i = 0; i < size; i++) {
        eanom[i] = calc_ecc_anom(manom[i], ecc[i], tol, max_iter);
    }
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
    double vz[]) {
    /*
    Calculates the right ascension offsets, declination offset, and radial velocities of the body given array of
    orbital parameters (size n_orbits) at given epochs (array of size n_epochs) solved in c

    Based on orbit solvers from James Graham and Rob De Rosa.
    Adapted by Jason Wang and Henry Ngo.
    Converted and optimized in C by Eshel Dror.

    Args:
        n_orbits (int): length of orbital parameter arrays
        n_epochs (int): length of epochs array
        epochs (double[]): MJD times for which we want the positions of the planet
        sma (double[]): semi-major axis of orbit [au]
        ecc (double[]): eccentricity of the orbit [0,1]
        inc (double[]): inclination [radians]
        aop (double[]): argument of periastron [radians]
        pan (double[]): longitude of the ascending node [radians]
        tau (double[]): epoch of periastron passage in fraction of orbital period past MJD=0 [0,1]
        plx (double[]): parallax [mas]
        mtot (double[]): total mass of the two-body orbit (M_* + M_planet) [Solar masses]
        mass_for_Kamp (double[]): mass of the body that causes the RV signal.
            For example, if you want to return the stellar RV, this is the planet mass.
            If you want to return the planetary RV, this is the stellar mass. [Solar masses].
            For planet mass ~ 0, mass_for_Kamp ~ M_tot, and function returns planetary RV.
        tau_ref_epoch (double): reference date that tau is defined with respect to (i.e., tau=0)
        tolerance (double): absolute tolerance of iterative computation
        max_iter (int): maximum number of iterations before switching
        raoff (double[]): array of length (n_dates x n_orbs) to update with RA offsets between the bodies [mas]
        deoff (double[]): array of length (n_dates x n_orbs) to update with Dec offsets between the bodies [mas]
        vz (double[]): array of length (n_dates x n_orbs) to update with radial
            velocities of one of the bodies according to mass_for_Kamp [km/s]

    Return:
        None (updates raoff, deoff, and vz with the respective values, with orbit i and epoch j at position i * n_epochs + j)

    Written: Eshel Dror, 2026
    */
    int i, j, k;
    double period, manom, eanom, partial_tanom, radius, c2i2, s2i2, c1, c2, s1, s2, rad_plx, Kv;
    // double tanom, arg1, arg2;
    double ecc_cos_aop, cos_aop, sin_aop;
    double cos_p1, sin_p1, cos_p2, sin_p2, c_tanom, s_tanom;
    double a, b, b_squared, c, c_squared;

    for (i = 0; i < n_orbits; i ++) {
        period = sqrt(
            pow(sma[i], 3.0) / (mtot[i])
        ) * PER_CONST;
        
        c2i2 = pow(cos(0.5*inc[i]),2);
        s2i2 = 1.0 - c2i2; // s2i2 = pow(sin(0.5*inc[i]),2);
        
        partial_tanom = sqrt((1.0 + ecc[i])/(1.0 - ecc[i]));
        
        Kv = sqrt(1 / (1.0 - pow(ecc[i],2))) * (mass_for_Kamp[i] *
                                               sin(inc[i])) / sqrt(mtot[i]) / sqrt(sma[i]);
        Kv *= KV_CONVERSION;
        
        cos_aop = cos(aop[i]), sin_aop = sin(aop[i]);
        ecc_cos_aop = ecc[i]*cos_aop;

        cos_p1 = cos(aop[i] + pan[i]), sin_p1 = sin(aop[i] + pan[i]);
        cos_p2 = cos(aop[i] - pan[i]), sin_p2 = sin(aop[i] - pan[i]);

        for (j = 0; j < n_epochs; j++) {
            k = i * n_epochs + j;
            manom = tau_to_manom(epochs[j], period, tau[i], tau_ref_epoch);
            eanom = calc_ecc_anom(manom, ecc[i], tolerance, max_iter);
            
            // tanom = 2.0*atan(partial_tanom*tan(0.5*eanom));
            // c_tanom = cos(tanom), s_tanom = sin(tanom);
            c = partial_tanom * tan(0.5*eanom);
            c_squared = c * c;
            // c = cos(atan(c))
            a = 1 / sqrt(c_squared + 1);
            // b = sin(atan(c))
            b = c * a;
            b_squared = b * b;
            // c_tanom = cos(2*atan(c)) = 1 - 2 * sin^2(atan(c))
            c_tanom = 1.0 - 2.0 * b_squared;
            // s_tan = sin(2*atan(c)) = 2 * sin(atan(c)) * cos(atan(c))
            s_tanom = 2.0 * b * a;
            
            // arg1 = tanom + aop[i] + pan[i];
            // arg2 = tanom + aop[i] - pan[i];
            // c1 = cos(arg1);
            // c2 = cos(arg2);
            // s1 = sin(arg1);
            // s2 = sin(arg2);
            c1 = c_tanom * cos_p1 - s_tanom * sin_p1;
            s1 = s_tanom * cos_p1 + c_tanom * sin_p1;
            c2 = c_tanom * cos_p2 - s_tanom * sin_p2;
            s2 = s_tanom * cos_p2 + c_tanom * sin_p2;
            
            radius = sma[i] * (1.0 - ecc[i] * cos(eanom));
            rad_plx = radius * plx[i];
            raoff[k] = rad_plx * (c2i2 * s1 - s2i2 * s2);
            deoff[k] = rad_plx * (c2i2 * c1 + s2i2 * c2);

            // vz[k] = Kv * (ecc[i] * cos(aop[i]) + cos(aop[i] + tanom));
            vz[k] = Kv * (ecc_cos_aop + (cos_aop * c_tanom - sin_aop * s_tanom));
        }
    }
}
