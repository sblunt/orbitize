
#ifndef M_PI
#define M_PI           3.14159265358979323846  /* pi */
#endif


__global__ void newton_gpu(const double *manom, 
            const double *ecc, 
            double *eanom,
            const int *max_iter,
            const double *tol){
    /* 
    Vectorized C++ Newton-Raphson solver for eccentric anomaly.
    Args:
        manom (double[]): array of mean anomalies
        ecc (double[]): array of eccentricities
        eanom0 (double[]): array of first guess for eccentric anomaly, same shape as manom (optional)
    Return:
        None: eanom is changed by reference
    Written: Devin Cody, 2018
    */
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    double diff;
    int niter = 0;
    int half_max = *max_iter/2.0; // divide convergence->max_iter by 2 using bit shift
    
    // Let's do one iteration to start with
    eanom[i] -= (eanom[i] - (ecc[i] * sin(eanom[i])) - manom[i]) / (1.0 - (ecc[i] * cos(eanom[i])));
    diff = (eanom[i] - (ecc[i] * sin(eanom[i])) - manom[i]) / (1.0 - (ecc[i] * cos(eanom[i])));

    while ((fabs(diff) > *tol) && (niter <= *max_iter)){
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
    if (niter >= *max_iter){
        printf("%f %f %f %f >= %d iter\n", manom[i], eanom[i], diff, ecc[i], *max_iter);
        eanom[i] = -1;
    }
}
