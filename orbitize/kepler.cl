// #include <stdio.h>
// #include <math.h>

#ifndef M_PI
#define M_PI           3.14159265358979323846  /* pi */
#endif

typedef struct Params{
	double tol;
	double max_iter;
} Params;


__kernel void newton_gpu(__global const double *manom, 
							__global const double *ecc, 
							__global double *eanom,
							__constant struct Params* convergence){
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
	int i = get_global_id(0);
	double diff;
	int niter = 0;
	int half_max = convergence->max_iter/2.0; // divide convergence->max_iter by 2 using bit shift
	
	// Let's do one iteration to start with
	eanom[i] -= (eanom[i] - (ecc[i] * native_sin(eanom[i])) - manom[i]) / (1.0 - (ecc[i] * native_cos(eanom[i])));
	diff = (eanom[i] - (ecc[i] * native_sin(eanom[i])) - manom[i]) / (1.0 - (ecc[i] * native_cos(eanom[i])));

	while ((fabs(diff) > convergence->tol) && (niter <= convergence->max_iter)){
		eanom[i] -= diff;

		// If it hasn't converged after half the iterations are done, try starting from pi
		if (niter == half_max) {
			eanom[i] = M_PI;
		}

		diff = (eanom[i] - (ecc[i] * native_sin(eanom[i])) - manom[i]) / (1.0 - (ecc[i] * native_cos(eanom[i])));
		niter += 1;
	}

	// If it has not converged, set eccentricity to -1 to signal that it needs to be
	// solved using the analytical version. Note this behavior is a bit different from the 
	// numpy implementation
	if (niter >= convergence->max_iter){
		printf("%f %f %f %f >= %d iter\n", manom[i], eanom[i], diff, ecc[i], convergence->max_iter);
		eanom[i] = -1;
	}
}