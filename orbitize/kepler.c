#include <stdio.h>
#include <math.h>

# define M_PI           3.14159265358979323846  /* pi */

void newton(const double manom,
		 	  const double ecc,
		 	  const double tol, 
		 	  const int max_iter, 
		 	  double *eanom){
	/* 
    Scalar C Newton-Raphson solver for eccentric anomaly.

    Args:
        manom (double): array of mean anomalies
        ecc (double): array of eccentricities
        eanom (double): array of first guess for eccentric anomaly, same shape as manom (optional)
    Return:
        None: eanom is changed by reference

    Written: Devin Cody, 2018
	*/
	double diff;
	int niter = 0;
	int half_max = max_iter/2.0; // divide max_iter by 2 using bit shift
	
	// Let's do one iteration to start with
	*eanom -= (*eanom - (ecc * sin(*eanom)) - manom) / (1.0 - (ecc * cos(*eanom)));
	diff = (*eanom - (ecc * sin(*eanom)) - manom) / (1.0 - (ecc * cos(*eanom)));

	while ((fabs(diff) > tol) && (niter <= max_iter)){
		*eanom -= diff;

		// If it hasn't converged after half the iterations are done, try starting from pi
		if (niter == half_max) {
			*eanom = M_PI;
		}

		diff = (*eanom - (ecc * sin(*eanom)) - manom) / (1.0 - (ecc * cos(*eanom)));
		niter += 1;
	}

	// If it has not converged, set eccentricity to -1 to signal that it needs to be
	// solved using the analytical version. Note this behavior is a bit differnt from the 
	// numpy implementation
	if (niter >= max_iter){
		printf("%f %f %f %f >= %d iter\n", manom, *eanom, diff, ecc, max_iter);// << manom << " " << *eanom << " " << diff << " " << ecc << " >= " << max_iter << " iter"<< std::endl;
		*eanom = -1;
	}
}



void newton_array(const int n_elements,
					const double manom[], 
					const double ecc[], 
					const double tol, 
					const int max_iter, 
					double eanom[]){
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

	for (int i = 0; i < n_elements; i ++){
		newton(manom[i], ecc[i], tol, max_iter, &(eanom[i]));
	}	
}

int main(){
	// Test functions with a small program

	//Define variables for newton array method
	double m[] = {.5, 1, 1.5};
	double ecc[] = {.25, .75, .83};
	double tol = 1e-9;
	int mi = 100;
	double eanom[] = {0, 0, 0};

	//test newton_array
	newton_array(3,  m, ecc, tol, mi, eanom);
	//Answer should be: [ 0.65161852,  1.73936894,  2.18046524])

	return 0;
}