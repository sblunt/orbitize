//test_kep.cc
#include <iostream>
#include <cmath>
#include <vector>
#include <thread>

#include "kepler.hh"


void newton(const double manom,
		 	  const double ecc,
		 	  const double tol, 
		 	  const int max_iter, 
		 	  double& eanom){
	/* 
    Scalar C++ Newton-Raphson solver for eccentric anomaly.
    Args:
        manom (double): array of mean anomalies
        ecc (double): array of eccentricities
        eanom (double): array of first guess for eccentric anomaly, same shape as manom (optional)
    Return:
        None: eanom is changed by reference

    Written: Devin Cody, 2018
	*/
	register double diff;
	register int niter = 0;
	register int half_max = max_iter>>1; // divide max_iter by 2 using bit shift
	
	// Let's do one iteration to start with
	eanom -= (eanom - (ecc * sin(eanom)) - manom) / (1.0 - (ecc * cos(eanom)));
	diff = (eanom - (ecc * sin(eanom)) - manom) / (1.0 - (ecc * cos(eanom)));

	while ((std::abs(diff) > tol) && (niter <= max_iter)){
		eanom -= diff;

		// If it hasn't converged after half the iterations are done, try starting from pi
		if (niter == half_max) eanom = M_PI;

		diff = (eanom - (ecc * sin(eanom)) - manom) / (1.0 - (ecc * cos(eanom)));
		niter += 1;
	}

	// If it has not converged, set eccentricity to -1 to signal that it needs to be
	// solved using the analytical version. Note this behavior is a bit differnt from the 
	// numpy implementation
	if (niter >= max_iter){
		std::cout << " " << manom << " " << eanom << " " << diff << " " << ecc << " >= " << max_iter << " iter"<< std::endl;
		eanom = -1;
	}
}



void newton_array(const int start,
					const int end,
					const int threads, 
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

	if (threads <= 1){
		// base case, only one thread, execute immediately
		for (int i = start; i < end; i ++){
			newton(manom[i], ecc[i], tol, max_iter, eanom[i]);
		}	
	} else {
		//More than one thread, recurse.
		// Declare Variables
		std::vector<std::thread> t;
		const int stride = (end - start)/threads;
		const int last_idx = threads-1;

		// Execute code on N-1 Threads
		for (int i = 0; i < last_idx; i++){
			t.push_back(std::thread(newton_array, start + stride*i, start + stride*(i+1)+1,1, manom, ecc, tol, max_iter, eanom));
		}

		// Execute Final part of the array on current thread
		for (int i = start + stride*last_idx; i < start + stride*(last_idx+1)+1; i ++){
			newton(manom[i], ecc[i], tol, max_iter, eanom[i]);
		}

		// Join with the other N-1 Threads
		for (int i = 0; i < last_idx; i++){
			t[i].join();
		}
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

	//test newton array
	newton_array(0, 3, 1,  m, ecc, tol, mi, eanom);
	//start at index 0 goto 3, 1 thread

	return 0;
}


