//test_kep.cc
#include <iostream>
#include <cmath>
// #include <vector>
// #include <future>

#include "_kepler.hh"


double newton(const double manom,
		 	  const double ecc,
		 	  const double tol = 1e-9, 
		 	  const int max_iter = 100, 
		 	  double eanom = -1)
{
//	double eanom;
	double diff;
	int niter = 0;

//	if (eanom0 == -1){
//		eanom = manom;
//	} else {
//		eanom = eanom0;
//	}

	// std::cout << "eanom is " << eanom << std::endl;
	
	eanom -= (eanom - (ecc * sin(eanom)) - manom) / (1.0 - (ecc * cos(eanom)));
	diff = (eanom - (ecc * sin(eanom)) - manom) / (1.0 - (ecc * cos(eanom)));

	while ((std::abs(diff) > tol) && (niter <= max_iter)){
		eanom -= diff;

		if (niter == (max_iter/2)) {
			eanom = M_PI;
		}
		diff = (eanom - (ecc * sin(eanom)) - manom) / (1.0 - (ecc * cos(eanom)));
		niter += 1;
	}

	// std::cout << "eanom is " << eanom << std::endl;
	// std::cout << "diff is " << diff << std::endl;
	// std::cout << "niter is " << niter << std::endl;

	return eanom;
}

int newton_array(const int N, 
					const double manom[], 
					const double ecc[], 
					const double tol, 
					const int max_iter, 
					double eanom[])
{
	for (int i = 0; i < N; i ++){
		// std::cout << "i is " << i << std::endl;
		eanom[i] =  newton(manom[i], ecc[i], tol, max_iter, eanom[i]);
		// std::cout << "eanom[i] is " << eanom[i] << std::endl;
	}

	return N;
}
		

int main(){
	double m[] = {.5, 1, 1.5};
	double ecc[] = {.25, .75, .83};
	double tol = 1e-9;
	int mi = 100;
	double eanom[] = {0, 0, 0};


	newton_array(3, m, ecc, tol, mi, eanom);
	return 0;
}


