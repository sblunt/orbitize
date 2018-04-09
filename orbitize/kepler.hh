//test_kep.hh
#include <vector>

# define M_PI           3.14159265358979323846  /* pi */

void newton(const double manom,
		 	  const double ecc,
		 	  const double tol, 
		 	  const int max_iter,
		 	  double &eanom);


void newton_array(const int start,
					const int end,
					const int threads, 
					const double manom[], 
					const double ecc[], 
					const double tol, 
					const int max_iter, 
					double eanom[]);