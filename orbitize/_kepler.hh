//test_kep.hh

# define M_PI           3.14159265358979323846  /* pi */

double newton(const double manom,
		 	  const double ecc,
		 	  const double tol, 
		 	  const int max_iter,
		 	  double eanom);

int newton_array(const int N, 
					const double manom[], 
					const double ecc[], 
					const double tol, 
					const int max_iter, 
					double eanom[]);

