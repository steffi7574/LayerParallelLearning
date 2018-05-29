/* Set matrix to identity */
int 
set_identity(int N,
             double* A);

/* Return prd = x^Ty */
double
vecT_vec_product(int     N,
                 double *x, 
                 double *y);
              
/* Return A = xy^T */
int 
vec_vecT_product(int N,
                 double *x,
                 double *y,
                 double *A);

/* Return y = Ax */
int
mat_vec_product(int N,
                double *A,
                double *x,
                double *y);

/* Return y = x^TA */
int
vecT_mat_product(int N,
                 double *x,
                 double *A,
                 double *y);
                 
/**
 * Return bfgs update of Hk 
 * Compare Nocedal & Wright: Numerical Optimization, Chapter 6.1, and Sherman-Morrison formula for the inverse
 */
int
bfgs_update(int N,
            double *sk,
            double *yk,
            double *Hk);