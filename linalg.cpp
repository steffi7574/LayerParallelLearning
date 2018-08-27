void matvec(int dimN,
            double* H, 
            double* x,
            double* Hx)
{
    double sum_j;

    for (int i=0; i<dimN; i++)
    {
       sum_j = 0.0;
       for (int j=0; j<dimN; j++)
       {
          sum_j +=  H[i*dimN+j] * x[j];
       }
       Hx[i] = sum_j;
    } 
}                           



double vecdot(int     dimN,
              double* x,
              double* y)
{
   double dotprod = 0.0;
   for (int i = 0; i < dimN; i++)
   {
      dotprod += x[i] * y[i];
   }
   
   return dotprod;
}
