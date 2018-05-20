#include <stdio.h>
#include "cblas.h"

//http: //www.netlib.org/lapack/

/*        

[20 5] * [2 1]
         [1 4]  = [45 40]

*/

int main()
{
    const int dim = 2;
    double a[4] = {1.0, 1.0, 1.0, 1.0};
    double b[4] = {2.0, 2.0, 2.0, 2.0};
    double c[4] = {0.0, };

    int m = dim, n = dim, k = dim, lda = dim, ldb = dim, ldc = dim;

    double al=1.0;
    double be=0.0;

/*   
    void cblas_dgemm (  
       const enum CBLAS_ORDER Order,        //Specifies row-major (C) or column-major (Fortran) data ordering.
       const enum CBLAS_TRANSPOSE TransA,   //Specifies whether to transpose matrix A.
       const enum CBLAS_TRANSPOSE TransB,   //Specifies whether to transpose matrix B.
       const int M,                         //Number of rows in matrices A and C.
       const int N,                         //Number of columns in matrices B and C.
       const int K,                         //Number of columns in matrix A; number of rows in matrix B.

       const double alpha,                  //Scaling factor for the product of matrices A and B.
       const double *A,                     //Matrix A.
       const int lda,                       //The size of the first dimention of matrix A;
                                            //if you are passing a matrix A[m][n], the value should be m.
       const double *B,
       const int ldb,  

       const double beta,  
       double *C,  
       const int ldc  
    );  

    简单来说，这个函数的作用就是： C = alpha*AB + beta*C, 其中M、N分别是C的num_rows 和 num_cols, 
    详细信息可以参考 https://developer.apple.com/documentation/accelerate/1513282-cblas_dgemm
    
 */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, al, a, lda, b, ldb, be, c, ldc);

    printf("the matrix c is: \n%f,%f\n%f,%f\n", c[0], c[1], c[2], c[3]);

    return 0;
}
