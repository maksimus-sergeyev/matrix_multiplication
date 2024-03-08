#include "matrix.h"
#include "tmp.cpp"
#include <omp.h>
#include <time.h>
#include <chrono>
#include <iomanip> 
#include <thread>
#include "mkl.h"

/*
CPU:
	11th Gen Intel(R) Core(TM) i7-11700F @ 2.50GHz
	Base speed:	2,50 GHz
	Sockets:	1
	Cores:	8
	Logical processors:	16
	Virtualization:	Enabled
	L1 cache:	640 KB
	L2 cache:	4,0 MB
	L3 cache:	16,0 MB

	//micro architecture - rocket lake

options: /GS /TP /Qiopenmp /W3 /QxHost /Gy /Zc:wchar_t /Zi /O3 /D "NDEBUG" /D "_CONSOLE" /D "__INTEL_LLVM_COMPILER=20230200" /D "_UNICODE" /D "UNICODE" /Qipo /Zc:forScope /Oi /MD /Fa"x64\Release\" /EHsc /nologo /Fo"x64\Release\" /FA /Ot /Fp"x64\Release\matrix_multiplication.pch"   /Qopt-zmm-usage=high 

*/

const bool flagC = 0; // checking for correctness
const bool flag0 = 0; // i, k, j multiplication
const bool flag1 = 0; // parallel i, k, j multiplication
const bool flag2 = 0; // block multiplication
const bool flag3 = 0; // parallel block multiplication

const bool flag4 = 0;	// parallel block multiplication2 (with trans. block 2nd matrix)					
						//size = 8192;  best_time = 6900 ms; median_time = 7700 ms; 

const bool flag5 = 0;	// parallel block multiplication3 (with subblock)
						//size = 8192;  best_time = 13000 ms; median_time = 14600 ms;

const bool flag6 = 0;	// parallel block multiplication4 (with trans. block 2nd matrix && subblock)
						//size = 8192;	best_time = 13000 ms; median_time = 14600 ms;

const bool flag7 = 1;	// parallel block multiplication5 (parallel block multiplication3 with intrinsics)
						//size = 8256;	best_time = 3750 ms; median_time = 4400 ms;  
						//size = 8064;  best_time = 4000 ms; median_time = 4300 ms;
						 //(sizes divisible by block_size)

						//size = 8192 	best_time = 9300 ms; median_time = 9800 ms;
						//size = 8400 	time = 5900 ms;
						//size = 8300 	time = 4900 ms;
						//size = 8200 	time = 5400 ms;

const bool flag8 = 1;	// parallel block multiplication6 (parallel block multiplication5 with other borders policy)
						//size = 8256;	best_time = 3900 ms; median_time = 4200 ms;  
						//size = 8064;  best_time = 3650 ms; median_time = 4100 ms;
						 //(sizes divisible by block_size)

						//size = 8192 	best_time = 5200 ms; median_time = 5800 ms;
						//size = 8400 	time = 4700 ms;
						//size = 8300 	time = 4850 ms;
						//size = 8200 	time = 5000 ms;
														
																			
const bool flag9 = 1;	//MKL cblas_dgemm(1, 0) = (1 * A * B + 0 * C) = A * B
						//size = 8256; best_time = 3300 ms; median_time = 3600 ms;
						//size = 8064; best_time = 2950 ms; median_time = 3300 ms;
						//size = 8192; best_time = 3100 ms; median_time = 3500 ms;
						
						


const double EPS = 0.001;

int main() 
{
	int size = 4032;
	
	matrix<double> A(size, size), B(size, size), C(size, size);

	srand(time(NULL));

	if (flag0) 
	{
		if (flagC) std::cout << mult_correctness(EPS) << std::endl;

		A.randomfill();
		B.randomfill();

		auto begin = std::chrono::steady_clock::now();

		mult(A, B, C);

		auto end = std::chrono::steady_clock::now();

		auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

		std::cout << "The time of 'i, k, j multiplication': " << elapsed_ms.count() << " ms\n";
	}

	if (flag1)
	{
		if (flagC) std::cout << parallel_mult_correctness(EPS) << std::endl;

		A.randomfill();
		B.randomfill();

		auto begin = std::chrono::steady_clock::now();

		parallel_mult(A, B, C);

		auto end = std::chrono::steady_clock::now();

		auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

		std::cout << "The time of 'parallel i, k, j multiplication': " << elapsed_ms.count() << " ms\n";
	}

	if (flag2) 
	{
		if (flagC) std::cout << block_mult_correctness(EPS) << std::endl;

		A.randomfill();
		B.randomfill();

		auto begin = std::chrono::steady_clock::now();

		block_mult(A, B, C);

		auto end = std::chrono::steady_clock::now();

		auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

		std::cout << "The time of 'block multiplication': " << elapsed_ms.count() << " ms\n";
	}

	if (flag3) 
	{
		if (flagC) std::cout << parallel_block_mult_correctness(EPS) << std::endl;

		A.randomfill();
		B.randomfill();

		auto begin = std::chrono::steady_clock::now();

		parallel_block_mult(A, B, C);

		auto end = std::chrono::steady_clock::now();

		auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

		std::cout << "The time of 'parallel block multiplication': " << elapsed_ms.count() << " ms\n";
	}

	if (flag4)
	{
		if (flagC) std::cout << parallel_block_mult2_correctness(EPS) << std::endl;

		A.randomfill();
		B.randomfill();

		auto begin = std::chrono::steady_clock::now();

		parallel_block_mult2(A, B, C);

		auto end = std::chrono::steady_clock::now();

		auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

		std::cout << "The time of 'parallel block multiplication2': " << elapsed_ms.count() << " ms\n";
	}
	
	if (flag5)
	{
		if (flagC) std::cout << parallel_block_mult3_correctness(EPS) << std::endl;

		A.randomfill();
		B.randomfill();

		auto begin = std::chrono::steady_clock::now();

		parallel_block_mult3(A, B, C);

		auto end = std::chrono::steady_clock::now();

		auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

		std::cout << "The time of 'parallel block multiplication3': " << elapsed_ms.count() << " ms\n";

	}

	if (flag6)
	{
		if (flagC) std::cout << parallel_block_mult4_correctness(EPS) << std::endl;

		A.randomfill();
		B.randomfill();

		auto begin = std::chrono::steady_clock::now();

		parallel_block_mult4(A, B, C);

		auto end = std::chrono::steady_clock::now();

		auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

		std::cout << "The time of 'parallel block multiplication4': " << elapsed_ms.count() << " ms\n";
	}

	if (flag7)
	{
		if (flagC) std::cout << parallel_block_mult5_correctness(EPS) << std::endl;

		A.randomfill();
		B.randomfill();

		auto begin = std::chrono::steady_clock::now();

		parallel_block_mult5(A, B, C);

		auto end = std::chrono::steady_clock::now();

		auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

		std::cout << "The time of 'parallel block multiplication5': " << elapsed_ms.count() << " ms\n";
	}

	if (flag8)
	{
		if (flagC) std::cout << parallel_block_mult6_correctness(EPS) << std::endl;

		A.randomfill();
		B.randomfill();

		auto begin = std::chrono::steady_clock::now();

		parallel_block_mult6(A, B, C);

		auto end = std::chrono::steady_clock::now();

		auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

		std::cout << "The time of 'parallel block multiplication6': " << elapsed_ms.count() << " ms\n";
	}
	
	if (flag9)
	{
		srand(time(NULL));

		double* a, * b, * c, alpha, beta;
		MKL_INT m, n, k;
		MKL_INT sizea, sizeb, sizec;

		alpha = 1.0;
		beta = 0.0;

		m = size;
		n = size;
		k = size;

		sizea = m * k;
		sizeb = k * n;
		sizec = m * n;

		a = (double*)mkl_malloc(sizea * sizeof(double), 64);
		b = (double*)mkl_malloc(sizeb * sizeof(double), 64);
		c = (double*)mkl_malloc(sizec * sizeof(double), 64);

		for (MKL_INT i = 0; i < sizea; i++)
			a[i] = rand() / (double)RAND_MAX - .5;

		for (MKL_INT i = 0; i < sizeb; i++)
			b[i] = rand() / (double)RAND_MAX - .5;

		for (MKL_INT i = 0; i < sizec; i++)
			c[i] = 0;

		auto begin = std::chrono::steady_clock::now();

		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, k, b, n, beta, c, n);
		
		auto end = std::chrono::steady_clock::now();

		auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

		std::cout << "The time of 'mkl dgemm(1, 0)': " << elapsed_ms.count() << " ms\n";

		mkl_free(a);
		mkl_free(b);
		mkl_free(c);
	}
	

	return 0;
}