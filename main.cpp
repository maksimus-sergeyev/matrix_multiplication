#include "matrix.h"
#include "tmp.h";
#include <omp.h>
#include <time.h>
#include <chrono>
#include <iomanip> 

const bool flagC = 0; // checking for correctness
const bool flag0 = 0; // i, k, j multiplication
const bool flag1 = 0; // block multiplication
const bool flag2 = 1; // parallel block multiplication

const double EPS = 0.001;

int main() 
{
	if (flagC) std::cout << correctness(EPS) << std::endl; // 0 = correctly

	int size = 4000;

	matrix<double> A(size, size), B(size, size), C(size, size), D(size, size), F(size, size);

	srand(time(NULL));

	A.randomfill();
	B.randomfill();

	//-----------------------------------------------------------------------------------------------

	if (flag0) 
	{
		auto begin = std::chrono::steady_clock::now();

		mult(A, B, C);

		auto end = std::chrono::steady_clock::now();

		auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

		std::cout << "The time of 'i, k, j multiplication': " << elapsed_ms.count() << " ms\n";
	}

	//-----------------------------------------------------------------------------------------------

	if (flag1) 
	{
		auto begin = std::chrono::steady_clock::now();

		block_mult(A, B, D, 40, 40);

		auto end = std::chrono::steady_clock::now();

		auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

		std::cout << "The time of 'block multiplication': " << elapsed_ms.count() << " ms\n";
	}

	//-----------------------------------------------------------------------------------------------

	if (flag2) 
	{
		auto begin = std::chrono::steady_clock::now();

		parallel_block_mult(A, B, F, 40, 40);

		auto end = std::chrono::steady_clock::now();

		auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

		std::cout << "The time of 'parallel block multiplication': " << elapsed_ms.count() << " ms\n";
	}
	
}
