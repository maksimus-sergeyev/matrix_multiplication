#include "matrix.h"
#include "tmp.cpp";
#include <omp.h>
#include <time.h>
#include <chrono>
#include <iomanip> 

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
ICC
options: /permissive /GS /Qopenmp /Qftz /W3 /QxHost /Gy /Zc:wchar_t /Zi /O3 /Fd"x64\Release\vc143.pdb" /Zc:inline /Quse-intel-optimized-headers /D "NDEBUG" /D "_CONSOLE" /D "_UNICODE" /D "UNICODE" /Qipo /Zc:forScope /Oi /MD /FC /Fa"x64\Release\" /EHsc /nologo /Qparallel /Fo"x64\Release\" /Ot /Fp"x64\Release\matrix_multiplication.pch" 

*/

const bool flagC = 0; // checking for correctness
const bool flag0 = 0; // i, k, j multiplication
const bool flag1 = 0; // parallel i, k, j multiplication
const bool flag2 = 0; // block multiplication
const bool flag3 = 1; // parallel block multiplication

const double EPS = 0.001;

int main() 
{
	if (flagC) std::cout << correctness(EPS) << std::endl; // 0 = correctly

	int size = 8192;

	matrix<double> A(size, size), B(size, size), C(size, size);

	srand(time(NULL));

	//-----------------------------------------------------------------------------------------------

	if (flag0) 
	{
		A.randomfill();
		B.randomfill();

		auto begin = std::chrono::steady_clock::now();

		mult(A, B, C);

		auto end = std::chrono::steady_clock::now();

		auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

		std::cout << "The time of 'i, k, j multiplication': " << elapsed_ms.count() << " ms\n";
	}

	//-----------------------------------------------------------------------------------------------

	if (flag1)
	{
		A.randomfill();
		B.randomfill();

		auto begin = std::chrono::steady_clock::now();

		parallel_mult(A, B, C);

		auto end = std::chrono::steady_clock::now();

		auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

		std::cout << "The time of 'parallel i, k, j multiplication': " << elapsed_ms.count() << " ms\n";
	}

	//-----------------------------------------------------------------------------------------------

	if (flag2) 
	{
		A.randomfill();
		B.randomfill();

		auto begin = std::chrono::steady_clock::now();

		block_mult(A, B, C);

		auto end = std::chrono::steady_clock::now();

		auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

		std::cout << "The time of 'block multiplication': " << elapsed_ms.count() << " ms\n";
	}

	//-----------------------------------------------------------------------------------------------

	if (flag3) 
	{
		A.randomfill();
		B.randomfill();

		auto begin = std::chrono::steady_clock::now();

		parallel_block_mult(A, B, C);

		auto end = std::chrono::steady_clock::now();

		auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

		std::cout << "The time of 'parallel block multiplication': " << elapsed_ms.count() << " ms\n";
	}
}
