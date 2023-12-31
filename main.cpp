#include "matrix.h"
#include "tmp.cpp"
#include <omp.h>
#include <time.h>
#include <chrono>
#include <iomanip> 
#include <thread>

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
const bool flag4 = 1; // parallel block multiplication2 (with trans. block 2nd matrix)				<- the fastest
const bool flag5 = 0; // parallel block multiplication3 (with subblock)
const bool flag6 = 0; // parallel block multiplication4 (with trans. block 2nd matrix && subblock)

const double EPS = 0.001;

int main() 
{																					   //???15000?									//???15000?
	int size = 8192;	//size = 8192 ~ 6900 ms (parallel block multiplication2) or ~ 15000 ms (parallel block multiplication3) or ~ 16000 ms (parallel block multiplication4)
						//mkl dgemm() ~ 3100 ms for same size

	matrix<double> A(size, size), B(size, size), C(size, size);

	srand(time(NULL));

	//-----------------------------------------------------------------------------------------------

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

	//-----------------------------------------------------------------------------------------------

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

	//-----------------------------------------------------------------------------------------------

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

	//-----------------------------------------------------------------------------------------------

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

	//-----------------------------------------------------------------------------------------------

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

	//-----------------------------------------------------------------------------------------------
	
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
}
