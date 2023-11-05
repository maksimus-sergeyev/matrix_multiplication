#include "square_matrix.h"
#include <time.h>
#include <chrono>

//L1 cache - 640 KB, L2 cache - 4 MB, L3 cache - 16 MB
//block_size <=  sqrt ( L2 / (3 * sizeof(T)) )
// T = double, block_size <= 418;

const bool flagC = 0; // checking for correctness
const bool flag0 = 0; // classic i, j, k multiplication
const bool flag1 = 0; // i, k, j multiplication
const bool flag2 = 1; // block multiplication
const bool flagBMP = 0; // block multiplication on pointers

const double EPS = 0.0001;

int main() 
{
	srand(time(NULL));

	int size = 4000;
	
	//------------------------------------mult0-----------------------------------------------------------
		
	if (flag0)
	{
		square_matrix<double> A(size), B(size), C(size);

		A.randomfill();
		B.randomfill();

		auto begin = std::chrono::steady_clock::now();

		C = A * B;

		auto end = std::chrono::steady_clock::now();

		auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

		std::cout << "The time of 'classic multiplication': " << elapsed_ms.count() << " ms\n";
	}

	//------------------------------------mult-----------------------------------------------------------

	if (flag1) 
	{
		square_matrix<double> A1(size), B1(size), C1(size);

		A1.randomfill();
		B1.randomfill();

		auto begin = std::chrono::steady_clock::now();

		mult(A1, B1, C1, size);

		auto end = std::chrono::steady_clock::now();

		auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

		std::cout << "The time of 'i, k, j multiplication': " << elapsed_ms.count() << " ms\n";

		if (flagC) 
		{
			square_matrix<double>D1(size);
			D1 = A1 * B1;
			if ((D1 - C1).abs() < EPS) std::cout << "'i, k, j multiplication' - correct" << std::endl;
		}
	}

	//------------------------------------mult2-----------------------------------------------------------

	if (flag2)
	{
		square_matrix<double> A2(size), B2(size), C2(size);

		A2.randomfill();
		B2.randomfill();

		auto begin = std::chrono::steady_clock::now();

		block_mult(A2, B2, C2, size);

		auto end = std::chrono::steady_clock::now();

		auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

		std::cout << "The time of 'block multiplication': " << elapsed_ms.count() << " ms\n";

		if (flagC)
		{
			square_matrix<double>D2(size);

			D2 = A2 * B2;

			if ((D2 - C2).abs() < EPS) std::cout << "'block multiplication' - correct" << std::endl;

		}
	}

	//------------------------------------PM-------------------------------------------------------------

	if (flagBMP)
	{
		square_matrix<double> A3(size), B3(size), C3(size);
		double* a3 = A3.getarr(), *b3 = B3.getarr(), *c3 = C3.getarr();

		A3.randomfill();
		B3.randomfill();

		auto begin = std::chrono::steady_clock::now();

		block_mult_pointers(a3, b3, c3, size);

		auto end = std::chrono::steady_clock::now();

		auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

		std::cout << "The time of 'block multiplication on pointers': " << elapsed_ms.count() << " ms\n";

		if (flagC)
		{
			square_matrix<double>D3(size);

			D3 = A3 * B3;

			if ((D3 - C3).abs() < EPS) std::cout << "'block multiplication on pointers' - correct" << std::endl;

		}
	}

	return 0;
}

/*
#include "square_matrix.h";
#include <time.h>
#include <chrono>

int main()
{
	int size = 4096;
	for (int i = 0; i < 10; i++)
	{
		square_matrix<double> A(size), B(size), C(size);

		A.randomfill();
		B.randomfill();

		auto begin = std::chrono::steady_clock::now();

		block_mult(A, B, C, size);

		auto end = std::chrono::steady_clock::now();

		auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

		std::cout << elapsed_ms.count() << "\n";
	}
}*/