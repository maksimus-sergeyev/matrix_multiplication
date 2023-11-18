#pragma once
#include "matrix.h"
#include <time.h>
#include <chrono>
#include <iomanip> 

int correctness(double EPS) // 0 = correctly
{
	srand(time(NULL));

		for (int row1 = 1; row1 < 50; row1++)
			for (int col1 = 1; col1 < 50; col1++)
				for (int col2 = 1; col2 < 50; col2++)
				{

					int row2 = col1;

					matrix<double> A(row1, col1), B(row2, col2), C(row1, col2), D(row1, col2), E(row1, col2), F(row1, col2), G(row1, col2);

					A.randomfill();
					B.randomfill();

					C = A * B;

					mult(A, B, D);

					parallel_mult(A, B, E);

					block_mult(A, B, F, 4, 8);

					parallel_block_mult(A, B, G, 4, 8);

					if ((C - D).norm() > EPS) return 1;

					if ((C - E).norm() > EPS) return 2;

					if ((C - F).norm() > EPS) return 3;

					if ((C - G).norm() > EPS) return 4;
				}

		return 0;
}
