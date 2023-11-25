#pragma once
#include "matrix.h"
#include <time.h>
#include <chrono>
#include <iomanip> 

const int sz = 10;

int correctness(double EPS) // 0 = correctly
{
	srand(time(NULL));

		for (int row1 = 1; row1 <= sz; row1++)
			for (int col1 = 1; col1 <= sz; col1++)
				for (int col2 = 1; col2 <= sz; col2++)
				{
					for (int i = 1; i <= 10; i++)
						for (int j = 1; j <= 10; j++)
					{
						
						int row2 = col1;

						matrix<double> A(row1, col1), B(row2, col2), C(row1, col2), D(row1, col2), E(row1, col2), F(row1, col2), G(row1, col2);

						A.randomfill();
						B.randomfill();

						C = A * B;

						D[0] = E[0] = F[0] = G[0] = 100;
						D[row1 * col2 - 1] = E[row1 * col2 - 1] = F[row1 * col2 - 1] = G[row1 * col2 - 1] = 100;

						mult(A, B, D);

						parallel_mult(A, B, E);

						block_mult(A, B, F, i, j);

						parallel_block_mult(A, B, G, i, j);

						if ((C - D).norm() > EPS) return 1;

						if ((C - E).norm() > EPS) return 2;

						if ((C - F).norm() > EPS) return 3;

						if ((C - G).norm() > EPS) return 4;
					}
				}

		return 0;
}
