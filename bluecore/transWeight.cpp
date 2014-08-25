#include <cstdio>
#include <cstdlib>
#include <cstring>
using namespace std;

const int N_INPUT = 12 * 12, N_HIDDEN = 64, N_OUTPUT = 62;
double w_ih[N_INPUT + 1][N_HIDDEN], w_ho[N_HIDDEN + 1][N_OUTPUT];

int main ()
{
	FILE *fsave = fopen("weights.log", "r");
	for (int i = 0; i < N_INPUT; i++) {
		for (int j = 0; j < N_HIDDEN; j++) {
			fscanf(fsave, "%lf", &w_ih[i][j]);
		}
	}
	for (int i = 0; i < N_HIDDEN; i++) {
		for (int j = 0; j < N_OUTPUT; j++) {
			fscanf(fsave, "%lf", &w_ho[i][j]);
		}
	}
	fclose(fsave);

	FILE *flog = fopen("weights_C.log", "w");
	fprintf(flog, "double w_ih[N_INPUT][N_HIDDEN] =\n{\n");
	for (int i = 0; i < N_INPUT; i++) {
		fprintf(flog, "\t{");
		for (int j = 0; j < N_HIDDEN; j++) {
			fprintf(flog, "%.8f%s", w_ih[i][j], j == N_HIDDEN - 1 ? "" : ", ");
		}
		fprintf(flog, "}%s\n", i == N_INPUT - 1 ? "" : ", ");
	}
	fprintf(flog, "};\n");
	fprintf(flog, "double w_ho[N_HIDDEN][N_OUTPUT] = \n{\n");
	for (int i = 0; i < N_HIDDEN; i++) {
		fprintf(flog, "\t{");
		for (int j = 0; j < N_OUTPUT; j++) {
			fprintf(flog, "%.8f%s", w_ho[i][j], j == N_OUTPUT - 1 ? "" : ", ");
		}
		fprintf(flog, "}%s\n", i == N_HIDDEN - 1 ? "" : ", ");
	}
	fprintf(flog, "};\n");
	fclose(flog);

	return 0;
}
