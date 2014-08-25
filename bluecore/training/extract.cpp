#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
using namespace std;

#include "/usr/local/include/jpeglib.h"

typedef unsigned char BYTE;

bool sch[512][512];
int w, h;
BYTE* bin;

inline char nextChar (char ch) {
	if (ch == '9') return 'A';
	else if (ch == 'Z') return 'a';
	else return ch + 1;
}

inline bool in (int x, int y) {
	return x >= 0 && x < h && y >= 0 && y < w;
}

const int max_dist_x = 4;
const int max_dist_y = 2;
const int max_dist = 4;
int moves;
int dx[1024];
int dy[1024];

int x_min, x_max, y_min, y_max;
bool canvas[512][512];

void floodfill (int x, int y) {
	x_min = min(x_min, x), x_max = max(x_max, x);
	y_min = min(y_min, y), y_max = max(y_max, y);

	canvas[x][y] = true;
	for (int k = 0; k < moves; k++) {
		int tx = x + dx[k], ty = y + dy[k];
		if (in(tx, ty) && !sch[tx][ty] && bin[tx * w + ty]) {
			sch[tx][ty] = true;
			floodfill(tx, ty);
		}
	}
}

const int o_width = 24;
const int o_height = 24;

bool has_char (int y, int max_h, int &rx, int &ry) {
	for (int i = 0; i < max_h; i++) {
		int tx = i, ty = y;
		if (in(tx, ty) && !sch[tx][ty] && bin[tx * w + ty]) {
			rx = tx, ry = ty;
			return true;
		}
	}
	return false;
}

const int h_step = 25;

void extract (const char* filename) {
	jpeg_decompress_struct cinfo;
	jpeg_error_mgr jerr;

	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_decompress(&cinfo);

	FILE *fp = fopen(filename, "rb");

	jpeg_stdio_src(&cinfo, fp);

	jpeg_read_header(&cinfo, TRUE);

	BYTE* data = new BYTE[cinfo.image_width * cinfo.image_height * cinfo.num_components];
	w = cinfo.image_width, h = cinfo.image_height;
	int cl = cinfo.num_components;
	printf("%s: %d px * %d px * %d channel(s)\n", filename, w, h, cl);

	jpeg_start_decompress(&cinfo);

	JSAMPROW row;
	while (cinfo.output_scanline < cinfo.output_height) {
		row = &data[(cinfo.output_height - cinfo.output_scanline - 1) * cinfo.image_width * cinfo.num_components];
		jpeg_read_scanlines(&cinfo, &row, 1);
	}
	jpeg_finish_decompress(&cinfo);

	jpeg_destroy_decompress(&cinfo);
	fclose(fp);

	bin = new BYTE[w * h];

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			int pt = cl * ((h - 1 - i) * w + j);
			bin[i * w + j] = data[pt] > 128 ? 0 : 1;
			// printf("%c", bin[i * w + j] ? '#' : ' ');
		}
		// printf("\n");
	}

	moves = 0;
	for (int i = -max_dist; i <= max_dist_x; i++) {
		for (int j = -max_dist; j <= max_dist_y; j++) {
			if (abs(i) + abs(j) <= max_dist) {
				dx[moves] = i, dy[moves] = j, ++moves;
			}
		}
	}

	memset(sch, 0, sizeof sch);
	char ch = '0';
	int max_h = h_step;
	while (ch <= 'z') {
		for (int j = 0; j < w; j++) {
			int rx, ry;
			if (has_char(j, max_h, rx, ry)) {
				x_min = y_min = 10000, x_max = y_max = -10000;
				memset(canvas, 0, sizeof canvas);
				sch[rx][ry] = true;
				printf("FOUND %c AT (%d, %d)\n", ch, rx, ry);
				floodfill(rx, ry);

				char tfile[64];
				sprintf(tfile, "data/%s-%d", filename, ch);
				fp = fopen(tfile, "w");

				int x_off = (x_max - x_min) / 2 - o_height / 2 + x_min;
				int y_off = (y_max - y_min) / 2 - o_width / 2 + y_min;
				printf("(%d, %d) -> (%d, %d)\n", x_min, y_min, x_max, y_max);
				printf("OFFSET: +(%d, %d)\n", x_off, y_off);

				char output[o_height][o_width + 1];
				for (int x = 0; x < o_height; x++) {
					for (int y = 0; y < o_width; y++) {
						if (in(x + x_off, y + y_off) && canvas[x + x_off][y + y_off]) {
							output[x][y] = '#';
						} else {
							output[x][y] = '.';
						}
					}
					output[x][o_width] = 0;
					fprintf(fp, "%s\n", output[x]);
					printf("%s\n", output[x]);
				}
				getchar();

				fclose(fp);
				ch = nextChar(ch);
			}
		}
		max_h += h_step;
	}

	delete[] data;
	delete[] bin;
}

int main (int argc, char *argv[]) {
	extract(argv[1]);
	return 0;
}
