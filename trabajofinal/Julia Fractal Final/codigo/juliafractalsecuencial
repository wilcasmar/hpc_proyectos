#include <iostream>
#include <cstdlib>
#include <cstdlib>
#include <cuda.h>
#include <highgui.h>
#include <cv.h>

#define DIM 200
using namespace std;
using namespace cv;

struct cuComplex 
{
	float r;
	float i;
	cuComplex( float a, float b ) : r(a), i(b) 
	{

	}

	float magnitude2( void ) 
	{ 
		return r * r + i * i; 
	}

	cuComplex operator*(const cuComplex& a) 
	{
		return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
	}
	cuComplex operator+(const cuComplex& a) 
	{
		return cuComplex(r+a.r, i+a.i);
	}
};


int julia(int x, int y)
{
	const float scale = 1.5;
	float jx = scale * (float)(DIM/2 - x)/(DIM/2);
	float jy = scale * (float)(DIM/2 - y)/(DIM/2);
	
	cuComplex c(-0.8, 0.156);
	cuComplex a(jx, jy);
	
	for (int i = 0; i < 200; i++) 
	{
		a = a * a + c;
		if (a.magnitude2() > 1000)
		return 0;
	}
	return 1;
}

void juliaCPU(unsigned char *ptr)
{
	for (int i = 0; i < DIM; ++i)
	{
		for (int j = 0; j < DIM; ++j)
		{
			int offset = j + i*DIM;
			int juliaValue = julia (j, i);
			//cout <<offset << ": " <<juliaValue <<endl;
			ptr[offset] = 255 * juliaValue;
		}
	}
}

int main()
{
	// Variables
	unsigned char *imageInput;
	clock_t inicio, final;
	double tiempo;
	double cont = 0, promedio = 0;

	Mat image (DIM, DIM, CV_8UC1, Scalar(255));

	if(!image.data)
	{
		printf("!!No se pudo cargar la Imagen!! \n");
		return -1;
	}

	Size s = image.size();
	int width = s.width;
	int height = s.height;
	int size = sizeof(unsigned char) * width * height * image.channels();

	imageInput = (unsigned char*)malloc(size);

	imageInput = image.data;

	for (int i = 0; i < 20; ++i)
	{
		inicio = clock();
		juliaCPU(imageInput);
		final = clock();	
		tiempo = (((double) (final - inicio)) / CLOCKS_PER_SEC );
		cont = cont + tiempo;
	}
	promedio = cont / 20;
	cout <<"El tiempo secuencial promedio para una dimension de " << DIM << "x" <<DIM <<" es de: " <<promedio <<" segundos.\n";
	Mat imageFractal;
	imageFractal.create(DIM,DIM,CV_8UC1);
  	imageFractal.data = imageInput;
  	imwrite("./outputs/1088273734.png", imageFractal);
}