#include <iostream>
#include <cstdlib>
#include <cstdlib>
#include <cuda.h>
#include <highgui.h>
#include <cv.h>

#define DIM 40
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

void kernel(unsigned char *ptr)
{
	for (int i = 0; i < DIM; ++i)
	{
		for (int j = 0; j < DIM; ++j)
		{
			int offset = j + i*DIM;
			int juliaValue = julia (j, i);
			cout <<offset << ": " <<juliaValue <<endl;
			ptr[offset] = 255 * juliaValue;
			/*
			ptr[offset*4 + 0] = 255 * juliaValue;
			ptr[offset*4 + 1] = 0;
			ptr[offset*4 + 2] = 0;
			ptr[offset*4 + 3] = 255;*/
		}
	}
}

int main()
{
	unsigned char *imageInput;

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

	cout << "El alto y ancho de la imagen tiene respectivamente " << width << " pixels por " << height << " pixels\n";
	
	kernel(imageInput);
	
	Mat imageFractal;
	imageFractal.create(DIM,DIM,CV_8UC1);
  	imageFractal.data = imageInput;
  	imwrite("./outputs/1088273734.png", imageFractal);
}