#include<iostream>
#include<cstdlib>
#include<cstdlib>
#include<cuda.h>
#include<highgui.h>
#include<cv.h>

#define Mask_size  3
#define TILE_SIZE  32
#define BLOCK_SIZE 32

__constant__ char M[Mask_size*Mask_size];
__constant__ char M1[Mask_size*Mask_size];
//#define clamp(x) (min(max((x), 0.0), 1.0))

using namespace std;
using namespace cv;



__device__ unsigned char clamp(int value)//__device__ because it's called by a kernel
{
    if(value < 0)
        value = 0;
    else
        if(value > 255)
            value = 255;
    return  value;
}


//====== KERNEL CONVOLUCION 2D CON MEMORIA GLOBAL ====

__global__ void convolution2DGlobalMemKernel(unsigned char *In,char *Mask, char *Mask1, unsigned char *Out,int Mask_Width,int Rowimg,int Colimg)
{

   unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
   unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;

   int Pvalue = 0;
   int PvalueY = 0;
   double suma = 0;

   int N_start_point_row = row - (Mask_Width/2);
   int N_start_point_col = col - (Mask_Width/2);

   for(int i = 0; i < Mask_Width; i++)
   {
       for(int j = 0; j < Mask_Width; j++ )
       {
        if((N_start_point_col + j >=0 && N_start_point_col + j < Rowimg)
        &&(N_start_point_row + i >=0 && N_start_point_row + i < Colimg))
        {
          Pvalue += In[(N_start_point_row + i)*Rowimg+(N_start_point_col + j)] * Mask[i*Mask_Width+j];
          PvalueY += In[(N_start_point_row + i)*Rowimg+(N_start_point_col + j)] * Mask1[i*Mask_Width+j];
        }
       }
   }

   suma = sqrt(pow((double)Pvalue,2) + pow((double)PvalueY,2));
   Out[row*Rowimg+col] = clamp(suma);


}

//==========KERNEL CONVOLUCION 2D CON MEMORIA CONSTANTE =========================

__global__ void convolution2DConstantMemKernel(unsigned char *In,unsigned char *Out,int Mask_Width,int Rowimg,int Colimg)
 {
   unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
   unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;

   int Pvalue = 0;
   int PvalueY = 0;
   double suma = 0;

   int N_start_point_row = row - (Mask_Width/2);
   int N_start_point_col = col - (Mask_Width/2);

   for(int i = 0; i < Mask_Width; i++)
   {
       for(int j = 0; j < Mask_Width; j++ )
       {
         if((N_start_point_col + j >=0 && N_start_point_col + j < Rowimg)
         &&(N_start_point_row + i >=0 && N_start_point_row + i < Colimg))
         {
           Pvalue += In[(N_start_point_row + i)*Rowimg+(N_start_point_col + j)] * M[i*Mask_Width+j];
           PvalueY += In[(N_start_point_row + i)*Rowimg+(N_start_point_col + j)] * M1[i*Mask_Width+j];
         }
       }
    }
    suma = sqrt(pow((double)Pvalue,2) + pow((double)PvalueY,2));
   Out[row*Rowimg+col] = clamp(suma);
}

//======== KERNEL CONVOLUCION 2D CON MEMORIA COMPARTIDA =========== 

__global__ void convolution2DSharedMemKernel(unsigned char *imageInput,unsigned char *imageOutput,
 int maskWidth, int width, int height)
{
    __shared__ float N_ds[TILE_SIZE + Mask_size - 1][TILE_SIZE + Mask_size - 1];
    int n = maskWidth/2;
    int dest = threadIdx.y*TILE_SIZE+threadIdx.x, destY = dest / (TILE_SIZE+Mask_size-1), destX = dest % (TILE_SIZE+Mask_size-1),
        srcY = blockIdx.y * TILE_SIZE + destY - n, srcX = blockIdx.x * TILE_SIZE + destX - n,
        src = (srcY * width + srcX);
    if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
        N_ds[destY][destX] = imageInput[src];
    else
        N_ds[destY][destX] = 0;

    // Second batch loading
    dest = threadIdx.y * TILE_SIZE + threadIdx.x + TILE_SIZE * TILE_SIZE;
    destY = dest /(TILE_SIZE + Mask_size - 1), destX = dest % (TILE_SIZE + Mask_size - 1);
    srcY = blockIdx.y * TILE_SIZE + destY - n;
    srcX = blockIdx.x * TILE_SIZE + destX - n;
    src = (srcY * width + srcX);
    if (destY < TILE_SIZE + Mask_size - 1)
    {
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
            N_ds[destY][destX] = imageInput[src];
        else
            N_ds[destY][destX] = 0;
    }
    __syncthreads();

    int Pvalue = 0;
    int y, x;
    for (y = 0; y < maskWidth; y++)
        for (x = 0; x < maskWidth; x++)
            Pvalue += N_ds[threadIdx.y + y][threadIdx.x + x] * M[y * maskWidth + x];
    y = blockIdx.y * TILE_SIZE + threadIdx.y;
    x = blockIdx.x * TILE_SIZE + threadIdx.x;
    if (y < height && x < width)
        imageOutput[(y * width + x)] = clamp(Pvalue);
    __syncthreads();
}

//============ LLAMADO AL KERNEL 2D =========================

void convolution2DKernelCall(Mat image,unsigned char *In,unsigned char *Out,char *h_Mask, char *h_Mask1,
  int Mask_Width,int Row,int Col, int op)
{
  // Variables
  int Size_of_bytes =  sizeof(unsigned char)*Row*Col*image.channels();
  int Mask_size_bytes =  sizeof(char)*(Mask_size*Mask_size);
  unsigned char *d_In, *d_Out;
  char *d_Mask, *d_Mask1;
  float Blocksize=BLOCK_SIZE;


  // Memory Allocation in device
  cudaMalloc((void**)&d_In,Size_of_bytes);
  cudaMalloc((void**)&d_Out,Size_of_bytes);
  cudaMalloc((void**)&d_Mask,Mask_size_bytes);
  cudaMalloc((void**)&d_Mask1,Mask_size_bytes);
  // Memcpy Host to device
  cudaMemcpy(d_In,In,Size_of_bytes,cudaMemcpyHostToDevice);

  cudaMemcpy(d_Mask,h_Mask,Mask_size_bytes,cudaMemcpyHostToDevice);
  cudaMemcpy(d_Mask1,h_Mask1,Mask_size_bytes,cudaMemcpyHostToDevice);

  // Memoria constante
  cudaMemcpyToSymbol(M,h_Mask,Mask_size_bytes);// Using constant mem
  cudaMemcpyToSymbol(M1,h_Mask1,Mask_size_bytes);

  dim3 dimGrid(ceil(Row/Blocksize),ceil(Col/Blocksize),1);
  dim3 dimBlock(Blocksize,Blocksize,1);
  switch(op)//Para seleccionar el kernel que necesitemos que se ejecute al mismo tiempo que la version secuencial.
  {
    case 1:
    cout<<"Convolucion 2D usando memoria GLOBAL"<<endl;
    convolution2DGlobalMemKernel<<<dimGrid,dimBlock>>>(d_In,d_Mask, d_Mask1, d_Out,Mask_Width,Row,Col);
    break;
    case 2:
    cout<<"Convolucion 2D usando memoria constante"<<endl;
    convolution2DConstantMemKernel<<<dimGrid,dimBlock>>>(d_In,d_Out,Mask_Width,Row,Col);
    break;
    case 3:
    cout<<"Convolucion 2D usando memoria compartida"<<endl;
    convolution2DSharedMemKernel<<<dimGrid,dimBlock>>>(d_In,d_Out,Mask_Width,Row,Col);
    break;
  }

  cudaDeviceSynchronize();
  // save output result.
  cudaMemcpy (Out,d_Out,Size_of_bytes,cudaMemcpyDeviceToHost);
  // Free device memory
  cudaFree(d_In);
  cudaFree(d_Out);
  cudaFree(d_Mask);
}



int main()
{

  clock_t start, finish; //Variables para el Clock
  double elapsedParallel;
  double elapsedSequential;
  int Mask_Width =  Mask_size;
  Mat image;
  image = imread("inputs/img6.jpg",0);   // Lectura del archivo, el 0 (cero) significa que ya cargada y lista la imagen en escala de grises.

  //int op = 1;
  Size s = image.size();
  int Row = s.width;
  int Col = s.height;
  char h_Mask[] = {-1,0,1,-2,0,2,-1,0,1}; // kernel para la deteccion de bordes.
  char h_Mask1[] = {-1,-2,-1,0,0,0,1,2,1}; //si se quiere usar este filtro en el eje y.

  //image.channels() no se requiere porque la imagen ya esta en escala de grises

  unsigned char *img = (unsigned char*)malloc(sizeof(unsigned char)*Row*Col*image.channels());
  unsigned char *imgOut = (unsigned char*)malloc(sizeof(unsigned char)*Row*Col*image.channels());

  if( !image.data )
  {//Prueba de carga de la imagen apropiadamente
    cout<<"Problems loading the image"<<endl;
    return -1;
  }

  img = image.data;

  cout<<"Resultado Serial"<<endl;
  Mat grad_x;
  start = clock();
  Sobel(image,grad_x,CV_8UC1,1,0,3,1,0,BORDER_DEFAULT);
  finish = clock();
  //imwrite("./outputs/18592193.png",grad_x);
  elapsedSequential = (((double) (finish - start)) / CLOCKS_PER_SEC );
  cout<< "Proceso secuencial: " << elapsedSequential << " Ejecucion en segundos "<< endl;

  start = clock();
  convolution2DKernelCall(image,img,imgOut,h_Mask,h_Mask1,Mask_Width,Row,Col,3);
  finish = clock();
  elapsedParallel = (((double) (finish - start)) / CLOCKS_PER_SEC );
  cout<< "Proceso paralelo: " << elapsedParallel << " Ejecucion en segundos "<< endl;

  Mat gray_image;
  gray_image.create(Col,Row,CV_8UC1);
  gray_image.data = imgOut;
  imwrite("./outputs/18592193.png",gray_image);
  
  return 0;
}