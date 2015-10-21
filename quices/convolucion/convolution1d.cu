#include<cstdlib>
#include<time.h>
#include<cuda.h>
#include<iostream>
#include<math.h> //Included just to use the Power function

#define BLOCK_SIZE 32
#define TILE_SIZE 32
#define MAX_MASK_WIDTH 10
__constant__ float M[MAX_MASK_WIDTH];

using namespace std;

//====== Function made to print vector =========================================
void printVector(float *A, int length)
{
  for (int i=0; i<length; i++)
  {
    cout<<A[i]<<" | ";
  }
  cout<<endl;
}

//====== Function made to fill the vector with some given value ================
void fillVector(float *A, float value, int length)
{
  for (int i=0; i<length; i++)
  {
    A[i] = value;
  }
}

//====== Compare results =======================================================
void compareVector (float *A, float *B,int n)
{
  for (int i=0; i<n; i++ )
  {
    if (A[i]!=B[i])
    {
      cout<<"## Secuential and Parallel results are NOT equal ##"<<endl;
    }
  }
  cout<<"== Secuential and Parallel results are equal =="<<endl;
}

//====== Serial Convolution ====================================================
void serialConvolution(float *input, float *output, float *mask, int mask_length, int length)
{
  int start = 0;
  float temp = 0.0;
  for (int i = 0; i < length; i++)
  {
    for (int j = 0; j < mask_length; j++)
    {
      start = i - (mask_length / 2);
      if (start + j >= 0 && start + j < length)
        temp += input[start + j] * mask[j];
    }
    output[i] = temp;
    temp = 0.0;
  }
}


//====== Basic convolution kernel ==============================================
__global__ void convolutionBasicKernel(float *N, float *M, float *P,
 int Mask_Width, int Width)
 {
   int i = blockIdx.x*blockDim.x + threadIdx.x;
   float Pvalue = 0;
   int N_start_point = i - (Mask_Width/2);
   for (int j = 0; j < Mask_Width; j++)
   {
     if (N_start_point + j >= 0 && N_start_point + j < Width)
     {
       Pvalue += N[N_start_point + j]*M[j];
     }
   }
   P[i] = Pvalue;
}

//====== Convolution kernel using constant memory and caching ==================
__global__ void convolutionKernelConstant(float *N, float *P, int Mask_Width,
 int Width)
 {
   int i = blockIdx.x*blockDim.x + threadIdx.x;
   float Pvalue = 0;
   int N_start_point = i - (Mask_Width/2);
   for (int j = 0; j < Mask_Width; j++)
   {
     if (N_start_point + j >= 0 && N_start_point + j < Width)
     {
       Pvalue += N[N_start_point + j]*M[j];
     }
   }
   P[i] = Pvalue;
}

//===== Tiled Convolution kernel using shared memory ===========================
__global__ void convolutionKernelShared(float *N, float *P, int Mask_Width,
 int Width)
 {
   int i = blockIdx.x*blockDim.x + threadIdx.x;
   __shared__ float N_ds[TILE_SIZE + MAX_MASK_WIDTH - 1];
   int n = Mask_Width/2;
   int halo_index_left = (blockIdx.x - 1)*blockDim.x + threadIdx.x;
   if (threadIdx.x >= blockDim.x - n)
   {
     N_ds[threadIdx.x - (blockDim.x - n)] =
     (halo_index_left < 0) ? 0 : N[halo_index_left];
   }
   N_ds[n + threadIdx.x] = N[blockIdx.x*blockDim.x + threadIdx.x];
   int halo_index_right = (blockIdx.x + 1)*blockDim.x + threadIdx.x;
   if (threadIdx.x < n)
   {
     N_ds[n + blockDim.x + threadIdx.x] =
     (halo_index_right >= Width) ? 0 : N[halo_index_right];
   }
   __syncthreads();
   float Pvalue = 0;
   for(int j = 0; j < Mask_Width; j++)
   {
     Pvalue += N_ds[threadIdx.x + j]*M[j];
   }
   P[i] = Pvalue;
}

//====== A simplier tiled convolution kernel using shared memory and general cahching
__global__ void convolutionKernelSharedSimplier(float *N, float *P, int Mask_Width,
 int Width)
{
 int i = blockIdx.x*blockDim.x + threadIdx.x;
 __shared__ float N_ds[TILE_SIZE];
 N_ds[threadIdx.x] = N[i];
 __syncthreads();
 int This_tile_start_point = blockIdx.x * blockDim.x;
 int Next_tile_start_point = (blockIdx.x + 1) * blockDim.x;
 int N_start_point = i - (Mask_Width/2);
 float Pvalue = 0;
 for (int j = 0; j < Mask_Width; j ++)
 {
   int N_index = N_start_point + j;
   if (N_index >= 0 && N_index < Width)
   {
     if ((N_index >= This_tile_start_point)
     && (N_index < Next_tile_start_point))
     {
       Pvalue += N_ds[threadIdx.x+j-(Mask_Width/2)]*M[j];
     } else
     {
       Pvalue += N[N_index] * M[j];
     }
   }
 }
 P[i] = Pvalue;
}


//===== Convolution kernel call ================================================
void convolutionCall (float *input, float *output, float *mask, int mask_length, int length)
{
  float *d_input;
  float *d_mask;
  float *d_output;
  float block_size = BLOCK_SIZE;//The compiler doesn't let me cast the variable

  cudaMalloc(&d_input, length * sizeof(float));
  cudaMalloc(&d_mask, mask_length * sizeof(float));
  cudaMalloc(&d_output, length * sizeof(float));

  cudaMemcpy (d_input, input, length * sizeof (float), cudaMemcpyHostToDevice);
  cudaMemcpy (d_mask, mask, mask_length * sizeof (float), cudaMemcpyHostToDevice);

  dim3 dimGrid (ceil (length / block_size), 1, 1);
  dim3 dimBlock (block_size, 1, 1);

  convolutionBasicKernel<<<dimGrid, dimBlock>>> (d_input, d_mask, d_output, mask_length, length);
  cudaDeviceSynchronize();

  cudaMemcpy (output, d_output, length * sizeof (float), cudaMemcpyDeviceToHost);
  cudaFree (d_input);
  cudaFree (d_mask);
  cudaFree (d_output);
}

//==============================================================================
void convolutionCallConstant (float *input, float *output, float *mask, int mask_length, int length)
{
  float *d_input;
  float *d_output;
  float block_size = BLOCK_SIZE;//The compiler doesn't let me cast the variable

  cudaMalloc(&d_input, length * sizeof(float));
  cudaMalloc(&d_output, length * sizeof(float));

  cudaMemcpy (d_input, input, length * sizeof (float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol (M, mask, mask_length * sizeof (float));

  dim3 dimGrid (ceil (length / block_size), 1, 1);
  dim3 dimBlock (block_size, 1, 1);

  convolutionKernelConstant<<<dimGrid, dimBlock>>> (d_input,d_output, mask_length, length);
  cudaDeviceSynchronize();

  cudaMemcpy (output, d_output, length * sizeof (float), cudaMemcpyDeviceToHost);
  cudaFree (d_input);
  cudaFree (d_output);
}

//==============================================================================
void convolutionCallWithTilesComplex (float *input, float *output, float *mask, int mask_length, int length)
{
  float *d_input;
  float *d_output;
  float block_size = BLOCK_SIZE;//The compiler doesn't let me cast the variable

  cudaMalloc(&d_input, length * sizeof(float));
  cudaMalloc(&d_output, length * sizeof(float));

  cudaMemcpy (d_input, input, length * sizeof (float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol (M, mask, mask_length * sizeof (float));

  dim3 dimGrid (ceil (length / block_size), 1, 1);
  dim3 dimBlock (block_size, 1, 1);

  convolutionKernelShared<<<dimGrid, dimBlock>>> (d_input,d_output, mask_length, length);
  cudaDeviceSynchronize();

  cudaMemcpy (output, d_output, length * sizeof (float), cudaMemcpyDeviceToHost);
  cudaFree (d_input);
  cudaFree (d_output);
}

//====== Convolution kernel call tiled version (the simplified one) ============
void convolutionCallWithTiles (float *input, float *output, float *mask, int mask_length, int length)
{
  float *d_input;
  float *d_output;
  float block_size = BLOCK_SIZE;//The compiler doesn't let me cast the variable

  cudaMalloc(&d_input, length * sizeof(float));
  cudaMalloc(&d_output, length * sizeof(float));

  cudaMemcpy (d_input, input, length * sizeof (float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol (M, mask, mask_length * sizeof (float));

  dim3 dimGrid (ceil (length / block_size), 1, 1);
  dim3 dimBlock (block_size, 1, 1);

  convolutionKernelSharedSimplier<<<dimGrid, dimBlock>>> (d_input,d_output, mask_length, length);
  cudaDeviceSynchronize();

  cudaMemcpy (output, d_output, length * sizeof (float), cudaMemcpyDeviceToHost);
  cudaFree (d_input);
  cudaFree (d_output);
}

//================= MAIN =======================================================
int main ()
{
  for(int i=5; i<=20;i++)//to execute the program many times just to get all the test values
  {
    cout<<"=> EXECUTION #"<<i-5<<endl;
    unsigned int length = pow(2,i);
    int mask_length = 5;
    int op = 1; //To select which parallel version we want to execute
    //1 Paralelo Basico - 2 Paralelo con memoria constante
    //3 paralelo con memoria compartida y tiling 
    clock_t start, finish; //Clock variables
    double elapsedSecuential, elapsedParallel, elapsedParallelConstant,
    elapsedParallelSharedComplex,elapsedParallelSharedTiles, optimization;

    float *A = (float *) malloc(length * sizeof(float));
    float *mask = (float *) malloc(mask_length * sizeof(float));
    float *Cserial = (float *) malloc(length * sizeof(float));
    float *Cparallel = (float *) malloc(length * sizeof(float));
    float *CparallelWithTiles = (float *) malloc(length * sizeof(float));
    float *CparallelConstant = (float *) malloc (length * sizeof(float));
    float *CparallelWithTilesComplex = (float *) malloc(length * sizeof(float));

    fillVector(A,1.0,length);
    fillVector(mask,2.0,mask_length);
    fillVector(Cserial,0.0,length);
    fillVector(Cparallel,0.0,length);
    fillVector(CparallelWithTiles,0.0,length);
    fillVector(CparallelConstant,0.0,length);
    fillVector(CparallelWithTilesComplex,0.0,length);

    //============================================================================
    cout<<"Serial result"<<endl;
    start = clock();
    serialConvolution(A,Cserial,mask,mask_length,length);
    finish = clock();
    elapsedSecuential = (((double) (finish - start)) / CLOCKS_PER_SEC );
    cout<< "The Secuential process took: " << elapsedSecuential << " seconds to execute "<< endl;
    //printVector(Cserial,length);
    cout<<endl;

    //============================================================================
    switch (op)
    {
      case 1:
              cout<<"==============================================================="<<endl;
              cout<<"Paralelo basico"<<endl;
              start = clock();
              convolutionCall(A,Cparallel,mask,mask_length,length);
              finish = clock();
              elapsedParallel = (((double) (finish - start)) / CLOCKS_PER_SEC );
              cout<< "The parallel process took: " << elapsedParallel << " seconds to execute "<< endl;
              optimization = elapsedSecuential/elapsedParallel;
              cout<< "The acceleration we've got: " << optimization <<endl;
              //printVector(Cparallel,length);
              compareVector(Cserial,Cparallel,length);
              cout<<endl;
              break;

      case 2:
              cout<<"==============================================================="<<endl;
              cout<<"Paralelo con memoria constante"<<endl;
              start = clock();
              convolutionCallConstant(A,CparallelConstant,mask,mask_length,length);
              finish = clock();
              elapsedParallelConstant = (((double) (finish - start)) / CLOCKS_PER_SEC );
              cout<< "The parallel process took: " << elapsedParallelConstant << " seconds to execute "<< endl;
              optimization = elapsedSecuential/elapsedParallelConstant;
              cout<< "The acceleration we've got: " << optimization <<endl;
              //printVector(CparallelConstant,length);
              compareVector(Cserial,CparallelConstant,length);
              cout<<endl;
              break;

      case 3:
              cout<<"==============================================================="<<endl;
              cout<<"Paralelo con memoria compartida y Tiling"<<endl;
              start = clock();
              convolutionCallWithTilesComplex(A,CparallelWithTilesComplex,mask,mask_length,length);
              finish = clock();
              elapsedParallelSharedComplex = (((double) (finish - start)) / CLOCKS_PER_SEC );
              cout<< "The parallel process took: " << elapsedParallelSharedComplex << " seconds to execute "<< endl;
              optimization = elapsedSecuential/elapsedParallelSharedComplex;
              cout<< "The acceleration we've got: " << optimization <<endl;
              //printVector(CparallelWithTilesComplex,length);
              compareVector(Cserial,CparallelWithTilesComplex,length);
              cout<<endl;
              break;
      case 4:
              cout<<"==============================================================="<<endl;
              cout<<"Parallel with shared memory result simplified"<<endl;
              start = clock();
              convolutionCallWithTiles(A,CparallelWithTiles,mask,mask_length,length);
              finish = clock();
              elapsedParallelSharedTiles = (((double) (finish - start)) / CLOCKS_PER_SEC );
              cout<< "The parallel process took: " << elapsedParallelSharedTiles << " seconds to execute "<< endl;
              optimization = elapsedSecuential/elapsedParallelSharedTiles;
              cout<< "The acceleration we've got: " << optimization <<endl;
              //printVector(CparallelWithTiles,length);
              compareVector(Cserial,CparallelWithTiles,length);
              cout<<endl;
              break;
    }


    free(A);
    free(mask);
    free(Cserial);
    free(Cparallel);
    free(CparallelWithTiles);
    free(CparallelConstant);
    free(CparallelWithTilesComplex);
  }
}

 