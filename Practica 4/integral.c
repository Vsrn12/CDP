#include <stdio.h> 
#include <omp.h>

float f(float x) {
  float return_val; 
  return_val =  (x*x*x/3)+4*x; 
  return return_val; 
} 
 
int main (int argc, char *argv[]) 
{ 
  int nthreads, i, tid,chunk;
  float integral=0.0;
  float total =0.0;  		
  float part = 0.0;
  float a = atoi(argv[1]);
  float b = atoi(argv[2]);
  int n = atoi(argv[3]);  
  float h;  				
  float x = 0.0; 
  float local_a;         
  float local_b;         
  int local_n;           
  h = (b-a)/n;
  #pragma omp parallel
  {	   
	nthreads = omp_get_num_threads();
	tid = omp_get_thread_num();
	
	local_n = n/nthreads;
	local_a = a + tid * local_n * h; 
	local_b = local_a + local_n * h; 
	integral = (f(local_a) + f(local_b))/2.0;		  
	x = local_a;
		
	#pragma omp parallel for schedule (static) private(x,integral) ordered
		for (i = 1; i < local_n; i++) {
			#pragma omp ordered
			{   x += h;
			    integral += f(x); 
			}
		} 		
		#pragma omp critical
		{   total += integral;}
			
  }
  total *= h;
  if (total < 0) total *= -1;
  printf("Com n = %d trapezóides, a estimativa \n",  n); 
  printf("da  integral de %f até %f = %f \n",  a, b, total); 
  return 0;
}
