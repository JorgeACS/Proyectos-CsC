#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <accelmath.h> // Note: when compiling in linux, also type -lm for the math.h to work
#include <openacc.h>
#include "dislin.h"
//#include <math.h>

void main(){
    int i,j,k,ntime,nsteps,nx=256,ny=256,nz=256;

    float u[nx+1][ny+1][nz+1],ut[nx][ny][nz];
    float gama,PI;
    float lap,h,h2,dt;

    FILE *fl;
/*
    double u[nx+1][ny+1],v[nx+1][ny+1],ut[nx][ny];
    double E_h,E_n,E_ast,tau_E,tau_n,Re,M,gama;
    double p,D_n,h_E,R_n,dv,du,xlap,h,h2,dt,tanu;
*/

/* Parameters to be used in the model */
    PI=3.1416f;
    gama=0.001f;

/* Numerical constants for the euler method */

    h=(float)(1.0f/nx); //0.025;
    h2=h*h;
    dt=0.001;
    nsteps=3000000;

// Setting the device 
  acc_set_device_num(0,acc_device_nvidia); 

// Condición inicial
      for( k = 1; k < nz; k++){
        for( i = 1; i < nx; i++ ){
          for( j = 1; j < ny; j++ ){
          //  if( ((i-nx/2)*(i-nx/2) + (j-ny/2)*(j-ny/2)) < 400 )
           // if( i > nx/2-20 && i < nx/2+20)
            if ( (i)*(i)+(j)*(j) > 1500 && (i)*(i)+(j)*(j) < 2500 )
              u[i][j][k] = 3.0f;
          //    u[i][j]=6*sin(PI*i/256);
            else
              u[i][j][k]=0.0f;
	  }
	}	
      }

#pragma acc data copyin(u,ut)
{

/* Main Iteration loop */

   for( ntime = 0; ntime < 5000; ntime++){

/* Boundary Conditions: No-flux*/
#pragma acc kernels
{
  for(k = 1; k < nz; k++){
   for( i = 1; i < nx; i++ ){
     u[i][1][k]=u[i][3][k];
     u[i][ny-1][k]=u[i][ny-2][k];
   }
  }
}
#pragma acc kernels
{
  for (k=1; k < nz; k++){
   for( j = 1; j < ny; j++ ){
      u[1][j][k]=u[3][j][k];
      u[nx-1][j][k]=u[nx-2][j][k];
   }
  }
}

/* Euler Scheme */

#pragma acc kernels
{
  for( k = 1; k < nz; k++) {
   for( i = 1; i < nx; i++ ){
      for( j = 1; j < ny; j++ ){                 
       lap=(u[i+1][j][k]+u[i-1][j][k]+
            u[i][j-1][k]+u[i][j+1][k]+
            u[i][j][k+1]+u[i][j][k-1]-6.*u[i][j][k]);
       
        ut[i][j][k]=u[i][j][k]+lap*dt*gama/h2;
      } 
    }
  }
}
/* Update of the mesh */     
#pragma acc kernels
{
  for ( k = 1; k < nz; k++){
   for( i = 1; i < nx; i++ ){
     for( j = 1; j < ny; j++ ){
       u[i][j][k]=ut[i][j][k];
     }
   }
  }
}

   if(ntime%250 == 0){
     printf("%d iteraciones \n", ntime);
     char cad1[20];

     sprintf(cad1,"calor_3d_%i.csv",ntime);

     //strcat(cad1,cad2);
     puts(cad1);

#pragma acc update host(u)
       
    fl=fopen(cad1,"w");

    for ( k = 1; k < nz; k++){
      for( i = 1; i < nx; i++ ){
        for( j = 1; j < ny; j++ ){
          if (u[i][j][k] > 0.1f){
            fprintf(fl, "%d , %d , %d , %f \n", i,j,k,u[i][j][k]);
           }
        }
      }
    }

   }   //este es del if
}

}
        close(fl); 
        printf("Ya terminamos!!!!!! \n");
        return;
}
