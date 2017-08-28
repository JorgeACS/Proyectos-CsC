#include <stdio.h>
#include <accelmath.h> // Note: when compiling in linux, also type -lm for the math.h to work
#include <openacc.h>
#include "dislin.h"
//#include <math.h>

void main(){
    int i,j,ntime,nsteps,nx=256,ny=256;

    float u[nx+1][ny+1],ut[nx][ny];
    float gama,PI;
    float lap,h,h2,dt;
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

        for( i = 1; i < nx; i++ ){
          for( j = 1; j < ny; j++ ){
          //  if( ((i-nx/2)*(i-nx/2) + (j-ny/2)*(j-ny/2)) < 400 )
           // if( i > nx/2-20 && i < nx/2+20)
            if ( (i)*(i)+(j)*(j) > 1500 && (i)*(i)+(j)*(j) < 2500 )
              u[i][j] = 3.0f;
          //    u[i][j]=6*sin(PI*i/256);
            else
              u[i][j]=0.0f;
	  }
	}	

#pragma acc data copyin(u,ut)
{

/* Main Iteration loop */

   for( ntime = 0; ntime < 100000; ntime++){

/* Boundary Conditions: No-flux*/
#pragma acc kernels
{
   for( i = 1; i < nx; i++ ){
     u[i][1]=u[i][3];
     u[i][ny-1]=u[i][ny-2];
   }
}
#pragma acc kernels
{
   for( j = 1; j < ny; j++ ){
      u[1][j]=u[3][j];
      u[nx-1][j]=u[nx-2][j];
   }
}

/* Euler Scheme */

#pragma acc kernels
{
   for( i = 1; i < nx; i++ ){
     for( j = 1; j < ny; j++ ){                 
       lap=(u[i+1][j]+u[i-1][j]+u[i][j-1]+u[i][j+1]-4.*u[i][j]);
       ut[i][j]=u[i][j]+lap*dt*gama/h2;
     
     } 
   }
}
/* Update of the mesh */     
#pragma acc kernels
{
   for( i = 1; i < nx; i++ ){
     for( j = 1; j < ny; j++ ){
       u[i][j]=ut[i][j];
     }
   }
}

   if(ntime%500 == 0){
     printf("%d \n", ntime);

#pragma acc update host(u)
// Dislin Plotting 

        metafl ("GL");
        disini ();
        pagera ();

        titlin("Ecuación del calor en 2-D",2);

        axspos (450, 1800);
        axslen (2200, 1200);

        name   ("Eje X", "x");
        name   ("Eje Y", "y");

	intax();
	autres(nx,ny);
	axspos(600,1850);
	ax3len(1500,1500,1500);
	
        graf3(0.0,nx,0.0,100.0,1.0,ny,0.0,100.0,
                 0.0,1.5,0.0,1.1);

	crvmat((float *) u,nx+1,ny+1,1,1);
	
	height(50);
	title();
	endgrf();
	erase();

   }

   }
}
         
        printf("Ya terminamos!!!!!! \n");
        return;
}
