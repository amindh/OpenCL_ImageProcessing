
/*
// kernel for copying
kernel void copy_image(__global const unsigned char *imageInput,
		       __global       unsigned char *imageOutput,
		       int width, int height) 
{
  // Get the index of the current element to be processed
  int2 coord = (int2)(get_global_id(0), get_global_id(1));

  int rowOffset = coord.y * width * 4;
  int index = rowOffset + coord.x * 4;
  if (coord.x < width && coord.y < height)
    {
      imageOutput[index]     = imageInput[index];
      imageOutput[index + 1] = imageInput[index + 1];
      imageOutput[index + 2] = imageInput[index + 2];
      imageOutput[index + 3] = imageInput[index + 3];
     
    }
}
*/

// kernel for mean filter
kernel void mean_filter(__global const unsigned char *imageInput,
		       __global       unsigned char *imageOutput,
		       int width, int height, int filter_size) 
{
  // Get the index of the current element to be processed
  int2 coord = (int2)(get_global_id(0), get_global_id(1));
  
  int x ,y;
  int rowOffset = coord.y * width * 4;
  int index = rowOffset + coord.x * 4;
  float a0=0.0;
  float a1=0.0;	
  float a2=0.0;	
  float a3=0.0;	

  if (coord.x < width && coord.y < height)
    {  
	for (x=coord.x-filter_size; x<coord.x+1+filter_size; x++){
		for (y=coord.y-filter_size; y<coord.y+filter_size+1; y++){
			a0 += imageInput[y*width*4+ x*4  ];
			a1 += imageInput[y*width*4+ x*4+1];
     			a2 += imageInput[y*width*4+ x*4+2];
      			a3 += imageInput[y*width*4+ x*4+3];	

		}
	}
      imageOutput[index]     = a0/((2*filter_size*1)*(2*filter_size*1));
      imageOutput[index + 1] = a1/((2*filter_size*1)*(2*filter_size*1));
      imageOutput[index + 2] = a2/((2*filter_size*1)*(2*filter_size*1));
      imageOutput[index + 3] = a3/((2*filter_size*1)*(2*filter_size*1));
     
    }

}


//version de départ du filtre gaussien

// kernel for gauss filter
kernel void gauss_filter(__global const unsigned char *imageInput,
		       __global       unsigned char *imageOutput,
		       int width, int height, int filter_size, float sigma) 
{
  // Get the index of the current element to be processed
  int2 coord = (int2)(get_global_id(0), get_global_id(1));
  
  int x ,y;
  int rowOffset = coord.y * width * 4;
  int index = rowOffset + coord.x * 4;
  float a=0.0f;
  float e=0.0f;
  float norm=0.0f;
  a=1/sqrt(2.0f*3.14*sigma*sigma);
  float a0=0.0;
  float a1=0.0;	
  float a2=0.0;	
  float a3=0.0;	

  if (coord.x < width && coord.y < height)
    {  
        imageOutput[index  ]=0;
        imageOutput[index+1]=0;
        imageOutput[index+2]=0;
        imageOutput[index+3]=0;
	for (x=coord.x-filter_size; x<coord.x+1+filter_size; x++){
		for (y=coord.y-filter_size; y<coord.y+filter_size+1; y++){
			e=exp(-((coord.x-x)*(coord.x-x)+(coord.y-y)*(coord.y-y))/(2.0f*sigma*sigma));
			a0 += a*e*imageInput[y*width*4+ x*4  ];
			a1 += a*e*imageInput[y*width*4+ x*4+1];
     			a2 += a*e*imageInput[y*width*4+ x*4+2];
      			a3 += a*e*imageInput[y*width*4+ x*4+3];
			norm=norm+a*e;	

		}
	}
	imageOutput[index    ] = a0/norm;
	imageOutput[index + 1] = a1/norm;
	imageOutput[index + 2] = a2/norm;
	imageOutput[index + 3] = a3/norm;
    }

}


// Première optimisation : le calcul de a=1/sqrt(2.0f*3.14*sigma*sigma) est fait dans le programme principal
// kernel for gauss1 filter
kernel void gauss1_filter(__global const unsigned char *imageInput,
		       __global       unsigned char *imageOutput,
		       int width, int height, int filter_size, float sigma, float a) 
{
  // Get the index of the current element to be processed
  int2 coord = (int2)(get_global_id(0), get_global_id(1));
  
  int x ,y;
  int rowOffset = coord.y * width * 4;
  int index = rowOffset + coord.x * 4;
  float e=0.0f;
  float norm=0.0f;
  float a0=0.0;
  float a1=0.0;	
  float a2=0.0;	
  float a3=0.0;	

  if (coord.x < width && coord.y < height)
    {  
	for (x=coord.x-filter_size; x<coord.x+1+filter_size; x++){
		for (y=coord.y-filter_size; y<coord.y+filter_size+1; y++){
			e=exp(-((coord.x-x)*(coord.x-x)+(coord.y-y)*(coord.y-y))/(2.0f*sigma*sigma));
			a0  += a*e*imageInput[y*width*4+ x*4  ];
			a1  += a*e*imageInput[y*width*4+ x*4+1];
     			a2  += a*e*imageInput[y*width*4+ x*4+2];
      			a3  += a*e*imageInput[y*width*4+ x*4+3];
			norm+= a*e;	

		}
	}
	imageOutput[index    ] = a0/norm;
	imageOutput[index + 1] = a1/norm;
	imageOutput[index + 2] = a2/norm;
	imageOutput[index + 3] = a3/norm;
    }

}


//Deuxième optimisation : le calcule des exponentielle est fait dans le programme principal
// kernel for gauss2 filter
kernel void gauss2_filter(__global const unsigned char *imageInput,
		       __global       unsigned char *imageOutput,
		       int width, int height, int filter_size, float sigma, float a , __global float *e) 
{
  // Get the index of the current element to be processed
  int2 coord = (int2)(get_global_id(0), get_global_id(1));
  
  int x ,y;
  int rowOffset = coord.y * width * 4;
  int index = rowOffset + coord.x * 4;
  float norm=0.0f;
  float a0=0.0;
  float a1=0.0;	
  float a2=0.0;	
  float a3=0.0;	

  if (coord.x < width && coord.y < height)
    {  
       
	for (x=coord.x-filter_size; x<coord.x+1+filter_size; x++){
		for (y=coord.y-filter_size; y<coord.y+filter_size+1; y++){
			a0  += a*e[x-coord.x+filter_size]*e[y-coord.y+filter_size]*imageInput[y*width*4+ x*4  ];
			a1  += a*e[x-coord.x+filter_size]*e[y-coord.y+filter_size]*imageInput[y*width*4+ x*4+1];
     			a2  += a*e[x-coord.x+filter_size]*e[y-coord.y+filter_size]*imageInput[y*width*4+ x*4+2];
      			a3  += a*e[x-coord.x+filter_size]*e[y-coord.y+filter_size]*imageInput[y*width*4+ x*4+3];
			norm+= a*e[x-coord.x+filter_size]*e[y-coord.y+filter_size];	

		}
	}

	imageOutput[index    ] = a0/norm;
	imageOutput[index + 1] = a1/norm;
	imageOutput[index + 2] = a2/norm;
	imageOutput[index + 3] = a3/norm;	
    }

}


// troisième optimisation : les valeurs de e sont chargées dans la mémoire locale
// kernel for gauss3 filter
kernel void gauss3_filter(__global const unsigned char *imageInput,
		       __global       unsigned char *imageOutput,
		       int width, int height, int filter_size, float sigma, float a , __constant float *e) 
{
  // Get the index of the current element to be processed
  int2 coord = (int2)(get_global_id(0), get_global_id(1));
  
  int x ,y;
  int rowOffset = coord.y * width * 4;
  int index = rowOffset + coord.x * 4;
  float norm=0.0f;
  float a0=0.0;
  float a1=0.0;	
  float a2=0.0;	
  float a3=0.0;	

  if (coord.x < width && coord.y < height)
    {  
	for (x=coord.x-filter_size; x<coord.x+1+filter_size; x++){
		for (y=coord.y-filter_size; y<coord.y+filter_size+1; y++){
			a0  += a*e[x-coord.x+filter_size]*e[y-coord.y+filter_size]*imageInput[y*width*4+ x*4  ];
			a1  += a*e[x-coord.x+filter_size]*e[y-coord.y+filter_size]*imageInput[y*width*4+ x*4+1];
     			a2  += a*e[x-coord.x+filter_size]*e[y-coord.y+filter_size]*imageInput[y*width*4+ x*4+2];
      			a3  += a*e[x-coord.x+filter_size]*e[y-coord.y+filter_size]*imageInput[y*width*4+ x*4+3];
			norm+= a*e[x-coord.x+filter_size]*e[y-coord.y+filter_size];

		}
	}
	imageOutput[index    ] = a0/norm;
	imageOutput[index + 1] = a1/norm;
	imageOutput[index + 2] = a2/norm;
	imageOutput[index + 3] = a3/norm;
    }

}

/* Quatrième optimisation : vectorisation des calculs. !! cette partie du programme ne marche pas erreur : parameter may not be qualified with an address space

*/


/*
// kernel for gauss4 filter
kernel void gauss4_filter(__global const unsigned  char4* imageInput,
		       __global       unsigned char4 *imageOutput,
		       int width, int height, int filter_size, float sigma, float a , __constant float *e) 
{
  // Get the index of the current element to be processed
  int2 coord = (int2)(get_global_id(0), get_global_id(1));
  
  int x ,y;
  int rowOffset = coord.y * width ;
  int index = rowOffset + coord.x ;
  float norm=0.0f;
  float4 a0=(float4) 0.0;
  if (coord.x < width && coord.y < height)
    {  
       
	for (x=coord.x-filter_size; x<coord.x+1+filter_size; x++){
		for (y=coord.y-filter_size; y<coord.y+filter_size+1; y++){
			float4 a0  += a*e[x-coord.x+filter_size]*e[y-coord.y+filter_size]*convert_float4(imageInput[y*width+ x  ]);
			
			norm+= a*e[x-coord.x+filter_size]*e[y-coord.y+filter_size];	

		}
	}
	imageOutput[index    ] = convert_uchar4(a0/norm);
	
    }

}
*/

