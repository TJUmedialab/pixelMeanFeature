#include <mex.h>
#include <iostream>
#include "k-means.h"
using namespace std;

//#define KMEANS

enum INPUTARRAY
{
	IMAGE, MASK, CENTERS, PATCHSIZE, MODELTYPE, NUMBEROFBINS, GRADIENTDIRECTION
};

template <typename T>
T *expandImage(T *image, int m, int n,int,  int patch_size);

void mexFunction ( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
	double *image_full = (double*)mxGetPr(prhs[IMAGE]);
	int M = mxGetDimensions(prhs[IMAGE])[0];
	int N = mxGetDimensions(prhs[IMAGE])[1];
	int channels=0 ;
	if ( N == (int)mxGetN(prhs[IMAGE])) {
		channels = 1;
	} else {
		channels = 3;
	}
	double *masks = (double *)mxGetPr(prhs[MASK]);
	double *centers = (double *)mxGetPr(prhs[CENTERS]);
	int regionNumbers = mxGetDimensions(prhs[CENTERS])[0];
	int patch_size = mxGetScalar(prhs[PATCHSIZE]);

	int model_name_length = mxGetN(prhs[MODELTYPE])*sizeof(mxChar) + 1;
	char *model_name = new char[model_name_length];
	mxGetString(prhs[MODELTYPE], model_name, model_name_length);
	int number_of_bins=0;
	double *direction=NULL;
	number_of_bins = mxGetScalar(prhs[NUMBEROFBINS]);
	printf("the number of bins is %d\n", number_of_bins);
	direction = mxGetPr(prhs[GRADIENTDIRECTION]);
	int pixel_label;

	int offset = patch_size/2;

	int max_label = 0, min_label = 100000;
	for ( int i = 0 ; i < M*N ; i ++) {
		max_label = (max_label < *(masks + i )) ? *(masks + i ) : max_label ;
		min_label = (min_label > *(masks + i )) ? *(masks + i ) : min_label;
	}

	if ( !strcmp(model_name, "gray") || !strcmp(model_name, "magnitude") ) {
		plhs[0] = mxCreateDoubleMatrix(patch_size*patch_size, max_label - min_label + 1 , mxREAL);
	}else if ( !strcmp(model_name, "append") ) {
		plhs[0] = mxCreateDoubleMatrix(patch_size*patch_size+number_of_bins, max_label - min_label + 1 , mxREAL);
	}else if ( !strcmp(model_name, "RGB") || !strcmp(model_name, "rgb") ) {
		plhs[0] = mxCreateDoubleMatrix(patch_size*patch_size*3, max_label - min_label + 1 , mxREAL);
	} else if (!strcmp(model_name, "direction")) {
		plhs[0] = mxCreateDoubleMatrix(number_of_bins, max_label - min_label + 1 , mxREAL);
	}
	
	plhs[1] = mxCreateDoubleMatrix(max_label - min_label + 1, 1, mxREAL);
	double *region_feature = (double *)mxGetPr(plhs[0]);
	double *region_pixels = (double *)mxGetPr(plhs[1]);

	// initial number of pixels of each region, as well as the region feature
	for ( int i = 0 ; i < (max_label - min_label + 1) ; i ++) {
		region_pixels[i] = 0;
	}
	
	double *new_image = expandImage(image_full, M, N, channels, patch_size);


#ifdef KMEANS

	int kmeans_size = 10;			//Number of samples
	int kmeans_dim = patch_size*patch_size;			//Dimension of feature
	int kmeans_cluster_num = 1;	//Cluster number
	double **pixels_feature ;

	// first we should get the number of pixels per region
	for ( int i = 0 ; i< M ; i ++) {
		for (int j = 0; j < N; j++) {
			pixel_label = *(masks + j * M + i)-min_label;
			region_pixels[pixel_label] ++;
		}
	}
	pixels_feature = new double *[max_label - min_label + 1];
	for ( int i = 0; i < max_label - min_label + 1 ; i++ ) {
		pixels_feature[i] = new double [patch_size*patch_size*region_pixels[i]];
	}
	int *region_pixels_count = new int[max_label - min_label + 1 ];

	// initial number of pixels of each region, as well as the region feature
	for ( int i = 0 ; i < (max_label - min_label + 1) ; i ++) {
		region_pixels_count[i] = 0;
	}
	for ( int i = 0 ; i< M ; i +=patch_size) {
		for (int j = 0; j < N; j+=patch_size) {
			pixel_label = *(masks + j * M + i)-min_label;
			int k = 0 ;
			for (int k1 = 0; k1 < patch_size; k1++) {
				for ( int k2 = 0 ; k2 < patch_size ; k2 ++) {
					pixels_feature[pixel_label][region_pixels_count[pixel_label] * patch_size*patch_size + k] = 
						new_image[(i+k2)*(N+patch_size-1)+j+k1];
					//region_feature[pixel_label*patch_size*patch_size+k] += 
					//	new_image[(i+k2)*(M+patch_size-1)+j+k1];
					k ++;
				}
			}
			region_pixels_count[pixel_label]++;
		}
	}
	for ( int i = 0 ; i<(max_label - min_label + 1); i ++ ) {
		if ( region_pixels_count [i] == 0 ) {
			for ( int j = 0 ; j < M ; j ++) {
				for ( int k = 0 ;  k < N ; k ++) {
					pixel_label = *(masks + j * M + i)-min_label ;
					if ( pixel_label == i) {
						int k = 0 ;
						for (int k1 = 0; k1 < patch_size; k1++) {
							for ( int k2 = 0 ; k2 < patch_size ; k2 ++) {
								pixels_feature[pixel_label][region_pixels_count[pixel_label] * patch_size*patch_size + k] = 
									new_image[(i+k2)*(N+patch_size-1)+j+k1];
								k ++;
							}
						}
						region_pixels_count[i]++;
						break;
					}
				}
				if ( region_pixels_count[i] == 1) {
					break;
				}
			}
		}
	}
	KMeans* kmeans = new KMeans(kmeans_dim,kmeans_cluster_num);
	kmeans->SetInitMode(KMeans::InitUniform);
//#pragma omp parallel for
	for ( int i = 0  ; i < (max_label - min_label + 1) ; i ++) {
		kmeans_size = region_pixels_count[i];
		kmeans->Cluster(pixels_feature[i],kmeans_size,region_feature+i*patch_size*patch_size);
	}
	for ( int i = 0; i < max_label - min_label + 1 ; i++ ) {
		delete [] pixels_feature[i] ;
	}
	delete[] pixels_feature;
	delete[] region_pixels_count;
#else
	
	for ( int i = 0 ; i < (max_label - min_label + 1) ; i ++) {
		region_pixels[i] = 0;
	}

	//
	if (!strcmp(model_name, "gray") || !strcmp(model_name, "magnitude")) {
		for ( int i = 0 ; i < (patch_size*patch_size)*(max_label - min_label + 1) ; i ++) {
			region_feature[i] = 0;
		}
		for ( int i = 0 ; i< M ; i +=patch_size) {
			for (int j = 0; j < N; j+=patch_size) {
				pixel_label = *(masks + j * M + i)-min_label ;
				int k = 0 ;
				for (int k1 = 0; k1 < patch_size; k1++) {
					for ( int k2 = 0 ; k2 < patch_size ; k2 ++) {
						region_feature[pixel_label*patch_size*patch_size+k] += 
							new_image[(i+k2)*(N+patch_size-1)+j+k1];
						k ++;
					}
				}
				region_pixels[pixel_label]++;
			}
		}

		int noMeanCount = 0;
		int kk, centerx, centery;
		for ( int j = 0 ; j < M ; j ++) {
			for ( int k = 0 ;  k < N ; k ++) {
				pixel_label = *(masks + k * M + j)-min_label ;
				if ( region_pixels [pixel_label] == 0) {
					region_pixels[pixel_label]++;
					noMeanCount++;
					kk = 0 ;
					centerx = centers[pixel_label];
					centery = centers[regionNumbers+pixel_label];
					//printf("centers: %d, %d\n", centerx, centery);
					for (int k1 = 0; k1 < patch_size; k1++) {
						for ( int k2 = 0 ; k2 < patch_size ; k2 ++) {
							region_feature[pixel_label*patch_size*patch_size+kk] += 
								new_image[(centerx+k2)*(N+patch_size-1)+centery+k1];
							kk ++;
						}
					}
				}
			}
		}
		printf("number of regions without centre: %d\n", noMeanCount);
		for ( int i = 0 ; i<(max_label - min_label + 1); i ++ ) {
			for ( int j = 0 ; j < patch_size*patch_size ; j ++) {
				//if (region_pixels[i]==0) {
				//	printf("%d ", i);
				//}
				region_feature[i*patch_size*patch_size+j] /= region_pixels[i];
			}
		}
	} else if ( !strcmp(model_name, "RGB") || !strcmp(model_name, "rgb") ) {
		for ( int i = 0 ; i < (patch_size*patch_size*3)*(max_label - min_label + 1) ; i ++) {
			region_feature[i] = 0;
		}
		int region_height = patch_size * patch_size * 3;
		int image_step = (M+patch_size-1)*(N+patch_size-1);
		for ( int i = 0 ; i< M ; i +=patch_size) {
			for (int j = 0; j < N; j+=patch_size) {
				pixel_label = *(masks + j * M + i)-min_label ;
				int k = 0 ;
				for (int k1 = 0; k1 < patch_size; k1++) {
					for ( int k2 = 0 ; k2 < patch_size ; k2 ++) {
						region_feature[pixel_label*region_height+k] += 
							new_image[(i+k2)*(N+patch_size-1)+j+k1];
						region_feature[pixel_label*region_height+patch_size*patch_size+k] += 
							new_image[image_step+(i+k2)*(N+patch_size-1)+j+k1];
						region_feature[pixel_label*region_height+patch_size*patch_size*2+k] += 
							new_image[image_step*2+(i+k2)*(N+patch_size-1)+j+k1];
						k ++;
					}
				}
				region_pixels[pixel_label]++;
			}
		}
		for ( int j = 0 ; j < M ; j ++) {
			for ( int k = 0 ;  k < N ; k ++) {
				pixel_label = *(masks + k * M + j)-min_label ;
				if ( region_pixels [pixel_label] == 0) {
					region_pixels[pixel_label]++;
					int kk = 0 ;
					int centerx = centers[(max_label - min_label + 1)*pixel_label],
						centery = centers[(max_label - min_label + 1)*(pixel_label+1)];
					for (int k1 = 0; k1 < patch_size; k1++) {
						for ( int k2 = 0 ; k2 < patch_size ; k2 ++) {
							region_feature[pixel_label*region_height+patch_size*patch_size+kk] += 
								new_image[(centerx+k2)*(N+patch_size-1)+centery+k1];
							region_feature[pixel_label*region_height+patch_size*patch_size+kk] += 
								new_image[image_step+(centerx+k2)*(N+patch_size-1)+centery+k1];
							region_feature[pixel_label*region_height+patch_size*patch_size+kk] += 
								new_image[image_step*2+(centerx+k2)*(N+patch_size-1)+centery+k1];
							kk ++;
						}
					}
				}
			}
		}
		
		for ( int i = 0 ; i<(max_label - min_label + 1); i ++ ) {
			for ( int j = 0 ; j < region_height ; j ++) {
				//if (region_pixels[i]==0) {
				//	printf("%d ", i);
				//}
				region_feature[i*patch_size*patch_size+j] /= region_pixels[i];
			}
		}
	} else if (!strcmp(model_name, "direction")) {
		double angle_width = 360.0/number_of_bins;
		double dirc;
		int belong_bin;
		int region_height = number_of_bins;
		for ( int i = 0 ; i < number_of_bins*(max_label - min_label + 1) ; i ++) {
			region_feature[i] = 0;
		}
		for ( int i = 0 ; i < M ; i ++) {
			for ( int j = 0 ; j < N ; j ++ ) {
				pixel_label = *(masks + j * M + i)-min_label ;
				dirc = direction[j*M+i];
				belong_bin = dirc / angle_width;
				region_feature[pixel_label*region_height+belong_bin] ++;
				region_pixels[pixel_label]++;
			}
		}
		for ( int i = 0 ; i<(max_label - min_label + 1); i ++ ) {
			for ( int j = 0 ; j < region_height; j ++) {
				//if (region_pixels[i]==0) {
				//	printf("%d ", i);
				//}
				region_feature[i*region_height+j] /= region_pixels[i];
			}
		}
	} else if ( !strcmp(model_name, "append") ){
		for ( int i = 0 ; i < (patch_size*patch_size+number_of_bins)*(max_label - min_label + 1) ; i ++) {
			region_feature[i] = 0;
		}
		for ( int i = 0 ; i< M ; i +=patch_size) {
			for (int j = 0; j < N; j+=patch_size) {
				pixel_label = *(masks + j * M + i)-min_label ;
				int k = 0 ;
				for (int k1 = 0; k1 < patch_size; k1++) {
					for ( int k2 = 0 ; k2 < patch_size ; k2 ++) {
						region_feature[pixel_label*(patch_size*patch_size+number_of_bins)+k] += 
							new_image[(i+k2)*(N+patch_size-1)+j+k1];
						k ++;
					}
				}
				region_pixels[pixel_label]++;
			}
		}
		for ( int j = 0 ; j < M ; j ++) {
			for ( int k = 0 ;  k < N ; k ++) {
				pixel_label = *(masks + k * M + j)-min_label ;
				if ( region_pixels [pixel_label] == 0) {
					region_pixels[pixel_label]++;
					int kk = 0 ;
					for (int k1 = 0; k1 < patch_size; k1++) {
						for ( int k2 = 0 ; k2 < patch_size ; k2 ++) {
							region_feature[pixel_label*(patch_size*patch_size+number_of_bins)+kk] += 
								new_image[(j+k2)*(N+patch_size-1)+k+k1];
							kk ++;
						}
					}
				}
			}
		}
		for ( int i = 0 ; i<(max_label - min_label + 1); i ++ ) {
			for ( int j = 0 ; j < patch_size*patch_size ; j ++) {
				//if (region_pixels[i]==0) {
				//	printf("%d ", i);
				//}
				region_feature[i*(patch_size*patch_size+number_of_bins)+j] /= region_pixels[i];
			}
		}
		for ( int i = 0 ; i < (max_label - min_label + 1) ; i ++) {
			region_pixels[i] = 0;
		}
		double angle_width = 360.0/number_of_bins;
		double dirc;
		int belong_bin;

		for ( int i = 0 ; i < M ; i ++) {
			for ( int j = 0 ; j < N ; j ++ ) {
				pixel_label = *(masks + j * M + i)-min_label ;
				dirc = direction[j*M+i];
				belong_bin = dirc / angle_width;
				region_feature[pixel_label*(patch_size*patch_size+number_of_bins)+patch_size*patch_size+belong_bin] ++;
				region_pixels[pixel_label]++;
			}
		}
		for ( int i = 0 ; i<(max_label - min_label + 1); i ++ ) {
			for ( int j = patch_size*patch_size ; j < patch_size*patch_size +number_of_bins; j ++) {
				//if (region_pixels[i]==0) {
				//	printf("%d ", i);
				//}
				region_feature[i*(patch_size*patch_size+number_of_bins)+j] /= region_pixels[i];
			}
		}
	}
	
#endif // KMEANS
	
	delete[] new_image;
}


template <typename T>
T *expandImage(T *image, int m, int n,int channels, int patch_size) {
	//row first
	int step = patch_size / 2; 
	T *newImage = new T[(m+patch_size-1)*(n+patch_size-1)*channels] ;
	int i, j;

	for ( int channeli = 0 ; channeli < channels ; channeli++) {
		int channel_step = channeli*m*n;
		int new_channel_step = channeli*(m+patch_size-1)*(n+patch_size-1);
		
		//copy the middle part to new image
		for ( i = 0 ; i < m ; i ++) {
			for ( j = 0 ; j < n; j ++) {
				newImage[(i+step)*(n+patch_size-1) + (j+step) + new_channel_step] = image[j*m + i + channel_step];
			}
		}

		//build top side using mirror map
		for ( i = 0 ; i < step ; i ++ ) {
			for ( j = step ; j < n + step ; j ++) {
				newImage[ i*(n+patch_size-1) + j + new_channel_step] = newImage[ (patch_size-i-1)*(n+patch_size-1) + j  + new_channel_step];
			}
		}

		//build bottom side using mirror map
		for ( i = m + step ; i < m + patch_size - 1 ; i ++ ) {
			for ( j = step ; j < n + step ; j ++) {
				newImage[ i*(n+patch_size-1) + j + new_channel_step ] = newImage[ (2*(m+step-1)-i)*(n+patch_size-1) + j + new_channel_step ];
			}
		}

		//build left side 
		for ( i = 0 ; i < m + patch_size - 1 ; i ++ ) {
			for ( j = 0 ; j < step ; j ++) {
				newImage[ i*(n+patch_size-1) + j + new_channel_step ] = newImage[ i*(n+patch_size-1) + (patch_size - 1 - j) + new_channel_step ];
			}
		}

		//build right side 
		for ( i = 0 ; i < m + patch_size - 1 ; i ++ ) {
			for ( j = n+step ; j < n+patch_size-1 ; j ++) {
				newImage[ i*(n+patch_size-1) + j + new_channel_step ] = newImage[ i*(n+patch_size-1) + 2*(n+step-1)-j + new_channel_step ];
			}
		}
	}
	
	return newImage ;
}