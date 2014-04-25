#include <mex.h>
#include <iostream>
#include "k-means.h"
using namespace std;

//#define KMEANS

enum INPUTARRAY
{
	IMAGE, MASK, PATCHSIZE
};

template <typename T>
T *expandImage(T *image, int m, int n, int patch_size);

void mexFunction ( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
	double *image_full = (double*)mxGetPr(prhs[IMAGE]);
	int M = mxGetM(prhs[IMAGE]);
	int N = mxGetN(prhs[IMAGE]);
	double *masks = (double *)mxGetPr(prhs[MASK]);
	int patch_size = mxGetScalar(prhs[PATCHSIZE]);
	int pixel_label;

	int offset = patch_size/2;

	int max_label = 0, min_label = 100000;
	for ( int i = 0 ; i < M*N ; i ++) {
		max_label = (max_label < *(masks + i )) ? *(masks + i ) : max_label ;
		min_label = (min_label > *(masks + i )) ? *(masks + i ) : min_label;
	}
	plhs[0] = mxCreateDoubleMatrix(patch_size*patch_size, max_label - min_label + 1 , mxREAL);
	plhs[1] = mxCreateDoubleMatrix(max_label - min_label + 1, 1, mxREAL);
	double *region_feature = (double *)mxGetPr(plhs[0]);
	double *region_pixels = (double *)mxGetPr(plhs[1]);

	// initial number of pixels of each region, as well as the region feature
	for ( int i = 0 ; i < (max_label - min_label + 1) ; i ++) {
		region_pixels[i] = 0;
	}
	for ( int i = 0 ; i < (patch_size*patch_size)*(max_label - min_label + 1) ; i ++) {
		region_feature[i] = 0;
	}

	double *new_image = expandImage(image_full, M, N, patch_size);

	int kmeans_size = 10;			//Number of samples
	int kmeans_dim = patch_size*patch_size;			//Dimension of feature
	int kmeans_cluster_num = 1;	//Cluster number
	double **pixels_feature ;

#ifdef KMEANS


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

	for ( int i = 0 ; i<(max_label - min_label + 1); i ++ ) {
		if ( region_pixels [i] == 0 ) {
			//printf("%d ", i);
			for ( int j = 0 ; j < M ; j ++) {
				for ( int k = 0 ;  k < N ; k ++) {
					pixel_label = *(masks + k * M + j)-min_label ;
					if ( pixel_label == i) {
						region_pixels[i]++;
						int kk = 0 ;
						for (int k1 = 0; k1 < patch_size; k1++) {
							for ( int k2 = 0 ; k2 < patch_size ; k2 ++) {
								region_feature[pixel_label*patch_size*patch_size+kk] += 
									new_image[(j+k2)*(N+patch_size-1)+k+k1];
								kk ++;
							}
						}
						break;
					}
				}
				if ( region_pixels[i] == 1) {
					break;
				}
			}
		}
	}
	for ( int i = 0 ; i<(max_label - min_label + 1); i ++ ) {
		for ( int j = 0 ; j < patch_size*patch_size ; j ++) {
			//if (region_pixels[i]==0) {
			//	printf("%d ", i);
			//}
			region_feature[i*patch_size*patch_size+j] /= region_pixels[i];
		}
	}
#endif // KMEANS
	
	delete[] new_image;
}


template <typename T>
T *expandImage(T *image, int m, int n, int patch_size) {
	//row first
	int step = patch_size / 2; 
	T *newImage = new T[(m+patch_size-1)*(n+patch_size-1)] ;
	int i, j;

	//copy the middle part to new image
	for ( i = 0 ; i < m ; i ++) {
		for ( j = 0 ; j < n; j ++) {
			newImage[(i+step)*(n+patch_size-1) + (j+step)] = image[j*m + i];
		}
	}

	//build top side using mirror map
	for ( i = 0 ; i < step ; i ++ ) {
		for ( j = step ; j < n + step ; j ++) {
			newImage[ i*(n+patch_size-1) + j ] = newImage[ (patch_size-i-1)*(n+patch_size-1) + j ];
		}
	}

	//build bottom side using mirror map
	for ( i = m + step ; i < m + patch_size - 1 ; i ++ ) {
		for ( j = step ; j < n + step ; j ++) {
			newImage[ i*(n+patch_size-1) + j ] = newImage[ (2*(m+step-1)-i)*(n+patch_size-1) + j ];
		}
	}

	//build left side 
	for ( i = 0 ; i < m + patch_size - 1 ; i ++ ) {
		for ( j = 0 ; j < step ; j ++) {
			newImage[ i*(n+patch_size-1) + j ] = newImage[ i*(n+patch_size-1) + (patch_size - 1 - j) ];
		}
	}

	//build right side 
	for ( i = 0 ; i < m + patch_size - 1 ; i ++ ) {
		for ( j = n+step ; j < n+patch_size-1 ; j ++) {
			newImage[ i*(n+patch_size-1) + j ] = newImage[ i*(n+patch_size-1) + 2*(n+step-1)-j ];
		}
	}
	return newImage ;
}