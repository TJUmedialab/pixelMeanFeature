#ifndef _EMD_H
#define _EMD_H
/*
    emd.h

    Last update: 3/24/98

    An implementation of the Earth Movers Distance.
    Based of the solution for the Transportation problem as described in
    "Introduction to Mathematical Programming" by F. S. Hillier and 
    G. J. Lieberman, McGraw-Hill, 1990.

    Copyright (C) 1998 Yossi Rubner
    Computer Science Department, Stanford University
    E-Mail: rubner@cs.stanford.edu   URL: http://vision.stanford.edu/~rubner
*/


/* DEFINITIONS */
#define MAX_SIG_SIZE   100
#define MAX_ITERATIONS 500
#define INFINITY       1e20
#define EPSILON        1e-6

/*****************************************************************************/
/* feature_t SHOULD BE MODIFIED BY THE USER TO REFLECT THE FEATURE TYPE      */
typedef int feature_t;
/*****************************************************************************/


typedef struct
{
  int n;                /* Number of features in the signature */
  feature_t *Features;  /* Pointer to the features vector */
  double *Weights;       /* Pointer to the weights of the features */
} signature_t;


typedef struct
{
  int from;             /* Feature number in signature 1 */
  int to;               /* Feature number in signature 2 */
  float amount;         /* Amount of flow from "from" to "to" */
} flow_t;

#define DEBUG_LEVEL 0
/*
 DEBUG_LEVEL:
   0 = NO MESSAGES
   1 = PRINT THE NUMBER OF ITERATIONS AND THE FINAL RESULT
   2 = PRINT THE RESULT AFTER EVERY ITERATION
   3 = PRINT ALSO THE FLOW AFTER EVERY ITERATION
   4 = PRINT A LOT OF INFORMATION (PROBABLY USEFUL ONLY FOR THE AUTHOR)
*/


#define MAX_SIG_SIZE1  (MAX_SIG_SIZE+1)  /* FOR THE POSIBLE DUMMY FEATURE */

/* NEW TYPES DEFINITION */

/* node1_t IS USED FOR SINGLE-LINKED LISTS */
typedef struct node1_t {
  int i;
  double val;
  struct node1_t *Next;
} node1_t;

/* node1_t IS USED FOR DOUBLE-LINKED LISTS */
typedef struct node2_t {
  int i, j;
  double val;
  struct node2_t *NextC;               /* NEXT COLUMN */
  struct node2_t *NextR;               /* NEXT ROW */
} node2_t;

class EMD
{
public:
	EMD(signature_t *Signature1, signature_t *Signature2,
		float (*Dist)(feature_t *, feature_t *),
		flow_t *Flow, int *FlowSize);
	~EMD();
	float emd();
private:
	signature_t *Signature1;
	signature_t *Signature2;
	flow_t *Flow;
	int *FlowSize;
	/* GLOBAL VARIABLE DECLARATION */
	int _n1, _n2;                          /* SIGNATURES SIZES */
	float **_C;/* THE COST MATRIX */
	node2_t *_X;            /* THE BASIC VARIABLES VECTOR */
	/* VARIABLES TO HANDLE _X EFFICIENTLY */
	node2_t *_EndX, *_EnterX;
	char **_IsX;
	node2_t **_RowsX, **_ColsX;
	double _maxW;
	float _maxC;

	int itr;
	double totalCost;
	float w;
	node2_t *XP;
	flow_t *FlowP;
	node1_t *U, *V;

	/* DECLARATION OF FUNCTIONS */
	float init(signature_t *Signature1, signature_t *Signature2,
		float (*Dist)(feature_t *, feature_t *));
	void findBasicVariables();
	int isOptimal();
	int findLoop(node2_t **Loop);
	void newSol();
	void russel(double *S, double *D);
	void addBasicVariable(int minI, int minJ, double *S, double *D, 
		node1_t *PrevUMinI, node1_t *PrevVMinJ,
		node1_t *UHead);
#if DEBUG_LEVEL > 0
	void printSolution();
#endif
};



float emd(signature_t *Signature1, signature_t *Signature2,
		  float (*func)(feature_t *, feature_t *),
		  flow_t *Flow, int *FlowSize);

#endif
