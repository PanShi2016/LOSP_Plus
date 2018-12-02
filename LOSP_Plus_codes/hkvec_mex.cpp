/**
 * @file hkvec_mex.cpp
 * Implement a seeded heat-kernel clustering scheme.
 *
 *  Call with debugflag = 1 to display parameter values
 * before/after each call to a major function
 *
 *
 * USAGE:
 * [bestset,cond,cut,vol,y,npushes] = hkvec_mex(A,set,t,eps,debugflag)
 *
 *
 * TO COMPILE:
 *
 * if ismac
 *      mex -O -largeArrayDims hkvec_mex.cpp
 * else
 * mex -O CXXFLAGS="\$CXXFLAGS -std=c++0x" -largeArrayDims hkvec_mex.cpp
 *
 *
 */

#include <vector>
#include <queue>
#include <utility> // for pair sorting
#include <assert.h>
#include <limits>
#include <algorithm>
#include <math.h>

#include <unordered_set>
#include <unordered_map>
#define tr1ns std

#ifndef __APPLE__
#define __STDC_UTF_16__ 1
#endif

#include <mex.h>

#define DEBUGPRINT(x) do { if (debugflag) { \
mexPrintf x; mexEvalString("drawnow"); } \
} while (0)

int debugflag = 0;

struct sparsevec {
    typedef tr1ns::unordered_map<mwIndex,double> map_type;
    map_type map;
    /** Get an element and provide a default value when it doesn't exist
     * This command does not insert the element into the vector
     */
    double get(mwIndex index, double default_value=0.0) {
        map_type::iterator it = map.find(index);
        if (it == map.end()) {
            return default_value;
        } else {
            return it->second;
        }
    }

    /** Compute the sum of all the elements
     * Implements compensated summation
     */
    double sum() {
        double s=0.;
        for (map_type::iterator it=map.begin(),itend=map.end();it!=itend;++it) {
            s += it->second;
        }
        return s;
    }

    /** Compute the max of the element values
     * This operation returns the first element if the vector is empty.
     */
    mwIndex max_index() {
        mwIndex index=0;
        double maxval=std::numeric_limits<double>::min();
        for (map_type::iterator it=map.begin(),itend=map.end();it!=itend;++it) {
            if (it->second>maxval) { maxval = it->second; index = it->first; }
        }
        return index;
    }
};

struct sparserow {
    mwSize n, m;
    mwIndex *ai;
    mwIndex *aj;
    double *a;
};


/**
 * Returns the degree of node u in sparse graph s
 */
mwIndex sr_degree(sparserow *s, mwIndex u) {
    return (s->ai[u+1] - s->ai[u]);
}


/**
 * Computes the degree N for the Taylor polynomial
 * of exp(tP) to have error less than eps*exp(t)
 *
 * ( so exp(-t(I-P)) has error less than eps )
 */
unsigned int taylordegree(const double t, const double eps) {
    double eps_exp_t = eps*exp(t);
    double error = exp(t)-1;
    double last = 1.;
    double k = 0.;
    while(error > eps_exp_t){
        k = k + 1.;
        last = (last*t)/k;
        error = error - last;
    }
    return std::max((int)k, (int)1);
}

/*****
 *
 *          above:  DATA STRUCTURES
 *
 *
 *
 *          below:  CLUSTERING FUNCTIONS
 *
 ****/

/**
 *
 *  gsqexpmseed inputs:
 *      G   -   adjacency matrix of an undirected graph
 *      set -   seed vector: the indices of a seed set of vertices
 *              around which cluster forms; normalized so
 *                  set[i] = 1/set.size(); )
 *  output:
 *      y = exp(tP) * set
 *              with infinity-norm accuracy of eps * e^t
 *              in the degree weighted norm
 *  parameters:
 *      t   - the value of t
 *      eps - the accuracy
 *      max_push_count - the total number of steps to run
 *      Q - the queue data structure
 */
template <class Queue>
int gsqexpmseed(sparserow * G, sparsevec& set, sparsevec& y,
                const double t, const double eps,
                const mwIndex max_push_count, Queue& Q)
{
    DEBUGPRINT(("gsqexpmseed interior: t=%f eps=%f \n", t, eps));
    mwIndex n = G->n;
    mwIndex N = (mwIndex)taylordegree(t, eps);
    DEBUGPRINT(("gsqexpmedseed: n=%i N=%i \n", n, N));

    // initialize the weights for the different residual partitions
    // r(i,j) > d(i)*exp(t)*eps/(N*psi_j(t))
    //  since each coefficient but d(i) stays the same,
    //  we combine all coefficients except d(i)
    //  into the vector "pushcoeff"
    std::vector<double> psivec(N+1,0.);
    psivec[N] = 1;
    for (int k = 1; k <= N ; k++){
        psivec[N-k] = psivec[N-k+1]*t/(double)(N-k+1) + 1;
    } // psivec[k] = psi_k(t)
    std::vector<double> pushcoeff(N+1,0.);
    pushcoeff[0] = ((exp(t)*eps)/(double)N)/psivec[0]; // This is the correct version
    for (int k = 1; k <= N ; k++){
        pushcoeff[k] = pushcoeff[k-1]*(psivec[k-1]/psivec[k]);
    } // pushcoeff[j] = exp(t)*eps/(N*psivec[j])

    mwIndex ri = 0;
    mwIndex npush = 0;
    double rij = 0;
    // allocate data
    sparsevec rvec;

    // i is the node index, j is the "step"
#define rentry(i,j) ((i)+(j)*n)

    // set the initial residual, add to the queue
    for (sparsevec::map_type::iterator it=set.map.begin(),itend=set.map.end();
         it!=itend;++it) {
        ri = it->first;
        rij = it->second;
        rvec.map[rentry(ri,0)]+=rij;
        Q.push(rentry(ri,0));
    }

    while (npush < max_push_count) {
        // STEP 1: pop top element off of heap
        ri = Q.front();
        Q.pop();
        // decode incides i,j
        mwIndex i = ri%n;
        mwIndex j = ri/n;

        double degofi = (double)sr_degree(G,i);
        rij = rvec.map[ri];
        //
        // update yi
        y.map[i] += rij;

        // update r, no need to update heap here
        rvec.map[ri] = 0;

        double rijs = t*rij/(double)(j+1);
        double ajv = 1./degofi;
        double update = rijs*ajv;

        if (j == N-1) {
            // this is the terminal case, and so we add the column of A
            // directly to the solution vector y
            for (mwIndex nzi=G->ai[i]; nzi < G->ai[i+1]; ++nzi) {
                mwIndex v = G->aj[nzi];
                y.map[v] += update;
            }
            npush += degofi;
        }
        else {
            // this is the interior case, and so we add the column of A
            // to the residual at the next time step.
            for (mwIndex nzi=G->ai[i]; nzi < G->ai[i+1]; ++nzi) {
                mwIndex v = G->aj[nzi];
                mwIndex re = rentry(v,j+1);
                double reold = rvec.get(re);
                double renew = reold + update;
                double dv = sr_degree(G,v);
                rvec.map[re] = renew;
                if (renew >= dv*pushcoeff[j+1] && reold < dv*pushcoeff[j+1]) {
                    Q.push(re);
                }
            }
            npush+=degofi;
        }
        // terminate when Q is empty, i.e. we've pushed all r(i,j) > eps*exp(t)*d(i)/(N*psi_j(t))
        if ( Q.size() == 0) { return npush; }
    }//end 'while'
    return (npush);
}


struct greater2nd {
    template <typename P> bool operator() (const P& p1, const P& p2) {
        return p1.second > p2.second;
    }
};


/** Cluster will contain a list of all the vertices in the cluster
 * @param set the set of starting vertices to use
 * @param t the value of t in the heatkernelPageRank computation
 * @param eps the solution tolerance eps
 * @param p the heatkernelpagerank vector
 * @param r the residual vector
 * @param a vector which supports .push_back to add vertices for the cluster
 * @param stats a structure for statistics of the computation
 */
template <class Queue>
int hypercluster_heatkernel_multiple(sparserow* G,
                                     const std::vector<mwIndex>& set, double t, double eps,
                                     sparsevec& p, sparsevec &r, Queue& q)
{
    // reset data
    p.map.clear();
    r.map.clear();
    q.empty();
    DEBUGPRINT(("beginning of hypercluster \n"));

    size_t maxdeg = 0;
    for (size_t i=0; i<set.size(); ++i) { //populate r with indices of "set"
        assert(set[i] >= 0); assert(set[i] < G->n); // assert that "set" contains indices i: 1<=i<=n
        size_t setideg = sr_degree(G,set[i]);
        r.map[set[i]] = 1./(double)(set.size()); // r is normalized to be stochastic
        //    DEBUGPRINT(("i = %i \t set[i] = %i \t setideg = %i \n", i, set[i], setideg));
        maxdeg = std::max(maxdeg, setideg);
    }

    DEBUGPRINT(("at last, gsqexpm: t=%f eps=%f \n", t, eps));

    int nsteps = gsqexpmseed(G, r, p, t, eps, ceil(pow(G->n,1.5)), q);
    /**
     *      **********
     *
     *        ***       GSQEXPMSEED       is called         ***
     *
     *      **********
     */

    if (nsteps == 0) {
        p = r; // just copy over the residual
    }

    // scale the probablities by their degree
/*    for (sparsevec::map_type::iterator it=p.map.begin(),itend=p.map.end();
         it!=itend;++it) {
        it->second *= (1.0/(double)std::max(sr_degree(G,it->first),(mwIndex)1));
    }
 */

    return (0);
}

/** Grow a set of seeds via the heat-kernel.
 *
 * @param G sparserow version of input matrix A
 * @param seeds a vector of input seeds seeds (index 0, N-1), and then
 *          updated to have the final solution nodes as well.
 * @param t the value of t in the heat-kernel
 * @param eps the solution tolerance epsilon
 * @param fcond the final conductance score of the set.
 * @param fcut the final cut score of the set
 * @param fvol the final volume score of the set
 */
void hkgrow(sparserow* G, std::vector<mwIndex>& seeds, double t,
            double eps, sparsevec& p )
{
    sparsevec r;
    std::queue<mwIndex> q;

    DEBUGPRINT(("hkvec_mex: call to hypercluster_heatkernel() start\n"));
    hypercluster_heatkernel_multiple(G, seeds, t, eps, p, r, q );
    DEBUGPRINT(("hkvec_mex: call to hypercluster_heatkernel() DONE\n"));
}

void copy_array_to_index_vector(const mxArray* v, std::vector<mwIndex>& vec)
{
    mxAssert(mxIsDouble(v), "array type is not double");
    size_t n = mxGetNumberOfElements(v);
    double *p = mxGetPr(v);

    vec.resize(n);

    for (size_t i=0; i<n; ++i) {
        double elem = p[i];
        mxAssert(elem >= 1, "Only positive integer elements allowed");
        vec[i] = (mwIndex)elem - 1;
    }
}


// USAGE
// [y] = hkvec_mex(A,set,t,eps,debugflag)
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs < 2 || nrhs > 5) {
        mexErrMsgIdAndTxt("hkvec_mex:wrongNumberArguments",
                          "hkvec_mex needs two to five arguments, not %i", nrhs);
    }
    if (nrhs == 5) {
        debugflag = (int)mxGetScalar(prhs[4]);
    }
    DEBUGPRINT(("hkvec_mex: preprocessing start: debugflag = %i\n", debugflag));

    const mxArray* mat = prhs[0];
    const mxArray* set = prhs[1];

    if ( mxIsSparse(mat) == false ){
        mexErrMsgIdAndTxt("hkvec_mex:wrongInputMatrix",
                          "hkvec_mex needs sparse input matrix");
    }
    if ( mxGetM(mat) != mxGetN(mat) ){
        mexErrMsgIdAndTxt("hkvec_mex:wrongInputMatrixDimensions",
                          "hkvec_mex needs square input matrix");
    }

    if ( nlhs > 1 || nlhs < 0 ){
        mexErrMsgIdAndTxt("hkvec_mex:wrongNumberOutputs",
                          "hkvec_mex needs 0 to 1 outputs, not %i", nlhs);
    }

    double eps = pow(10,-4);
    double t = 10.;

    if (nrhs >= 3) { t = mxGetScalar(prhs[2]); }
    if (nrhs >= 4) { eps = mxGetScalar(prhs[3]); }

    sparserow r;
    r.m = mxGetM(mat);
    r.n = mxGetN(mat);
    r.ai = mxGetJc(mat);
    r.aj = mxGetIr(mat);
    r.a = mxGetPr(mat);

    std::vector< mwIndex > seeds;
    copy_array_to_index_vector( set, seeds );
    sparsevec hkpr;

    DEBUGPRINT(("hkvec_mex: preprocessing end: \n"));

    hkgrow(&r, seeds, t, eps, hkpr );

    DEBUGPRINT(("hkvec_mex: call to hkgrow() done\n"));

    if (nlhs > 0) { // sets output "y" to the heat kernel vector computed
        mwIndex soln_size = hkpr.map.size();

        mxArray* passign = mxCreateDoubleMatrix(soln_size,2,mxREAL);
        plhs[0] = passign;

        double *pi = mxGetPr(passign);
        size_t i = 0;
        for (sparsevec::map_type::iterator it=hkpr.map.begin(),itend=hkpr.map.end();it!=itend;++it) {
            pi[i] = (double)(it->first)+1.0; // must shift 1 to convert from C indices to MATLAB indices
            pi[i + soln_size] = (double)(it->second);
            i++;
        }
    }
}
