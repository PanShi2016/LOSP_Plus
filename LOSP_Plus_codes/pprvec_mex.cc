/**
 * @file pprvec_mex.cc
 * Implement a fast, local PPR push scheme.
 * Simply computes the vector efficiently, not a cluster.
 */


#include <vector>
#include <queue>
#include <utility> // for pair sorting
#include <assert.h>
#include <limits>
#include <algorithm>


#include <unordered_set>
#include <unordered_map>
#define tr1ns std
#ifndef __APPLE__
#define __STDC_UTF_16__ 1
#endif

#include <mex.h>

/** A replacement for std::queue<int> using a circular buffer array */
class array_queue {
    public:
    size_t max_size;
    std::vector<int> array;
    size_t head, tail;
    size_t cursize;
    array_queue(size_t _max_size)
    : max_size(_max_size), array(_max_size), head(0), tail(0), cursize(0)
    {}

    void empty() {
        head = 0;
        tail = 0;
        cursize = 0;
    }

    size_t size() {
        return cursize;
    }

    void push(int i) {
        assert(size() < max_size);
        array[tail] = i;
        tail ++;
        if (tail == max_size) {
            tail = 0;
        }
        cursize ++;
    }

    int front() {
        assert(size() > 0);
        return array[head];
    }

    void pop() {
        assert(size() > 0);
        head ++;
        if (head == max_size) {
            head = 0;
        }
        cursize --;
    }
};

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

mwIndex sr_degree(sparserow *s, mwIndex u) {
    return (s->ai[u+1] - s->ai[u]);
}

template <class Queue>
int compute_local_pagerank(sparserow *s, sparsevec& r, sparsevec& p,
    double alpha, double epsilon, int max_push_count, Queue& q)
{
  for (sparsevec::map_type::iterator it=r.map.begin(),itend=r.map.end();
        it!=itend;++it){
    if (it->second > epsilon*sr_degree(s,it->first)) {
      q.push(it->first);
    }
  }

  int push_count = 0;
  while (q.size()>0 && push_count < max_push_count) {
    push_count += 1;
    mwIndex u = q.front();
    q.pop();
    mwIndex du = sr_degree(s, u);
    double moving_probability = r.map[u] - 0.5*epsilon*(double)du;
    r.map[u] = 0.5*epsilon*(double)du;
    p.map[u] += (1.-alpha)*moving_probability;

    double neighbor_update = alpha*moving_probability/(double)du;

    for (mwIndex nzi=s->ai[u]; nzi<s->ai[u+1]; nzi++) {
      mwIndex x = s->aj[nzi];
      mwIndex dx = sr_degree(s, x);
      double rxold = r.get(x);
      double rxnew = rxold + neighbor_update;
      r.map[x] = rxnew;
      if (rxnew > epsilon*dx && rxold <= epsilon*dx) {
        q.push(x);
      }
    }
  }

  return (push_count);
}



struct greater2nd {
  template <typename P> bool operator() (const P& p1, const P& p2) {
    return p1.second > p2.second;
  }
};



/** Cluster will contain a list of all the vertices in the cluster
 * @param set the set of starting vertices to use
 * @param alpha the value of alpha in the PageRank computation
 * @param target_vol the approximate number of edges in the cluster
 * @param p the pagerank vector
 * @param r the residual vector
 * @param a vector which supports .push_back to add vertices for the cluster
 * @param stats a structure for statistics of the computation
 */

template <class Queue>
int hypercluster_pagerank_multiple(sparserow* G,
    const std::vector<mwIndex>& set, double alpha, double pr_eps,
    sparsevec& p, sparsevec &r, Queue& q )
{
  // reset data
  p.map.clear();
  r.map.clear();
  q.empty();

  assert(pr_eps > 0);
  assert(alpha < 1.0); assert(alpha > 0.0);

  //r.map[start] = 1.0;
  size_t maxdeg = 0;
  for (size_t i=0; i<set.size(); ++i) {
    assert(set[i] >= 0); assert(set[i] < G->n);
    r.map[set[i]] = 1./(double)(set.size());
    //r.map[set[i]] = 1.;
    maxdeg = std::max(maxdeg, sr_degree(G,set[i]));
  }
  //double pr_eps = 1.0/std::max((double)sr_degree(G,start)*(double)target_vol, 100.0);
  //double pr_eps = std::min(1.0/std::max(10.*target_vol, 100.0),
    //1./(double)(set.size()*maxdeg + 1));
  // double pr_eps = 1.0/std::max(10.0*target_vol, 100.0);

  //printf("find_cluster: target_vol=%7lli alpha=%5.3ld pr_eps=%ld\n", target_vol, alpha, pr_eps);

  // calculate an integer number of maxsteps
  double maxsteps = 1./(pr_eps*(1.-alpha));
  maxsteps = std::min(maxsteps, 0.5*(double)std::numeric_limits<int>::max());

  int nsteps = compute_local_pagerank(G, r, p, alpha, pr_eps, (int)maxsteps, q);
  if (nsteps == 0) {
    p = r; // just copy over the residual
  }

  //mexPrintf("setsize=%zu, nsteps=%i, support=%i\n", set.size(), nsteps, support);

  return (0);
}


void pprgrow(sparserow* G, std::vector<mwIndex>& set, double alpha,
    double epsil, sparsevec& p)
{
    sparsevec r;
    std::queue<mwIndex> q;
    hypercluster_pagerank_multiple(G, set, alpha, epsil, p, r, q );
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
// [prvec] = pprgrow_mex(A,set,targetvol,alpha)
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    mxAssert(nrhs > 2 && nrhs < 6, "2-5 inputs required.");

    const mxArray* mat = prhs[0];
    const mxArray* set = prhs[1];

    mxAssert(mxIsSparse(mat), "Input matrix is not sparse");
    mxAssert(mxGetM(mat) == mxGetN(mat), "Input matrix not square");


    mxAssert(nlhs <= 1, "Too many output arguments");

    double alpha = 0.99;
    if (nrhs >= 4) { alpha = mxGetScalar(prhs[3]); }
    mxAssert(alpha >= 0. && alpha < 1, "alpha must be 0 <= alpha < 1");

    double epsil = pow(10,-4);
    if (nrhs >= 3) { epsil = mxGetScalar(prhs[2]); }

    sparserow r;
    r.m = mxGetM(mat);
    r.n = mxGetN(mat);
    r.ai = mxGetJc(mat);
    r.aj = mxGetIr(mat);
    r.a = mxGetPr(mat);

    std::vector< mwIndex > cluster;
    copy_array_to_index_vector( set, cluster );

    sparsevec p;

    pprgrow(&r, cluster, alpha, epsil, p);

    if (nlhs > 0) { // output the pagerank vector itself
        // p contains the pagerank vector info
        mwIndex soln_size = p.map.size();

        mxArray* passign = mxCreateDoubleMatrix(soln_size,2,mxREAL);
        plhs[0] = passign;

        double *pi = mxGetPr(passign);
        size_t i = 0;
        for (sparsevec::map_type::iterator it=p.map.begin(),itend=p.map.end();it!=itend;++it) {
            pi[i] = (double)(it->first)+1.0; // must shift 1 to convert from C indices to MATLAB indices
            pi[i + soln_size] = (double)(it->second);
            i++;
        }
    }

}
