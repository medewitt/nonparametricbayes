functions{
int count_neighbours(int k, int p, real[] proposal, matrix prior, real[] beta_widths) {
  // identify neighbour prior draws
  int n_neighbours=0;
  int neighbour_p=0;
  for(i in 1:k) {
    neighbour_p=0;
    for(j in 1:p) {
      if(fabs(prior[i,j]-proposal[j])<(0.99*beta_widths[j])) neighbour_p+=1;
      // 0.99 because we dont want rounding error to nudge a prior draw out of the domain of beta_ldpf()
    }
    if(neighbour_p==p) {
      n_neighbours+=1;
    }
  }
return(n_neighbours);
}

int[] find_neighbours(int k, int p, real[] proposal, matrix prior, real[] beta_widths, int n_neighbours) {
  // identify neighbour prior draws
  int loopcount=0;
  int neighbour_prior[n_neighbours];
  for(i in 1:k) {
    int neighbour_p=0;
    for(j in 1:p) {
      if(fabs(prior[i,j]-proposal[j])<(0.99*beta_widths[j])) neighbour_p+=1;
      // 0.99 because we dont want rounding error to nudge a prior draw out of the domain of beta_ldpf()
    }
    if(neighbour_p==p) {
      loopcount+=1;
      neighbour_prior[loopcount]=i;
    }
  }
  return(neighbour_prior);
}
}

data {
  int n;
  int k; // draws in the prior (in the hypercuboid version, including the phony one)
  int p; // columns in the prior; this should NOT include __lp
  real<lower=0> betapars; // shared alpha and beta parameters for the beta distribution
  real<lower=0> beta_widths[p];
  matrix[k,p] prior;
  real x1[n];
  real x2[n];
  int y[n];
}

parameters {
  real beta[3]; // this is simpler as a vector
}

model {
  real mu[n];
  real prob[n];
  real logpriorprob;
  real priorprob;
  int n_neighbours = count_neighbours(k, p, beta, prior, beta_widths);
  int neighbour_prior[n_neighbours] = find_neighbours(k, p, beta, prior, beta_widths, n_neighbours);

  for(i in 1:n) {
    mu[i] = beta[1] + beta[2]*x1[i] + beta[3]*x2[i];
    prob[i] = exp(mu[i])/(1+exp(mu[i]));
    // increment log-posterior by log-likelihood
    target += bernoulli_lpmf(y[i] | prob[i]);
  }
  // increment log-posterior by log-prior
  priorprob = 0.0;
  for(i in 1:n_neighbours) {
    logpriorprob = 0.0;
    for(j in 1:p) {
      logpriorprob += beta_lpdf(0.5+(beta[j]-prior[neighbour_prior[i],j])/(2*beta_widths[j]) | betapars, betapars);
    }
    priorprob += exp(logpriorprob); // add all kernels densities (not log-densities) together
  }
  priorprob = priorprob / (k+0.0); // average over k draws in the prior
  target += log(priorprob); // add log-prior to log-likelihood!
}
