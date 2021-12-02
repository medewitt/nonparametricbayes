data {
  int n;
  int k; // number of non-zero-count hypercubes (bins) in the prior
  int k_draws; // draws in the prior
  int p; // columns in the prior; this should NOT include the count per hypercube
  int<lower=1> count[k]; // number of draws in each hypercube
  real x1[n];
  real x2[n];
  int y[n];
  matrix[k,p] prior;
  real bandwidth[p];
}

transformed data {
  int<lower=1> sum_count;
  sum_count = sum(count);
}

parameters {
  real beta[3]; // this is simpler as a vector
}
model {
  real mu[n];
  real prob[n];
  real logcubeprob;
  real priorprob;
  // no prior specified here, so they are uniform
  for(i in 1:n) {
    mu[i] = beta[1] + beta[2]*x1[i] + beta[3]*x2[i];
    prob[i] = exp(mu[i])/(1+exp(mu[i]));
    // increment log-posterior by log-likelihood
    target += bernoulli_lpmf(y[i] | prob[i]);
  }
  // increment log-posterior by log-prior
  priorprob = 0.0;
  for(i in 1:k) {
    logcubeprob = 0.0;
    for(j in 1:p) {
      logcubeprob += normal_lpdf(beta[j] | prior[i,j], bandwidth[j]); // multivariate distribution in parameter space
    }
    priorprob += count[i]*exp(logcubeprob); // add all kernels together, weighted by count
  
  }
  
  priorprob = priorprob / (sum_count+0.0); // weighted average over all draws in the prior
  target += log(priorprob); // add to log-likelihood!
}
