data {
  int n;
  int k; // draws in the prior
  int p; // columns in the prior; this should NOT include __lp, for example, which we will ignore
  real x1[n];
  real x2[n];
  int y[n];
  matrix[k,p] prior;
  real bandwidth[p];
}
parameters {
  real beta[p]; // this is simpler as a vector
}
model {
  real mu[n];
  real prob[n];
  real logpriorprob;
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
    logpriorprob = 0.0;
    for(j in 1:p) {
      logpriorprob += normal_lpdf(beta[j] | prior[i,j], bandwidth[j]);
    }
    priorprob += exp(logpriorprob); // add all kernels together
  }
  priorprob = priorprob / (k+0.0); // average over k draws in the prior
  target += log(priorprob); // add to log-likelihood!
}
