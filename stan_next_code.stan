data {
  int n;
  real x1[n];
  real x2[n];
  int y[n];
  vector[3] priormean;
  matrix[3,3] priorcovar;
}

parameters {
  vector[3] beta;
}

model {
  real mu[n];
  real prob[n];
  beta ~ multi_normal(priormean,priorcovar);
  for(i in 1:n) {
   mu[i] = beta[1] + beta[2]*x1[i] + beta[3]*x2[i];
   prob[i] = exp(mu[i])/(1+exp(mu[i]));
   target += bernoulli_lpmf(y[i] | prob[i]);
  }
}
