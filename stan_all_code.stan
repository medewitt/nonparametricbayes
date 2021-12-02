data {
  int n;
  real x1[n];
  real x2[n];
  int y[n];
}

parameters {
  real beta0;
  real beta1;
  real beta2;
}

model {
  real mu[n];
  real prob[n];
  beta0 ~ normal(0,4);
  beta1 ~ normal(0,4);
  beta2 ~ normal(0,4);

  for(i in 1:n) {
    mu[i] = beta0 + beta1*x1[i] + beta2*x2[i];
    prob[i] = exp(mu[i])/(1+exp(mu[i]));
    target += bernoulli_lpmf(y[i] | prob[i]);
    }
}
