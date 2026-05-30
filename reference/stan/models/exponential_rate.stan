data {
  int<lower=1> N;
  vector<lower=0>[N] y;
  real<lower=0> prior_scale;
}
parameters {
  real<lower=0> rate;
}
model {
  rate ~ normal(0, prior_scale);
  y ~ exponential(rate);
}
