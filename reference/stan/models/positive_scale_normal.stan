data {
  int<lower=0> N;
  vector[N] y;
  real prior_loc;
  real<lower=0> prior_scale;
}
parameters {
  real<lower=0> sigma;
}
model {
  sigma ~ normal(prior_loc, prior_scale);
  y ~ normal(0, sigma);
}
