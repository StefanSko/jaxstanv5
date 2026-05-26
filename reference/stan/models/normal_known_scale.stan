data {
  int<lower=0> N;
  vector[N] y;
  real prior_loc;
  real<lower=0> prior_scale;
  real<lower=0> obs_scale;
}
parameters {
  real mu;
}
model {
  mu ~ normal(prior_loc, prior_scale);
  y ~ normal(mu, obs_scale);
}
