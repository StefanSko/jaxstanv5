data {
  int<lower=1> N;
  vector[N] y;
  real<lower=0> nu;
  real prior_loc;
  real<lower=0> prior_scale;
  real<lower=0> obs_scale;
}
parameters {
  real mu;
}
model {
  mu ~ normal(prior_loc, prior_scale);
  y ~ student_t(nu, mu, obs_scale);
}
