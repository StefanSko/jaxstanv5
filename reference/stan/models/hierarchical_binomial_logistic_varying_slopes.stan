data {
  int<lower=1> N;
  int<lower=1> G;
  array[N] int<lower=1, upper=G> group_idx;
  vector[N] x;
  array[N] int<lower=0> trials;
  array[N] int<lower=0> y;
}
parameters {
  real alpha_pop;
  real beta_pop;
  real<lower=0> sigma_alpha;
  real<lower=0> sigma_beta;
  vector[G] z_alpha;
  vector[G] z_beta;
}
transformed parameters {
  vector[G] alpha;
  vector[G] beta;

  alpha = alpha_pop + sigma_alpha * z_alpha;
  beta = beta_pop + sigma_beta * z_beta;
}
model {
  alpha_pop ~ normal(0, 1);
  beta_pop ~ normal(0, 1);
  sigma_alpha ~ normal(0, 0.5);
  sigma_beta ~ normal(0, 0.5);
  z_alpha ~ normal(0, 1);
  z_beta ~ normal(0, 1);

  for (n in 1:N) {
    y[n] ~ binomial_logit(trials[n], alpha[group_idx[n]] + beta[group_idx[n]] * x[n]);
  }
}
