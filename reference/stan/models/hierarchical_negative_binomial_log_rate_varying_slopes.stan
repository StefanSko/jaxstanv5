data {
  int<lower=1> N;
  int<lower=1> G;
  array[N] int<lower=1, upper=G> group_idx;
  vector[N] x;
  array[N] int<lower=0> y;
}
parameters {
  real alpha_pop;
  real beta_pop;
  real<lower=0> sigma_alpha;
  real<lower=0> sigma_beta;
  real log_overdispersion;
  vector[G] z_alpha;
  vector[G] z_beta;
}
transformed parameters {
  vector[G] alpha;
  vector[G] beta;
  real<lower=0> overdispersion;

  alpha = alpha_pop + sigma_alpha * z_alpha;
  beta = beta_pop + sigma_beta * z_beta;
  overdispersion = exp(log_overdispersion);
}
model {
  alpha_pop ~ normal(0, 0.5);
  beta_pop ~ normal(0, 0.5);
  sigma_alpha ~ normal(0, 0.4);
  sigma_beta ~ normal(0, 0.4);
  log_overdispersion ~ normal(log(5), 0.5);
  z_alpha ~ normal(0, 1);
  z_beta ~ normal(0, 1);

  for (n in 1:N) {
    y[n] ~ neg_binomial_2(exp(alpha[group_idx[n]] + beta[group_idx[n]] * x[n]), overdispersion);
  }
}
