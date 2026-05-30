data {
  int<lower=1> N;
  vector[N] y;
  matrix[N, N] chol;
  real<lower=0> prior_scale;
}
parameters {
  vector[N] mu;
}
model {
  mu ~ normal(0, prior_scale);
  y ~ multi_normal_cholesky(mu, chol);
}
