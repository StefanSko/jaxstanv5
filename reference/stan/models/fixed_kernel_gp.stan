data {
  int<lower=1> N;
  vector[N] y;
  matrix[N, N] chol;
  real<lower=0> obs_sd;
}
parameters {
  vector[N] f;
}
model {
  f ~ multi_normal_cholesky(rep_vector(0, N), chol);
  y ~ normal(f, obs_sd);
}
