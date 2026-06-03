data {
  int<lower=1> N;
  int<lower=0> N_obs;
  int<lower=0> N_mis;
  array[N_obs] int<lower=1, upper=N> observed_idx;
  array[N_mis] int<lower=1, upper=N> missing_idx;
  vector[N_obs] observed_values;
  matrix[N, N] chol;
}
parameters {
  vector[N_mis] y;
}
transformed parameters {
  vector[N] y_full;
  for (i in 1:N_obs) {
    y_full[observed_idx[i]] = observed_values[i];
  }
  for (i in 1:N_mis) {
    y_full[missing_idx[i]] = y[i];
  }
}
model {
  y_full ~ multi_normal_cholesky(rep_vector(0, N), chol);
}
