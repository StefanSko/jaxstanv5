data {
  int<lower=1> N;
  int<lower=1> K;
  vector[N] x;
  array[N] int<lower=1, upper=K + 1> y;
}

parameters {
  real beta;
  ordered[K] cutpoints;
}

model {
  beta ~ normal(0, 1);
  cutpoints ~ normal(0, 2);

  for (n in 1:N) {
    y[n] ~ ordered_logistic(beta * x[n], cutpoints);
  }
}
