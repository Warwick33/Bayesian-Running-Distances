
data {
  int<lower=1> N;       // Number of observations
  vector[N] y;          // Vector of run distances
}
parameters {
  // Mixture parameters
  real<lower=0, upper=1> pi; // Mixing proportion for component 1 (low variance)
  
  // Component 1 (Routine Runs / Low Variance) parameters
  real mu_1;
  real<lower=0.1> sigma_1;
  
  // Component 2 (Training Runs / High Variance) parameters
  real<lower=14> mu_2;
  // CRITICAL: Order constraint sigma_2 > sigma_1 for identifiability
  real<lower=sigma_1> sigma_2; 
}
model {
  // Priors
  pi ~ beta(1, 1);              // Uniform prior for mixing proportion
  
  // Priors for means (separate, based on domain knowledge)
  mu_1 ~ normal(10, 1.5); // Centered on typical routine run (e.g., 10km)
  mu_2 ~ normal(25, 2); // Centered on typical training run (e.g., half-marathon distance)
  
  // Half-Cauchy priors for standard deviations 
  sigma_1 ~ cauchy(0, 2);
  sigma_2 ~ cauchy(0, 4); // Prior on sigma_2 is conditional on sigma_1 > 0
  
  // Likelihood (Marginalized over component assignments)
  for (n in 1:N) {
    vector[2] log_lik_components;
    
    // Component 1: Routine Runs (pi, low variance)
    log_lik_components[1] = log(pi) + normal_lpdf(y[n] | mu_1, sigma_1);
    
    // Component 2: Training Runs (1-pi, high variance)
    log_lik_components[2] = log(1.0 - pi) + normal_lpdf(y[n] | mu_2, sigma_2);
    
    // Sum the likelihoods in log space to avoid underflow
    target += log_sum_exp(log_lik_components);
  }
}
generated quantities {
  // Post-hoc classification for each observation
  int z_pred[N];
  for (n in 1:N) {
    // Calculate log probabilities for component 1 and component 2
    real log_prob1 = log(pi) + normal_lpdf(y[n] | mu_1, sigma_1);
    real log_prob2 = log(1.0 - pi) + normal_lpdf(y[n] | mu_2, sigma_2);
    
    // Assign to component with higher probability
    if (log_prob1 > log_prob2) {
      z_pred[n] = 1;
    } else {
      z_pred[n] = 2;
    }
  }
}

