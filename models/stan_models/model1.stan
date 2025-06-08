data {
    int<lower=1> N;  // number of matches
    int<lower=1> K;  // number of players
    array[N] int<lower=1, upper=K> white;  // white player IDs
    array[N] int<lower=1, upper=K> black;  // black player IDs
    array[N] int<lower=0, upper=1> result;  // match results (1=white wins, 0=black wins)
    real prior_mu;  // prior mean for player skills
    real prior_sigma;  // prior std dev for player skills
}

parameters {
    array[K] real skill;  // latent skill parameters
}

model {
    // Priors
    for (k in 1:K) {
        skill[k] ~ normal(prior_mu, prior_sigma);
    }
    
    // Likelihood
    for (n in 1:N) {
        real skill_diff = skill[white[n]] - skill[black[n]];
        result[n] ~ bernoulli_logit(skill_diff);
    }
}
    
// For posterior predictive checks (evaluation)
generated quantities {
    array[N] int<lower=0, upper=1> pred_results;
    for (n in 1:N) {
        real skill_diff = skill[white[n]] - skill[black[n]];
        pred_results[n] = bernoulli_logit_rng(skill_diff);
    }
}