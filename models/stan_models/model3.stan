data {
    int<lower=1> N;          // number of matches
    int<lower=1> K;          // number of players
    array[N] int<lower=1, upper=K> white;
    array[N] int<lower=1, upper=K> black;
    array[N] int<lower=0, upper=2> result;  // 2=white win, 1=draw, 0=black win
    real<lower=1> theta_prior_mean;  // prior mean for theta
}

parameters {
    array[K] real skill_raw;  // non-centered parameterization
    real<lower=1> theta;      // threshold parameter (θ ≥ 1)
}

transformed parameters {
    array[K] real skill = skill_raw;
}

model {
    // Priors
    skill_raw ~ std_normal();  // implies skill ~ normal(0,3)
    theta ~ lognormal(log(theta_prior_mean), 0.5);
    
    // Rao-Kupper likelihood
    for (n in 1:N) {
        real gamma_w = exp(skill[white[n]]);
        real gamma_b = exp(skill[black[n]]);
        
        real denom_win = gamma_w + theta * gamma_b;
        real denom_loss = theta * gamma_w + gamma_b;
        
        if (result[n] == 2) {
            // White wins
            target += log(gamma_w) - log(denom_win);
        } else if (result[n] == 0) {
            // Black wins
            target += log(gamma_b) - log(denom_loss);
        } else {
            // Draw
            target += log(theta^2 - 1) + log(gamma_w) + log(gamma_b) - 
                     log(denom_win) - log(denom_loss);
        }
    }
}

generated quantities {
    array[N] real log_lik;
    array[K] real gamma = exp(skill);
    
    for (n in 1:N) {
        real gamma_w = gamma[white[n]];
        real gamma_b = gamma[black[n]];
        real denom_win = gamma_w + theta * gamma_b;
        real denom_loss = theta * gamma_w + gamma_b;
        
        if (result[n] == 2) {
            log_lik[n] = log(gamma_w) - log(denom_win);
        } else if (result[n] == 0) {
            log_lik[n] = log(gamma_b) - log(denom_loss);
        } else {
            log_lik[n] = log(theta^2 - 1) + log(gamma_w) + log(gamma_b) - 
                        log(denom_win) - log(denom_loss);
        }
    }
}