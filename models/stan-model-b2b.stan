data {
    int<lower=1> N;            // number of matches
    int<lower=1> n_teams;      // number of teams
    array[N] int<lower=1, upper=n_teams> home_team; // home team index per game
    array[N] int<lower=1, upper=n_teams> away_team; // away team index per game
    array[N] int<lower=0> home_total_goals;  // home goals
    array[N] int<lower=0> away_total_goals;  // away goals
    array[N] int<lower=0, upper=1> home_b2b; // home is playing back to back
    array[N] int<lower=0, upper=1> away_b2b; // away is playing back to back
}

parameters {
    real<lower=0> sigma;       // standard deviation for team abilities
    vector[n_teams] team_off;  // team offensive abilities
    vector[n_teams] team_def;  // team defensive abilities
    real<lower=0> alpha;       // intercept for log rate of goals per game per team
    real<lower=0> p_ha;        // home advantage in goal rate per game
    real<lower=-1> b2b_effect; // effect of playing back-to-back (should be negative)
}

model {
    // Priors
    sigma ~ normal(0, 1);
    alpha ~ normal(1.2, 0.5);
    team_off ~ normal(0, sigma);
    team_def ~ normal(0, sigma);
    p_ha ~ normal(0.05, 0.02);
    b2b_effect ~ normal(-0.1, 0.05);  // Prior suggesting negative effect of b2b games
    
    // Likelihood
    for (n in 1:N) {
        real lam_home_rate = exp(
            team_off[home_team[n]] - 
            team_def[away_team[n]] + 
            alpha + 
            p_ha +
            b2b_effect * home_b2b[n]  // Add b2b effect for home team
        );
        real lam_away_rate = exp(
            team_off[away_team[n]] - 
            team_def[home_team[n]] + 
            alpha - 
            p_ha +
            b2b_effect * away_b2b[n]  // Add b2b effect for away team
        );
        home_total_goals[n] ~ poisson(lam_home_rate);
        away_total_goals[n] ~ poisson(lam_away_rate);
    }
}

generated quantities {
    vector[N] lam_home_rate;
    vector[N] lam_away_rate;
    
    for (n in 1:N) {
        lam_home_rate[n] = exp(
            team_off[home_team[n]] - 
            team_def[away_team[n]] + 
            alpha + 
            p_ha +
            b2b_effect * home_b2b[n]
        );
        lam_away_rate[n] = exp(
            team_off[away_team[n]] - 
            team_def[home_team[n]] + 
            alpha - 
            p_ha +
            b2b_effect * away_b2b[n]
        );
    }
}