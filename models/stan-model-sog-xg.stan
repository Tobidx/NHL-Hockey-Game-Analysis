data {
    int<lower=1> N;                // number of matches
    int<lower=1> n_teams;          // number of teams
    array[N] int<lower=1, upper=n_teams> home_team;  // home team index per game
    array[N] int<lower=1, upper=n_teams> away_team;  // away team index per game
    array[N] int<lower=0> home_total_sog;    // home shots on goal
    array[N] int<lower=0> away_total_sog;    // away shots on goal
    vector<lower=0>[N] home_total_xg;        // home expected goals
    vector<lower=0>[N] away_total_xg;        // away expected goals
    array[N] int<lower=0, upper=1> home_b2b; // home is playing back to back
    array[N] int<lower=0, upper=1> away_b2b; // away is playing back to back
}

parameters {
    // Parameters for shots on goal (SOG)
    real<lower=0.1> sigma_sog;         // std dev for team SOG abilities
    vector[n_teams] team_sog_off;      // team SOG offensive abilities
    vector[n_teams] team_sog_def;      // team SOG defensive abilities
    real<lower=0.1> alpha_sog;         // intercept for SOG rate
    real<lower=0> p_ha_sog;            // home advantage in SOG rate
    real<lower=-1,upper=0> b2b_effect_sog;  // b2b effect on SOG
    
    // Parameters for expected goals (xG)
    real<lower=0.1> sigma_xg;          // std dev for team xG abilities
    vector[n_teams] team_xg_off;       // team xG offensive abilities
    vector[n_teams] team_xg_def;       // team xG defensive abilities
    real<lower=0.1> alpha_xg;          // intercept for xG rate
    real<lower=0> p_ha_xg;             // home advantage in xG rate
    real<lower=-1,upper=0> b2b_effect_xg;    // b2b effect on xG
    real<lower=0.1> xg_sigma;          // observation noise for xG
}

model {
    // Priors for SOG parameters
    sigma_sog ~ normal(0.5, 0.25);
    alpha_sog ~ normal(2.0, 0.5);      // Prior centered around typical shots on goal
    team_sog_off ~ normal(0, sigma_sog);
    team_sog_def ~ normal(0, sigma_sog);
    p_ha_sog ~ normal(0.1, 0.05);
    b2b_effect_sog ~ normal(-0.1, 0.05);
    
    // Priors for xG parameters
    sigma_xg ~ normal(0.5, 0.25);
    alpha_xg ~ normal(1.0, 0.5);       // Prior centered around typical xG values
    team_xg_off ~ normal(0, sigma_xg);
    team_xg_def ~ normal(0, sigma_xg);
    p_ha_xg ~ normal(0.1, 0.05);
    b2b_effect_xg ~ normal(-0.1, 0.05);
    xg_sigma ~ normal(0.5, 0.25);
    
    // Likelihood for SOG
    for (n in 1:N) {
        real log_lambda_sog_home = team_sog_off[home_team[n]] - 
                                 team_sog_def[away_team[n]] + 
                                 alpha_sog + 
                                 p_ha_sog + 
                                 b2b_effect_sog * home_b2b[n];
                                 
        real log_lambda_sog_away = team_sog_off[away_team[n]] - 
                                 team_sog_def[home_team[n]] + 
                                 alpha_sog - 
                                 p_ha_sog + 
                                 b2b_effect_sog * away_b2b[n];
        
        home_total_sog[n] ~ poisson_log(log_lambda_sog_home);
        away_total_sog[n] ~ poisson_log(log_lambda_sog_away);
    }
    
    // Likelihood for xG
    for (n in 1:N) {
        real log_mu_xg_home = team_xg_off[home_team[n]] - 
                             team_xg_def[away_team[n]] + 
                             alpha_xg + 
                             p_ha_xg + 
                             b2b_effect_xg * home_b2b[n];
                             
        real log_mu_xg_away = team_xg_off[away_team[n]] - 
                             team_xg_def[home_team[n]] + 
                             alpha_xg - 
                             p_ha_xg + 
                             b2b_effect_xg * away_b2b[n];
        
        home_total_xg[n] ~ lognormal(log_mu_xg_home, xg_sigma);
        away_total_xg[n] ~ lognormal(log_mu_xg_away, xg_sigma);
    }
}

generated quantities {
    vector[N] lambda_sog_home;
    vector[N] lambda_sog_away;
    vector[N] mu_xg_home;
    vector[N] mu_xg_away;
    
    for (n in 1:N) {
        lambda_sog_home[n] = exp(
            team_sog_off[home_team[n]] - 
            team_sog_def[away_team[n]] + 
            alpha_sog + 
            p_ha_sog + 
            b2b_effect_sog * home_b2b[n]
        );
        lambda_sog_away[n] = exp(
            team_sog_off[away_team[n]] - 
            team_sog_def[home_team[n]] + 
            alpha_sog - 
            p_ha_sog + 
            b2b_effect_sog * away_b2b[n]
        );
        mu_xg_home[n] = exp(
            team_xg_off[home_team[n]] - 
            team_xg_def[away_team[n]] + 
            alpha_xg + 
            p_ha_xg + 
            b2b_effect_xg * home_b2b[n]
        );
        mu_xg_away[n] = exp(
            team_xg_off[away_team[n]] - 
            team_xg_def[home_team[n]] + 
            alpha_xg - 
            p_ha_xg + 
            b2b_effect_xg * away_b2b[n]
        );
    }
}