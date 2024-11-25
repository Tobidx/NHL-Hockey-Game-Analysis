data {
    int<lower=1> N;                // number of matches
    int<lower=1> n_teams;          // number of teams
    array[N] int<lower=1, upper=n_teams> home_team;  // home team index per game
    array[N] int<lower=1, upper=n_teams> away_team;  // away team index per game
    array[N] int<lower=0> home_total_goals;    // home goals
    array[N] int<lower=0> away_total_goals;    // away goals
    array[N] int<lower=0, upper=1> home_b2b;   // home is playing back to back
    array[N] int<lower=0, upper=1> away_b2b;   // away is playing back to back
    
    // Pre-computed abilities
    vector[n_teams] team_sog_off;  // team SOG offensive abilities
    vector[n_teams] team_xg_off;   // team xG offensive abilities
    vector[n_teams] team_sog_def;  // team SOG defensive abilities
    vector[n_teams] team_xg_def;   // team xG defensive abilities
}

parameters {
    real<lower=0> sigma;           // std dev for team abilities
    vector[n_teams] team_goal_off_std; // standardized team effects
    vector[n_teams] team_goal_def_std;
    real<lower=0> alpha;           // intercept for goal rate
    real<lower=0> p_ha;            // home advantage in goal rate
    real<lower=-1> b2b_effect;     // b2b effect on goals
    real<lower=0> beta_sog;        // SOG effect
    real<lower=0> beta_xg;         // xG effect
}

transformed parameters {
    vector[n_teams] team_goal_off = team_goal_off_std * sigma;
    vector[n_teams] team_goal_def = team_goal_def_std * sigma;
}

model {
    // Priors
    sigma ~ normal(0, 0.5);
    team_goal_off_std ~ std_normal();
    team_goal_def_std ~ std_normal();
    alpha ~ normal(1.2, 0.5);
    p_ha ~ normal(0.05, 0.02);
    b2b_effect ~ normal(-0.1, 0.05);
    beta_sog ~ normal(0.5, 0.25);
    beta_xg ~ normal(0.5, 0.25);
    
    // Likelihood
    for (n in 1:N) {
        real sog_diff_home = team_sog_off[home_team[n]] - team_sog_def[away_team[n]];
        real sog_diff_away = team_sog_off[away_team[n]] - team_sog_def[home_team[n]];
        real xg_diff_home = team_xg_off[home_team[n]] - team_xg_def[away_team[n]];
        real xg_diff_away = team_xg_off[away_team[n]] - team_xg_def[home_team[n]];
        
        real log_lambda_home = team_goal_off[home_team[n]] - 
                              team_goal_def[away_team[n]] + 
                              alpha + 
                              p_ha + 
                              b2b_effect * home_b2b[n] +
                              beta_sog * sog_diff_home +
                              beta_xg * xg_diff_home;
                              
        real log_lambda_away = team_goal_off[away_team[n]] - 
                              team_goal_def[home_team[n]] + 
                              alpha - 
                              p_ha + 
                              b2b_effect * away_b2b[n] +
                              beta_sog * sog_diff_away +
                              beta_xg * xg_diff_away;
        
        home_total_goals[n] ~ poisson_log(log_lambda_home);
        away_total_goals[n] ~ poisson_log(log_lambda_away);
    }
}

generated quantities {
    vector[N] lambda_home;
    vector[N] lambda_away;
    vector[N] log_lik; // Add log likelihood for model comparison
    
    for (n in 1:N) {
        real sog_diff_home = team_sog_off[home_team[n]] - team_sog_def[away_team[n]];
        real sog_diff_away = team_sog_off[away_team[n]] - team_sog_def[home_team[n]];
        real xg_diff_home = team_xg_off[home_team[n]] - team_xg_def[away_team[n]];
        real xg_diff_away = team_xg_off[away_team[n]] - team_xg_def[home_team[n]];
        
        real log_lambda_home = team_goal_off[home_team[n]] - 
                              team_goal_def[away_team[n]] + 
                              alpha + 
                              p_ha + 
                              b2b_effect * home_b2b[n] +
                              beta_sog * sog_diff_home +
                              beta_xg * xg_diff_home;
                              
        real log_lambda_away = team_goal_off[away_team[n]] - 
                              team_goal_def[home_team[n]] + 
                              alpha - 
                              p_ha + 
                              b2b_effect * away_b2b[n] +
                              beta_sog * sog_diff_away +
                              beta_xg * xg_diff_away;
        
        lambda_home[n] = exp(log_lambda_home);
        lambda_away[n] = exp(log_lambda_away);
        
        // Calculate log likelihood for model comparison
        log_lik[n] = poisson_lpmf(home_total_goals[n] | lambda_home[n]) +
                     poisson_lpmf(away_total_goals[n] | lambda_away[n]);
    }
}