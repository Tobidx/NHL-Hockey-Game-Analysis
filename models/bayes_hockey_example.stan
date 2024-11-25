data {
    int<lower=1> N;            // number of matches
    int<lower=1> n_teams;            // number of teams
    array[N] int<lower=1, upper=n_teams> home_team; // home team index per game
    array[N] int<lower=1, upper=n_teams> away_team; // away team index per game
    array[N] int<lower=0> home_total_goals;  //home goals
    array[N] int<lower=0> away_total_goals; 
    
  
  }

parameters {
  real<lower=0> sigma;       // standard deviation for team abilities
  vector[n_teams] team_off;            // team abilities
  vector[n_teams] team_def; 
  real<lower=0>  alpha;  //intercept for log rate of goals per game per team
  real<lower=0> p_ha; // home advantage in goal rate per game
}


model {
  // Priors
  sigma ~ normal(0, 1);
  alpha ~ normal(1.2,0.5);
  team_off ~ normal(0, sigma);
  team_def ~ normal(0, sigma);
  p_ha ~ normal(0.05, 0.02); 
  
  // Likelihood
  for (n in 1:N) {
    real lam_home_rate =  exp(team_off[home_team[n]] - team_off[away_team[n]] + alpha + p_ha);
    real lam_away_rate =  exp( team_off[away_team[n]] - team_off[home_team[n]] + alpha - p_ha);
    home_total_goals[n] ~ poisson( lam_home_rate);
    away_total_goals[n] ~ poisson( lam_away_rate);
  }
}

generated quantities {
  vector[N]  lam_home_rate ;
  vector[N]  lam_away_rate ;
  for (n in 1:N) {
    lam_home_rate[n] = exp(team_off[home_team[n]] - team_off[away_team[n]]+ alpha + p_ha);
    lam_away_rate[n] =  exp( team_off[away_team[n]] - team_off[home_team[n]]+ alpha - p_ha);
  }
}
