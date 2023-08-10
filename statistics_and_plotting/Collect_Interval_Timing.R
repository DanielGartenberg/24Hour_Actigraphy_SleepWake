# Collect Interval Timing
library(jsonlite)

data_dir <- '../data/'
study_dirs <- list.files(data_dir, pattern = '*json', full.names = TRUE)

date_to_hour <- function(date_str, tz = ''){
  posix_date <- as.POSIXct(date_str, origin ="1970-01-01", tz = tz)
  hours <- as.numeric(format(posix_date, '%H'))
  minutes <- as.numeric(format(posix_date, '%M'))
  seconds <- as.numeric(format(posix_date, '%S'))
  hours_total = hours + (minutes / 60) + (seconds / 3600)
  return(hours_total)
}

hour_mean <- function(hours_24){
  # adapted from: https://stackoverflow.com/questions/32404222/circular-mean-in-r
  
  radians_per_hour <- 2*pi/24
  # convert to radians
  hours_radians = hours_24 * radians_per_hour
  
  
  sinr <- sum(sin(hours_radians))
  cosr <- sum(cos(hours_radians))
  circular_mean_radians <- atan2(sinr, cosr)
  circular_mean_hours <- circular_mean_radians / radians_per_hour
  circular_mean_formatted <- (circular_mean_hours + 24) %% 24
  return(circular_mean_formatted)
}


for (st in study_dirs){
  study_data <- list.files(st, pattern = '*json', full.names = TRUE)
  study_df <- data.frame(f_name = study_data, time_min = NA, time_max = NA, sleep_start = NA, sleep_end = NA)
  if (st ==  "../data//MESA_json"){
    tz = 'UTC'
  } else{
    tz = "America/New_York"
  }

  
  for (sf in study_data){
    output_idx <- which(study_df$f_name == sf)
    data <- RJSONIO::fromJSON(sf)
    timing_min_max <- range(data$sleep_staging$epoch_start_time)
    study_df$time_min[output_idx] = date_to_hour(timing_min_max[1], tz = tz)
    study_df$time_max[output_idx] = date_to_hour(timing_min_max[2], tz = tz)
    
    #sleep_indices <- which(is.element(data$sleep_staging$epoch_stage_label, c('N1', 'N2', 'N3', 'REM')))
    # study_df$sleep_start[output_idx] = date_to_hour(data$sleep_staging$epoch_start_time[min(sleep_indices)])
    # study_df$sleep_end[output_idx] =  date_to_hour(data$sleep_staging$epoch_start_time[max(sleep_indices)])
  
  }
  
  start_mean <- hour_mean(study_df$time_min)
  end_mean <- hour_mean(study_df$time_max)
  print(st)
  print(sprintf('%i:%02i', floor(start_mean), round(start_mean %% 1 * 60)))
  # add half a minute (1 epoch length) to the end, as the final epoch time is the epoch staret
  end_mean = end_mean + (30 / 3600)
  # average to nearest minute
  end_mean = round(end_mean * 60) / 60
  print(sprintf('%i:%02i', floor(end_mean), round(end_mean %% 1 * 60)))
  
}

