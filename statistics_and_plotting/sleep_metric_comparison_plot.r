library(ggplot2)
library(ggpmisc)
library(ggrepel)
library(dplyr)
library(scales)
debugSource('loa.R')

sleep_metric_comparison_plot <- function(data_df, x_var, y_var, group_category, group_colormap, variable_label, variable_units, tick_interval){
  base_font_size <- 8
  legend_font_size <- 10
  
  # all subsequent functions should ignore NA values, but remove just incase
  if (any(is.na(data_df[,y_var]))){
    num_na <- sum(is.na(data_df[,y_var]))
    print(sprintf('Removing %i NA rows for scatterplot', num_na))
    data_df <- data_df[!is.na(data_df[,y_var]), ]
  }
  
  # get axes limits
  x_values <- data_df[,x_var]
  y_values <- data_df[,y_var]
  combined_values <- c(x_values, y_values)
  axes_limits <- c(floor(min(combined_values, na.rm = TRUE)), ceiling(max(combined_values, na.rm = TRUE)))

  x_label_text <- sprintf('PSG %s (%s)', variable_label, variable_units)
  y_label_text <- sprintf('%s %s (%s)', group_category, variable_label, variable_units)

  plot_grob <- ggplot(data_df, aes(x=.data[[x_var]], y=.data[[y_var]])) +
    geom_abline(color="gray75", linetype="solid", size = rel(0.5)) + 
    geom_point(aes(color = data_set), alpha = 1, size=0.5, stroke=0.5, show.legend = TRUE, shape=1) + 
    geom_smooth(method='lm', se = FALSE, show.legend = FALSE, color='black', size=0.5) + 
    scale_x_continuous(breaks=scales::breaks_width(tick_interval)) +
    scale_y_continuous(breaks=scales::breaks_width(tick_interval)) +
    coord_fixed(ratio=1, xlim = axes_limits, ylim=axes_limits) +
    xlab(x_label_text) + ylab(y_label_text) + 
    ggtitle(sprintf('%s: True vs. Classification', variable_label)) + theme_bw(base_size = base_font_size) + scale_color_manual(values = group_colormap) +
    theme(axis.title=element_text(size=base_font_size), plot.title = element_text(face='bold', hjust = 0.5), legend.position = 'right', legend.background = element_rect(colour = 'black', fill = 'white', linetype='solid')) + 
    theme(plot.margin = margin(.2, .2, .2, .2, "cm")) + 
    theme(text=element_text(color="black"),axis.text=element_text(color="black")) +
    guides(color = guide_legend(nrow = 1, title.position = "top", title.hjust = 0.5, label.position = "right", override.aes = list(size=4))) +
    theme(legend.text = element_text(margin = margin(r = .25, unit = 'cm')), legend.spacing.x = unit(0.005, 'cm')) + 
    facet_wrap(vars(device), nrow = 1) + labs(color='Data Set')

  return(plot_grob)
}


sleep_metric_bland_altman_plot <- function(data_df, ref_var, comp_var, loa_type, group_category, group_colormap, bias_text_position, variable_label, variable_units, tick_interval, tick_label_min=-Inf, tick_label_max=Inf){
  base_font_size <- 8
  legend_font_size <- 10
  
  if (bias_text_position == 'top'){
    bias_text_x = -Inf
    bias_text_y = Inf
    bias_text_hjust = 0
    bias_text_vjust = 1.2
  } else if (bias_text_position == 'bottom'){
    bias_text_x = -Inf
    bias_text_y = -Inf
    bias_text_hjust = 0
    bias_text_vjust = -0.2
  } else {
    stop('unknown bias text position')
  }
  
  # create difference
  data_df$difference <- data_df[,comp_var] - data_df[,ref_var]
  
  # all subsequent functions should ignore NA values, but remove just incase
  if (any(is.na(data_df$difference))){
    num_na <- sum(is.na(data_df$difference))
    print(sprintf('Removing %i NA rows for bland-altman plot', num_na))
    data_df <- data_df[!is.na(data_df$difference), ]
  }
  
  x_label_text <- sprintf('PSG %s (%s)', variable_label, variable_units)
  y_label_text <- sprintf('%s - PSG %s (%s)', group_category, variable_label, variable_units)
  legend_title <- sprintf('%s:', group_category)
  legend_title <- ''
  
  # should limits of agreement be based on mixed-effects model, or not
  if (loa_type == 'mixed'){
    # mixed effects LoA for nested data
    # mixed LoA function expects the difference to have already been taken
    loa_list <- by(data_df, data_df$device, function(x) loa_mixed(x, 'difference', ref_var, 'participant_id'))
    for (n in 1:length(loa_list)){
      loa_list[[n]]$device <- names(loa_list)[n]
    }
    loa_df <- do.call(rbind, loa_list)
    
    # version without proportional bias to include mean bias text
    bias_list <- by(data_df, data_df$device, function(x) SimplyAgree::loa_lme('difference', ref_var, NULL, 'participant_id', x)$loa)
    for (n in 1:length(bias_list)){
      bias_list[[n]]$device <- names(bias_list)[n]
    }
    bias_df <- do.call(rbind, bias_list)
    
  } else {
    # non-mixed effects LOA for non-nested data
  
    loa_list <- by(data_df, data_df$device, function(x) loa_non_mixed(x, c(comp_var, ref_var), CI.type = 'boot'))
    for (n in 1:length(loa_list)){
      loa_list[[n]]$device <- names(loa_list)[n]
    }
    loa_df <- do.call(rbind, loa_list)
    
    # version without proportional bias to include mean bias text
    bias_list <- by(data_df, data_df$device, function(x) mean(x[,'difference'], na.rm = TRUE))
    bias_df <- data.frame(device = names(bias_list), term = 'Bias', estimate = bias_list[])
    
  }
  
  # get axes limits
  x_values <- data_df[,ref_var]
  y_values <- c(data_df[,'difference'], loa_df$lower.ci, loa_df$upper.ci)
  x_limits <- c(floor(min(x_values, na.rm = TRUE)), ceiling(max(x_values, na.rm = TRUE)))
  y_limits <- c(floor(min(y_values, na.rm = TRUE)), ceiling(max(y_values, na.rm = TRUE)))
  x_range <- diff(x_limits)
  y_range <- diff(y_limits)
  largest_range <- max(x_range, y_range)
  needed_x_range <- (largest_range - x_range) / 2
  needed_y_range <- (largest_range - y_range) / 2
  x_limits <- x_limits + c(needed_x_range * -1, needed_x_range)
  y_limits <- y_limits + c(needed_y_range * -1, needed_y_range)
  
  tick_label_limiter <- function(x) ifelse(x >= tick_label_min & x <= tick_label_max, x, '')
  
  plot_grob <- ggplot(data_df, aes(x=.data[[ref_var]], y=difference, color = data_set)) +
    geom_hline(yintercept = 0, color="gray75", linetype="solid", size = rel(0.5)) + 
    geom_point(alpha = 1, size=0.5, stroke=0.5, show.legend = TRUE, shape=1) +
    xlab(x_label_text) + ylab(y_label_text) + 
    geom_text(mapping = aes(label=sprintf("Mean Bias: %0.2f (%s)", estimate, variable_units)), color='black',
              x = bias_text_x, y = bias_text_y, hjust = bias_text_hjust, vjust = bias_text_vjust,
              size = 2, data = subset(bias_df, bias_df$term == 'Bias'), show.legend = FALSE) + 
    scale_x_continuous(breaks=scales::breaks_width(tick_interval), labels = tick_label_limiter) +
    scale_y_continuous(breaks=scales::breaks_width(tick_interval)) +
    coord_fixed(ratio=1, xlim = x_limits, ylim=y_limits) +
    ggtitle(sprintf('%s: True vs. Error', variable_label)) + theme_bw(base_size = base_font_size) + scale_color_manual(values = group_colormap) + scale_shape_discrete(name = legend_title) +
    theme(axis.title=element_text(size=base_font_size), plot.title = element_text(face='bold', hjust = 0.5), legend.position = 'right', legend.background = element_rect(colour = 'black', fill = 'white', linetype='solid')) + 
    theme(plot.margin = margin(.2, .2, .2, .2, "cm")) + 
    theme(text=element_text(color="black"),axis.text=element_text(color="black")) +
    guides(color = guide_legend(nrow = 1, title.position = "top", title.hjust = 0.5, label.position = "right", override.aes = list(size=4))) +
    theme(legend.text = element_text(margin = margin(r = .25, unit = 'cm')), legend.spacing.x = unit(0.005, 'cm')) + 
    facet_wrap(vars(device), nrow = 1) + labs(color='Data Set') +
    geom_ribbon(data = subset(loa_df, loa_df$term == 'Bias'), aes(x = ref, ymin = lower.ci, ymax = upper.ci, group = device), alpha = .15, show.legend = FALSE, inherit.aes = FALSE) + 
    geom_line(data = subset(loa_df, loa_df$term == 'Bias'), aes(x = ref, y = estimate, group = device), size=0.5, color='black', show.legend = FALSE) + 
    geom_ribbon(data = subset(loa_df, loa_df$term == 'Lower LoA'), aes(x = ref, ymin = lower.ci, ymax = upper.ci, group = device), alpha = .15, show.legend = FALSE, inherit.aes = FALSE) +
    geom_line(data = subset(loa_df, loa_df$term == 'Lower LoA'), aes(x = ref, y = estimate, group = device), linetype = 'dashed', size=0.5, color='black', show.legend = FALSE) + 
    geom_ribbon(data = subset(loa_df, loa_df$term == 'Upper LoA'), aes(x = ref, ymin = lower.ci, ymax = upper.ci, group = device), alpha = .15, show.legend = FALSE, inherit.aes = FALSE) + 
    geom_line(data = subset(loa_df, loa_df$term == 'Upper LoA'), aes(x = ref, y = estimate, group = device), linetype = 'dashed', size=0.5, color='black', show.legend = FALSE)

  return(plot_grob)
  
}
