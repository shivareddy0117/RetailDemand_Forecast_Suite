#!/usr/bin/env Rscript

# Time Series Analysis in R
# This script provides specialized time series analysis functions that complement the Python implementation

# Load required libraries
library(tidyverse)
library(forecast)
library(tseries)
library(lubridate)
library(zoo)
library(ggplot2)
library(vars)

# Function to read data from CSV file
read_retail_data <- function(file_path) {
  data <- read.csv(file_path)
  
  # Convert date column to Date type if it exists
  if("date" %in% colnames(data)) {
    data$date <- as.Date(data$date)
  }
  
  return(data)
}

# Function to convert data frame to time series object
create_ts_object <- function(data, date_col = "date", value_col = "sales", frequency = 7) {
  # Extract the time series data
  ts_data <- data %>%
    arrange(!!sym(date_col)) %>%
    pull(!!sym(value_col))
  
  # Create time series object
  ts_obj <- ts(ts_data, frequency = frequency)
  
  return(ts_obj)
}

# Function to perform seasonal decomposition
perform_seasonal_decomposition <- function(ts_obj, type = "multiplicative") {
  # Perform decomposition
  decomp <- decompose(ts_obj, type = type)
  
  # Plot decomposition
  plot(decomp)
  
  return(decomp)
}

# Function to test for stationarity using Augmented Dickey-Fuller test
test_stationarity <- function(ts_obj) {
  # Perform ADF test
  adf_test <- adf.test(ts_obj)
  
  # Print results
  cat("Augmented Dickey-Fuller Test Results:\n")
  cat("Test statistic:", adf_test$statistic, "\n")
  cat("p-value:", adf_test$p.value, "\n")
  cat("Critical values:\n")
  print(adf_test$critical)
  
  # Interpret results
  if(adf_test$p.value < 0.05) {
    cat("The time series is stationary (reject H0)\n")
  } else {
    cat("The time series is non-stationary (fail to reject H0)\n")
  }
  
  return(adf_test)
}

# Function to perform ARIMA modeling
fit_arima_model <- function(ts_obj, seasonal = TRUE) {
  # Automatically select best ARIMA model
  if(seasonal) {
    model <- auto.arima(ts_obj, seasonal = TRUE)
  } else {
    model <- auto.arima(ts_obj, seasonal = FALSE)
  }
  
  # Print model summary
  cat("ARIMA Model Summary:\n")
  print(summary(model))
  
  # Plot diagnostics
  checkresiduals(model)
  
  return(model)
}

# Function to forecast using ARIMA model
forecast_arima <- function(model, h = 30, level = c(80, 95)) {
  # Generate forecast
  fc <- forecast(model, h = h, level = level)
  
  # Plot forecast
  plot(fc)
  
  return(fc)
}

# Function to fit VAR model
fit_var_model <- function(data, vars, p = 5) {
  # Select variables for VAR model
  var_data <- data %>%
    select(all_of(vars)) %>%
    as.ts()
  
  # Select optimal lag order
  lag_selection <- VARselect(var_data, lag.max = p, type = "const")
  optimal_p <- lag_selection$selection["AIC(n)"]
  
  cat("Optimal lag order according to AIC:", optimal_p, "\n")
  
  # Fit VAR model
  var_model <- VAR(var_data, p = optimal_p, type = "const")
  
  # Print summary
  cat("VAR Model Summary:\n")
  print(summary(var_model))
  
  return(var_model)
}

# Function to forecast using VAR model
forecast_var <- function(model, h = 30) {
  # Generate forecast
  fc <- predict(model, n.ahead = h)
  
  # Plot forecast
  plot(fc)
  
  return(fc)
}

# Function to perform Granger causality test
test_granger_causality <- function(var_model, cause, effect) {
  # Perform Granger causality test
  gc_test <- causality(var_model, cause = cause)
  
  # Print results
  cat("Granger Causality Test Results:\n")
  print(gc_test)
  
  return(gc_test)
}

# Function to perform impulse response analysis
analyze_impulse_response <- function(var_model, n.ahead = 20) {
  # Calculate impulse response functions
  irf <- irf(var_model, n.ahead = n.ahead)
  
  # Plot impulse response functions
  plot(irf)
  
  return(irf)
}

# Function to perform forecast error variance decomposition
analyze_fevd <- function(var_model, n.ahead = 10) {
  # Calculate FEVD
  fevd <- fevd(var_model, n.ahead = n.ahead)
  
  # Plot FEVD
  plot(fevd)
  
  return(fevd)
}

# Function to export results to CSV for use in Python
export_forecast_to_csv <- function(forecast_obj, file_path) {
  # Extract forecast data
  if(class(forecast_obj)[1] == "forecast") {
    # For ARIMA forecasts
    forecast_df <- data.frame(
      date = seq(from = Sys.Date(), by = "day", length.out = length(forecast_obj$mean)),
      mean = as.numeric(forecast_obj$mean),
      lower80 = as.numeric(forecast_obj$lower[,1]),
      upper80 = as.numeric(forecast_obj$upper[,1]),
      lower95 = as.numeric(forecast_obj$lower[,2]),
      upper95 = as.numeric(forecast_obj$upper[,2])
    )
  } else {
    # For VAR forecasts
    forecast_df <- as.data.frame(forecast_obj$fcst)
    forecast_df$date <- seq(from = Sys.Date(), by = "day", length.out = nrow(forecast_df))
  }
  
  # Write to CSV
  write.csv(forecast_df, file = file_path, row.names = FALSE)
  
  cat("Forecast exported to", file_path, "\n")
  
  return(forecast_df)
}

# Example usage (commented out)
# 
# # Read data
# data <- read_retail_data("data/sales_data.csv")
# 
# # Create time series object
# sales_ts <- create_ts_object(data, date_col = "date", value_col = "sales", frequency = 7)
# 
# # Perform seasonal decomposition
# decomp <- perform_seasonal_decomposition(sales_ts)
# 
# # Test for stationarity
# adf_result <- test_stationarity(sales_ts)
# 
# # Fit ARIMA model
# arima_model <- fit_arima_model(sales_ts, seasonal = TRUE)
# 
# # Generate forecast
# arima_fc <- forecast_arima(arima_model, h = 30)
# 
# # Export forecast to CSV
# export_forecast_to_csv(arima_fc, "results/arima_forecast.csv")
# 
# # Fit VAR model
# var_model <- fit_var_model(data, vars = c("sales", "temperature", "price"), p = 5)
# 
# # Generate VAR forecast
# var_fc <- forecast_var(var_model, h = 30)
# 
# # Export VAR forecast to CSV
# export_forecast_to_csv(var_fc, "results/var_forecast.csv") 