#!/usr/bin/env Rscript

# Function to install packages if not already installed
install_if_missing <- function(package) {
  if (!require(package, character.only = TRUE, quietly = TRUE)) {
    cat(paste("Installing package:", package, "\n"))
    install.packages(package, repos = "https://cran.rstudio.com/")
  } else {
    cat(paste("Package", package, "is already installed.\n"))
  }
}

# List of required packages
packages <- c(
  # Core data manipulation
  "tidyverse",    # Data manipulation and visualization
  "data.table",   # Fast data manipulation
  "lubridate",    # Date/time manipulation
  
  # Time series analysis
  "forecast",     # Time series forecasting
  "prophet",      # Facebook's Prophet
  "vars",         # Vector Autoregression
  "tseries",      # Time series analysis
  "xts",          # Extended time series
  "zoo",          # Time series objects
  
  # Machine learning
  "caret",        # Classification and regression training
  "randomForest", # Random forest
  "gbm",          # Gradient boosting
  "e1071",        # Support vector machines
  
  # Visualization
  "ggplot2",      # Grammar of graphics
  "plotly",       # Interactive plots
  "RColorBrewer", # Color palettes
  
  # Statistical analysis
  "lmtest",       # Testing linear regression models
  "car",          # Companion to Applied Regression
  "MASS",         # Modern Applied Statistics
  
  # Integration with other systems
  "reticulate",   # R interface to Python
  "sparklyr",     # R interface to Spark
  
  # Optimization
  "ROI",          # R Optimization Infrastructure
  "ompr",         # Optimization modeling
  
  # Parallel processing
  "parallel",     # Parallel processing
  "doParallel",   # Parallel backend for foreach
  
  # Reporting
  "rmarkdown",    # Dynamic documents
  "knitr"         # Dynamic report generation
)

# Install packages
cat("Installing required R packages...\n")
for (package in packages) {
  install_if_missing(package)
}

cat("\nAll required R packages have been installed or were already present.\n") 