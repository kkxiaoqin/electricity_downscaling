# Load necessary packages
library(atakrig)
library(terra)
library(rgdal)
library(sp)
library(rgeos)
library(dplyr)
library(lubridate)

# Define start and end dates
start_date <- as.Date("2012-04-01")
end_date <- as.Date("2019-12-01")

# Calculate the number of months
num_months <- interval(start_date, end_date) / months(1)

# Read the shapefile
shp_filename <- r"./data/residual_shp.shp"
china <- readOGR(shp_filename)

# Loop through each month
for (i in 0:(num_months - 1)) {
  # Calculate the current date
  current_date <- start_date + months(i)
  
  # Construct filenames
  date_str <- format(current_date, "%Y%m")
  
  csv_filename <- paste0(r"./data/krige_output/point_weight_centroid_demo.csv")
  output_filename <- paste0("E:\\phd\\Downscaling\\Data\\csv_data\\exp\\exp10\\predict\\krige\\predict_weight\\", date_str, "_residual.csv")
  
  # Read the CSV file
  data <- read.csv(csv_filename)
  
  # Calculate centroids for each area
  centroids <- coordinates(gCentroid(china, byid = TRUE))
  
  # Create a new dataframe for shapefile data
  areaValues <- data.frame(
    areaId = china@data$PAC,  # Replace with the field name representing the ID in your shapefile
    centx = centroids[, 1],
    centy = centroids[, 2],
    value = china@data[[paste0("X", date_str)]]  # Replace with the field name representing power in your shapefile
  ) %>%
    mutate(areaId = as.integer(areaId)) %>%
    filter(value != 0)
  
  # Select and rename necessary columns
  new_data <- data %>%
    select(district, centroid_x, centroid_y, weight, grid_id) %>%
    rename(
      areaId = district,
      ptx = centroid_x,
      pty = centroid_y
    ) %>%
    mutate(areaId = as.integer(areaId))
  
  # Filter new_data to only include rows with areaId present in areaValues
  filtered_newdata <- new_data %>%
    filter(areaId %in% areaValues$areaId) %>%
    arrange(areaId)
  
  # Prepare data for prediction
  points.results <- filtered_newdata[, c('areaId', 'ptx', 'pty', 'weight')]
  points_1.results <- points.results[, c('ptx', 'pty')]  # Used for prediction
  
  # Combine data into a list
  combined_list <- list(areaValues = areaValues, discretePoints = points.results)
  
  # Fit the semivariogram model
  sv.1.ok <- deconvPointVgm(combined_list, model = "Sph", maxIter = 100, fig = TRUE)
  
  # Perform kriging interpolation
  pred_1.ataok <- atpKriging(combined_list, points_1.results, sv.1.ok, showProgress = TRUE)
  
  # Modify the output values
  pred_1.ataok$grid_id <- filtered_newdata$grid_id
  pred_1.ataok$areaId <- filtered_newdata$areaId
  
  # Merge prediction results with original values
  results <- merge(pred_1.ataok, areaValues[c('areaId', 'value')], by = 'areaId', all.x = TRUE)
  results <- results[!duplicated(results), ]
  
  # Write results to CSV file
  write.csv(results, file = output_filename, row.names = FALSE)
}