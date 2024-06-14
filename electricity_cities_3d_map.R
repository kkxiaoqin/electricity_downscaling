# Load required libraries
require(rayshader)
require(ggplot2)
require(viridis)
require(ggthemes)
require(terra)
require(sf)
require(ggspatial)
library(colorspace)
library(tigris)
library(stars)

# Read .gpkg data for China provinces
china <- st_read("./data/county_shp/province.gpkg")

# Read .gpkg data for cities
city <- st_read("./city.gpkg")

# Read electricity data from TIF file
electricity <- rast(r"./data/krige_output/final_result_demo.tif")

# Select specific provinces
cities <- china[china$省 %in% c("北京市", "河北省", "天津市"), ]

# Crop electricity data to the selected region
electricity <- terra::crop(electricity, terra::ext(cities))

# Mask the electricity data using the selected region
electricity <- mask(electricity, cities)

# Convert raster data to a dataframe
electricity_df <- as.data.frame(electricity, xy = TRUE)

# Rename dataframe columns
colnames(electricity_df) <- c("x", "y", "value")

# Replace all zero values with NA
electricity_df[electricity_df == 0] <- NA

# Visualize using ggplot2
colors <- scico::scico(n = 10, palette = "lajolla")
swatchplot(colors)
colors <- rev(colors)

# Convert colors to RGB and then to hexadecimal
rgb_values <- col2rgb(colors)
hex_values <- apply(rgb_values, 2, function(rgb) {
  sprintf("#%02X%02X%02X", rgb[1], rgb[2], rgb[3])
})

# Print hexadecimal values
print(hex_values)

# Create a color texture
texture <- grDevices::colorRampPalette(colors, bias=2)(256)
swatchplot(texture)

# Create ggplot visualization
gg <- ggplot() +
  geom_tile() +
  geom_sf(data = cities, fill = "#3b3f50", color = "grey25", linewidth = 0.15) +
  geom_raster(data = electricity_df, aes(x = x, y = y, fill = value)) +
  scale_fill_gradientn(colours = texture, na.value = '#00000000') +
  labs(fill = "Electricity (kWh)") +
  coord_sf() +
  theme_map() +
  theme(legend.position = "none") +
  annotation_scale(
    bar_cols = c("black", "white"),
    line_width = 1,
    height = unit(0.25, "cm"),
    pad_x = unit(0.25, "cm"),
    pad_y = unit(0.25, "cm"),
    text_pad = unit(0.15, "cm"),
    text_cex = 0.7,
    tick_height = 0.6
  ) +
  theme(legend.key.width = unit(1, "cm"))

# Display ggplot
gg

# Save ggplot as a TIFF file
tiff("./map/map_cities_electricity_jjj.tif",
     width = 8.8 * 2, height = 6 * 2, units = "cm", pointsize = 10,
     compression = "lzw", res = 600, type = "cairo", antialias = "subpixel")
gg
dev.off()

# Create 3D plot using plot_gg
gg3 <- plot_gg(gg,
               ggobj_height = gg,
               multicore = TRUE,
               width = 10,
               height = 10,
               units = "cm",
               scale = 350,
               shadow_darkness = 0.5,
               shadow_intensity = 0.8,
               triangulate = FALSE,
               height_aes = 'fill',
               windowsize = c(5000, 2500))

# Render high-quality 3D plot
render_highquality("./map/map_city_electricity_3d_jjj_2.tif",
                   width = 4096 * 0.75,
                   height = 4096 * 0.75,
                   parallel = TRUE,
                   progress = TRUE, 
                   ambient_light = TRUE,
                   camera_location = c(-126.00, 2282.85, 1218.86), 
                   print_scene_info = TRUE)
