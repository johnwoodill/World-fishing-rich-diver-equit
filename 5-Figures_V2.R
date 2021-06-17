library(tidyverse)
library(marmap)
library(ggmap)
library(viridis)
library(mapdata)
library(sf)
library(raster)
library(cowplot)
library(gridExtra)
library(scales)
library(feather)

# ToDO ---------------------------------
# 1. Map of Global Fishing Effort (log fighing hours/vessels count)
# 2. Maps of Global Fishing Effort for Chinese, US, Spanish, TWN, and Japanese
# 3. Spatial overlap maps for CHN, USA, ESP, JPN, TWN
# 4. Maps of shannon diversity continous globally
# 5. Maps of shannon diversity binned globally - zero, low, med and high div
# 6. ESM data: mean and var SST, Chl

# 8. Fishing Hours error (prediction now - obs)
# 9. Fishing hours difference maps
# 10. Fishing Hours Regression Latitudinal Gradient 

# 12. Shannon Diversity error (prediction now - obs)
# 13. Shannon Diversity difference maps
# 14. Shannon Diversity Regression Latitudinal Gradient 


# 1. Table + text of Random forest deets



# Complete -------------------------------------------
# 7. Fishing Hours continuous regression predicted values now and the future

# 11. Shannon diversity classiciation predicted values now and the future


# -------------------------------------------------------






# -------------------------------------------------------


setwd("~/Projects/World-fishing-rich-diver-equit/")

# World polygons from the maps package
world_shp <- sf::st_as_sf(maps::map("world", wrap = c(0, 360), plot = FALSE, fill = TRUE))

# Load EEZ polygons
eezs <- read_sf("data/World_EEZ_v11_20191118_HR_0_360/", layer = "eez_v11_0_360") %>% 
  filter(POL_TYPE == '200NM') # select the 200 nautical mile polygon layer

mpas <- read_sf('data/mpa_shapefiles/vlmpa.shp')
mpas <- st_shift_longitude(mpas)

rfmos <- read_sf('data/RFMO_shapefile/RFMO_coords.shp')

# Specific color pallete
nipy_spectral <- c("#000000", "#6a009d", "#0035dd", "#00a4bb", "#009b0f",
                   "#00e100", "#ccf900", "#ffb000", "#e50000")






# ---------------------------------------------------------------------
# 1. Maps of Chinese, US, Spanish and Japanese fishing effort globally

fdat <- as.data.frame(read_csv("data/total_fishing_effort_nation.csv"))

fdat$lon <- ifelse(fdat$lon < 0, fdat$lon + 360, fdat$lon)
fdat$lon <- round(fdat$lon, 1)
fdat$lat <- round(fdat$lat, 1)
fdat$lat_lon <- paste0(fdat$lat, "_", fdat$lon)

fdat1 <- filter(fdat, flag_gfw %in% c("USA", "CHN", "JPN", "ESP"))
fdat1$flag <- ifelse(fdat1$flag_gfw == "USA", "United States", fdat1$flag_gfw)
fdat1$flag <- ifelse(fdat1$flag_gfw == "CHN", "China", fdat1$flag_gfw)
fdat1$flag <- ifelse(fdat1$flag_gfw == "ESP", "Spain", fdat1$flag_gfw)
fdat1$flag <- ifelse(fdat1$flag_gfw == "JPN", "Japan", fdat1$flag_gfw)


# Draw map
# World polygons from the maps package
world_shp <- sf::st_as_sf(maps::map("world", wrap = c(0, 360), plot = FALSE, fill = TRUE))

# Load EEZ polygons
eezs <- read_sf("data/World_EEZ_v11_20191118_HR_0_360/", layer = "eez_v11_0_360") %>% 
  filter(POL_TYPE == '200NM') # select the 200 nautical mile polygon layer


ggplot() +
  # geom_sf(data = world_shp, fill = 'black', color = 'black',  size = 0.1)  +
  geom_tile(data = fdat1, aes(lon, lat, fill=log(1 + fishing_hours)), alpha = 0.8) + 
  # geom_sf(data = eezs, color = '#374a6d', alpha = 0.2, fill = NA, size = 0.1) +
  scale_fill_gradientn(colors = nipy_spectral, limits = c(0, 9), breaks = seq(0, 9, 1)) +
  theme_minimal() +
  labs(x=NULL, y=NULL, fill=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = "right",
        legend.title.align=0.5,
        panel.spacing = unit(0.5, "lines"),
        panel.border = element_rect(colour = "black", fill=NA, size=.5)) +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
  guides(fill = guide_colorbar(title.position = "top", 
                               direction = "vertical",
                               frame.colour = "black",
                               barwidth = .5,
                               barheight = 20)) +
  facet_wrap(~flag) +
  NULL

#




ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_tile(data = fdat1, aes(lon, lat, fill=log(1 + fishing_hours)), alpha = 0.85, inherit.aes = FALSE) + 
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  geom_polygon()  +
  theme_map() +
  scale_fill_gradientn(colors = nipy_spectral, limits = c(0, 8), breaks = seq(0, 8, 1)) +
  labs(x=NULL, y=NULL, 
       title=NULL, fill="Fishing Effort by Country (log cumulative hours 2012-2016)") +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = "bottom",
        legend.title.align=0.5,
        panel.spacing = unit(0.5, "lines"),
        panel.border = element_rect(colour = "black", fill=NA, size=.5)) +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
  guides(fill = guide_colorbar(title.position = "top", 
                              direction = "horizontal",
                              frame.colour = "black",
                              barwidth = 39,
                              barheight = .75,
                              label.position = 'bottom')) +
  facet_wrap(~flag) +
  NULL

ggsave("figures/1-Map-Fishing-Effort-USA-CHN-ESP-JPN.png", width = 8, height = 5)





# ---------------------------------------------------------------------------
# 7. Fishing Hours continuous regression predicted values now and the future
# ---------------------------------------------------------------------------
ddat <- as.data.frame(read_csv("~/Projects/World-fishing-rich-diver-equit/data/NN_fishing_effort_regression_model_results.csv"))

ddat$diff <- ddat$y_ssp585_pred_2075 - ddat$y_pred_historical

ddat <- dplyr::select(ddat, lon, lat, y_pred_historical, y_ssp126_pred_2075, y_ssp370_pred_2075, y_ssp585_pred_2075)
names(ddat)[3:6] <- c("Historical Prediction", "2075-2090 SSP126 Prediction", 
                      "2075-2090 SSP370 Prediction", "2075-2090 SSP585 Prediction")

ddat <- gather(ddat, key=var, value = value, -lon, -lat)

ddat$var <- factor(ddat$var, levels = c("Historical Prediction", "2075-2090 SSP126 Prediction", 
                      "2075-2090 SSP370 Prediction", "2075-2090 SSP585 Prediction"))


ggplot() +
  geom_sf(data = world_shp, fill = 'black', color = 'black',  size = 0.1)  +
  geom_tile(data = ddat, aes(lon, lat, fill=value), alpha = 0.8) + 
  geom_sf(data = eezs, color = 'grey', alpha = 0.5, fill = NA, size = 0.2) +
  geom_sf(data = mpas, color = 'grey', alpha = 0.05, fill = NA, size = 0.1) +
  scale_fill_viridis(na.value = "#440154FF", breaks = seq(0, 1.7, 0.2), limits = c(0, 1.7)) +
  theme_minimal(12) +
  labs(x=NULL, y=NULL, fill=NULL, title = "Fishing Effort Neural Network Regression Predictions") +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = "right",
        legend.title.align=0.5,
        panel.spacing = unit(0.5, "lines"),
        panel.border = element_rect(colour = "black", fill=NA, size=.5),
        panel.background = element_rect(fill = "#440154FF"),
        panel.grid.minor=element_blank(),
        panel.grid.major=element_blank()) +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
  guides(fill = guide_colorbar(title.position = "top", 
                               direction = "vertical",
                               frame.colour = "black",
                               barwidth = .8,
                               barheight = 16)) +
  facet_wrap(~var, ncol=2) +
  NULL






# ---------------------------------------------------------------------------
# 11. Shannon diversity classiciation predicted values now and the future
# ---------------------------------------------------------------------------

ddat11 <- as.data.frame(read_csv("~/Projects/World-fishing-rich-diver-equit/data/RF_shannon_div_classification_model_results.csv"))

ddat11$diff <- ddat11$y_ssp585_pred_2075 - ddat11$y_pred_historical

ddat11 <- dplyr::select(ddat11, lon, lat, y_pred_historical, y_ssp126_pred_2075, y_ssp370_pred_2075, y_ssp585_pred_2075)
names(ddat11)[3:6] <- c("Historical Prediction", "2075-2090 SSP126 Prediction", 
                      "2075-2090 SSP370 Prediction", "2075-2090 SSP585 Prediction")

ddat11 <- gather(ddat11, key=var, value = value, -lon, -lat)

ddat11$var <- factor(ddat11$var, levels = c("Historical Prediction", "2075-2090 SSP126 Prediction", 
                      "2075-2090 SSP370 Prediction", "2075-2090 SSP585 Prediction"))


ddat11$value <- ifelse(ddat11$value == 0, "None", ddat11$value)
ddat11$value <- ifelse(ddat11$value == 1, "Low", ddat11$value)
ddat11$value <- ifelse(ddat11$value == 2, "Medium", ddat11$value)
ddat11$value <- ifelse(ddat11$value == 3, "High", ddat11$value)

ddat11$value <- factor(ddat11$value, levels = c("None", "Low", "Medium", "High"))

ggplot() +
  geom_sf(data = world_shp, fill = 'black', color = 'black',  size = 0.1)  +
  geom_tile(data = ddat11, aes(lon, lat, fill=factor(value)), alpha = 0.8) + 
  geom_sf(data = eezs, color = 'grey', alpha = 0.5, fill = NA, size = 0.2) +
  geom_sf(data = mpas, color = 'grey', alpha = 0.05, fill = NA, size = 0.1) +
  scale_fill_viridis_d() +
  theme_minimal(12) +
  labs(x=NULL, y=NULL, fill=NULL, title = "Shannon Diversity Multiclass Random-Forest Classification Predictions") +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = "bottom",
        legend.title.align=0.5,
        panel.spacing = unit(0.5, "lines"),
        panel.border = element_rect(colour = "black", fill=NA, size=.5),
        panel.background = element_rect(fill = "#440154FF"),
        panel.grid.minor=element_blank(),
        panel.grid.major=element_blank()) +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
  facet_wrap(~var, ncol=2) +
  NULL

