library(tidyverse)
library(marmap)
library(ggmap)
library(viridis)
library(mapdata)
library(sf)
library(raster)
library(cowplot)
library(gridExtra)



# ToDO

# 3. Network figure of spatial overlap in those areas (color coded by cluster)



# Complete -------------------------------------------

# 1. Maps of Chinese, US, Spanish and Japanese fishing effort globally
# 2. Spatial overlap maps, Heatmap matrixes of spatial overlap in Pacific RFMO and/or Patagonia shelf

# 4. Maps of species diversity and shannon diversity globally
# 5. ESM data: mean and var SST, Chl and one other globally
# 6. Species richness and/or shannon diversity binned into zero, low, med and high div
# 7. Predicted species richness and/or shannon diversity now and in the future
# 8. Classification error map (predicted species richness now - observed species richness)

# 10. Difference map: future div (modeled) - present div (observed)
# 11. RFMO statistics: those RFMOs with the number of competing nations going up/down
# -------------------------------------------------------


# -------------------------------------------------------
# 9. Table + text of Random forest deets
# Data Description:
#  	- GFW AIS: 10d resolution from 2012 - 2016
#  		- flag
#  		- mmsi
#  		- fishing hours
#  		- gear type
# 
#  	- Measure of Diversity (dependent variable binned into none, low, medium, high)
#  		- Richness - # of unique flags in a grid cell
#  		- Shannon Diversity - calculated from # obs in grid and shannon div function
# 
# 	- CMIP6: 1d resolution at 15-year averages (historical: 2000-2014, ssp126 projections: 2014 - 2090)
# 	    - ARAGOS: Aragonite concentration
#         - CHLOS: Mass Concentration of Total Phytoplankton expressed as Chlorophyll
#         - O2SATOS: Surface Dissolved Oxygen Concentration at Saturation
#         - PHOS: ph
#         - SOS: Sea Surface Salinity
#         - SIOS: Surface Total Dissolved Inorganic Silicon Concentration
#         - TOS: Sea Surface Temperature
#         - ZOOCOS: Surface Zooplankton Carbon Concentration
# 
#  	- Features (at each grid)
#  		- CMIP6 data (mean, min, max, variance, skewness, kurtosis)
#  		- lat/lon
#  		- distance to coast
#  		- distance to major port
#  		- located inside EEZ
#  		- located inside MPA
#  		- located in major RFMO
#  		- flag has been present before
#  		- gear type has been present before
# 
# Data Step:
# 	1. Regrid GFW data to 1d to match CMIP6 (sum at each grid and year)
# 	2. Calculate measure of diversity using GFW and average across time series
# 	3. Bind CMIP6 variables across grids and average every 15-year interval
# 	4. Calculate features of CMIP6 grids (distance to coast, etc)
# 	5. Bind GFW and CMIP6 data together by grids (1d lat/lon)
# 
# Model Step:
# 	1. Estimate Random Forest Classification for each measure of diversity
# 	2. Cross-validation Hyper-parameter tuning to increase accuracy (5-10% increase)
# 	3. 10-fold cross validation to measure train/test accuracy
# 	4. Train model on full data set
# 	5. Predict using CMIP6 projection every 15 years
# 
# Summary Stats:
# 	- 46,129 # obs (unique grids)
# 	- 188 # features
# 	- 122 # flags
# 	- Classification Accuracy
# 		- Richness - 84.31%
# 		- Shannon Div - 76.45%


# -------------------------------------------------------


setwd("~/Projects/World-fishing-rich-diver-equit/")

eezs <- read_sf('data/World_EEZ_v11_20191118_HR_0_360/eez_boundaries_v11_0_360.shp')

mpas <- read_sf('data/mpa_shapefiles/vlmpa.shp')
mpas <- st_shift_longitude(mpas)

rfmos <- read_sf('data/RFMO_shapefile/RFMO_coords.shp')

# Setup map
mp1 <- fortify(map(fill=TRUE, plot=FALSE))
mp2 <- mp1
mp2$long <- mp2$long + 360
mp2$group <- mp2$group + max(mp2$group) + 1
mp <- rbind(mp1, mp2)

nipy_spectral <- c("#000000", "#6a009d", "#0035dd", "#00a4bb", "#009b0f",
                   "#00e100", "#ccf900", "#ffb000", "#e50000", "#cccccc")



# ---------------------------------------------------------------------
# 1. Maps of Chinese, US, Spanish and Japanese fishing effort globally

fdat <- as.data.frame(read_csv("data/total_fishing_effort_nation_1d.csv"))

fdat$lon <- ifelse(fdat$lon < 0, fdat$lon + 360, fdat$lon)
fdat$lat_lon <- paste0(fdat$lat, "_", fdat$lon)

fdat1 <- filter(fdat, flag %in% c("USA", "CHN", "JPN", "ESP"))
fdat1$flag <- ifelse(fdat1$flag == "USA", "United States", fdat1$flag)
fdat1$flag <- ifelse(fdat1$flag == "CHN", "China", fdat1$flag)
fdat1$flag <- ifelse(fdat1$flag == "ESP", "Spain", fdat1$flag)
fdat1$flag <- ifelse(fdat1$flag == "JPN", "Japan", fdat1$flag)

fdat1 <- fdat1 %>% 
  group_by(lat_lon, flag) %>% 
  summarise(lat = mean(lat),
            lon = mean(lon),
            fishing_hours = sum(fishing_hours))



ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = fdat1, aes(lon, lat, color=log(1 + fishing_hours)), size = .05, inherit.aes = FALSE, shape=15, alpha = 0.5) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  geom_polygon()  +
  # scale_colour_gradientn(colours = rev(rainbow(5))) +
  scale_color_gradientn(colors = nipy_spectral) +
  theme_bw() +
  labs(x=NULL, y=NULL, 
       title="Fishing Effort by Country (log cumulative hours 2012-2016)", color=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = c(.965, 0.15),
        legend.box.background = element_rect(colour = "black")) +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
  guides(fill = FALSE,
       color = guide_colorbar(title.position = "top",
                              frame.colour = "black",
                              barwidth = .30,
                              barheight = 4,
                              label.position = 'left')) +
  facet_wrap(~flag) +
  NULL

ggsave("figures/1-Map-Fishing-Effort-USA-CHN-ESP-JPN.png", width = 8, height = 5)

# ---------------------------------------------------------------------



# 2. Spatial overlap maps, Heatmap matrixes of spatial overlap in Pacific RFMO and/or Patagonia shelf
# ---------------------------------------------------------------------
fdat2 <- as.data.frame(read_csv("data/total_fishing_effort_nation.csv"))
fdat2$lon <- ifelse(fdat2$lon < 0, fdat2$lon + 361, fdat2$lon)

flags_ <- c("CHN")

# fdat2 <- filter(fdat2, flag %in% flags_)
fdat2$lat_lon <- paste0(fdat2$lat, "_", fdat2$lon)
fdat2 <- filter(fdat2, fishing_hours > 0)

fdat2 %>% group_by(lat_lon, flag) %>% summarise()

for (i in 1:length(flags_)){
  flag_ = flags_[i]
  mdat <- filter(fdat2, flag == flag_ | flag == "USA") 
  mdat <- mdat %>% group_by(lat_lon) %>% mutate(nn = n())
  mdat$flag_overlap <- ifelse(mdat$nn == 2, paste0("USA-", flag_), mdat$flag)
  mdat$flag_overlap <- factor(mdat$flag_overlap, levels = c("USA", flag_, paste0("USA-", flag_)))
  mdat <- as.data.frame(mdat)
}



ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = mdat, aes(lon, lat, color=flag_overlap), size = .55, inherit.aes = FALSE, shape=15) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
    # First segment to the east
    geom_polygon()  + 
    scale_color_viridis_d() +
    theme_bw() +
    labs(x=NULL, y=NULL, 
         title=paste0("USA-", flag_, " Interactions"), color=NULL) +
    theme(plot.title = element_text(hjust = 0.5),
          legend.position = c(.936, 0.15),
          legend.box.background = element_rect(colour = "black"),
          plot.margin = unit(c(0, 0, 0, 0), "cm")) +
    coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
    guides(color = guide_legend(override.aes = list(size = 3))) +
    NULL
 
 
 ggsave("figures/2-Map-Spatial-Overlap-CHN-USA.png")
 
 # ---------------------------------------------------------------------
 
 


# 4. Maps of species diversity and shannon diversity globally
# ---------------------------------------------------------------------

fdat2r <- as.data.frame(read_csv('~/Projects/World-fishing-rich-diver-equit/data/total_species_richness_1d.csv'))
fdat2s <- as.data.frame(read_csv('~/Projects/World-fishing-rich-diver-equit/data/shannon_div_equ_1d.csv'))

fdat2r$lon <- ifelse(fdat2r$lon < 0, fdat2r$lon + 360, fdat2r$lon)
fdat2r$lat_lon <- paste0(fdat2r$lat, "_", fdat2r$lon)

fdat2s$lon <- ifelse(fdat2s$lon < 0, fdat2s$lon + 360, fdat2s$lon)
fdat2s$lat_lon <- paste0(fdat2s$lat, "_", fdat2s$lon)


p1 <- ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = fdat2r, aes(lon, lat, color=richness), size = .05, inherit.aes = FALSE, shape=15, alpha = 0.5) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  geom_polygon()  +
  scale_color_gradientn(colors = nipy_spectral[3:10]) +
  theme_bw() +
  labs(x=NULL, y=NULL, 
       title="Flag Richness", color=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(0, 0, 0, 0), "cm")) +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
  guides(fill = FALSE,
       color = guide_colorbar(title.position = "top",
                              frame.colour = "black",
                              barwidth = .35,
                              barheight = 9,
                              label.position = 'right')) +
  NULL

p2 <- ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = fdat2s, aes(lon, lat, color=H), size = .05, inherit.aes = FALSE, shape=15, alpha = 0.5) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  geom_polygon()  +
  scale_color_gradientn(colors = nipy_spectral[3:10]) +
  theme_bw() +
  labs(x=NULL, y=NULL, 
       title="Shannon Diversity", color=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(0, 0, 0, 0), "cm")) +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
  guides(fill = FALSE,
       color = guide_colorbar(title.position = "top",
                              frame.colour = "black",
                              barwidth = .35,
                              barheight = 9,
                              label.position = 'right')) +
  NULL

plot_grid(p1, p2, ncol=1)

ggsave("figures/4-Map-Species-Richness-and-Shannon-div.png")

# ---------------------------------------------------------------------







# 5. ESM data: mean and var SST, Chl and one other globally
# ---------------------------------------------------------------------

fdat5 <- as.data.frame(read_csv('~/Projects/World-fishing-rich-diver-equit/data/full_gfw_cmip_dat.csv'))

fdat5 <- dplyr::select(fdat5, lat, lon, `mean_tos_2000-2014`, `var_tos_2000-2014`,
                       `mean_chlos_2000-2014`, `var_chlos_2000-2014`)

color_scheme <- c("#000000","#18001b","#300037","#480052","#60006e","#770088","#7b008c","#7e008f","#810092","#850096","#850099",
  "#6a009d","#4e00a0","#3300a4","#1700a7","#0000ac","#0000b6","#0000c0","#0000ca","#0000d5","#0005dd","#001ddd",
  "#0035dd","#004ddd","#0065dd","#0079dd","#0080dd","#0086dd","#008ddd","#0094dd","#009ada","#009dd0","#00a1c5",
  "#00a4bb","#00a8b1","#00aaa8","#00aaa1","#00aa9a","#00aa93","#00aa8c","#00a97d","#00a562","#00a246","#009e2b",
  "#009b0f","#009c00","#00a300","#00aa00","#00b100","#00b800","#00be00","#00c500","#00cc00","#00d300","#00da00",
  "#00e100","#00e800","#00ef00","#00f500","#00fc00","#17ff00","#3cff00","#62ff00","#88ff00","#aeff00","#c2fd00",
  "#ccf900","#d6f600","#e1f200","#ebef00","#f0e900","#f4e200","#f7db00","#fbd500","#fece00","#ffc400","#ffba00",
  "#ffb000","#ffa500","#ff9b00","#ff8000","#ff6100","#ff4200","#ff2400","#ff0500","#f90000","#f20000","#eb0000",
  "#e50000","#de0000","#da0000","#d60000","#d30000","#d00000","#cc0000","#cc2727","#cc5050","#cc7a7a","#cca3a3",
  "#cccccc")


mean_sst <- ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = fdat5, aes(lon, lat, color=`mean_tos_2000-2014`), size = 0.5,  inherit.aes = FALSE) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  geom_polygon()  +
  scale_color_gradientn(colors = color_scheme, limits = c(-2, 33)) +
  theme_bw() +
  labs(x=NULL, y=NULL, 
       title="Average Sea-surface Temperature", color=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(0, 0, 0, 0), "cm")) +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
  guides(fill = FALSE,
       color = guide_colorbar(title.position = "top",
                              frame.colour = "black",
                              barwidth = .35,
                              barheight = 7,
                              label.position = 'right')) +
  NULL

var_sst <- ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = fdat5, aes(lon, lat, color=sqrt(`var_tos_2000-2014`)), size = 0.5,  inherit.aes = FALSE) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  geom_polygon()  +
  scale_color_gradientn(colors = nipy_spectral[3:10], limits = c(0, 5)) +
  theme_bw() +
  labs(x=NULL, y=NULL, 
       title="Standard Deviation Sea-surface Temperature", color=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(0, 0, 0, 0), "cm")) +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
  guides(fill = FALSE,
       color = guide_colorbar(title.position = "top",
                              frame.colour = "black",
                              barwidth = .35,
                              barheight = 7,
                              label.position = 'right')) +
  NULL

mean_chl <- ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = fdat5, aes(lon, lat, color= (`mean_chlos_2000-2014`)*10000000), size = 0.5, inherit.aes = FALSE) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  geom_polygon()  +
  scale_color_gradientn(colors = color_scheme[25:100], limits = c(0, 20)) +
  theme_bw() +
  labs(x=NULL, y=NULL, 
       title="Average Chlorophyll (mg/m3)", color=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(0, 0, 0, 0), "cm")) +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
  guides(fill = FALSE,
         color = guide_colorbar(title.position = "top",
                                frame.colour = "black",
                                barwidth = .35,
                                barheight = 7,
                                label.position = 'right')) +
  NULL


var_chl <- ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = fdat5, aes(lon, lat, color=sqrt(`var_chlos_2000-2014`)), size = 0.5, inherit.aes = FALSE) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  geom_polygon()  +
  scale_color_gradientn(colors =  nipy_spectral[3:10], limits = c(0, 0.000002), labels = c("0", "0.5", "1", "1.5", "2"), breaks = c(0, 0.0000005, 0.000001, 0.0000015, 0.000002)) +
  theme_bw() +
  labs(x=NULL, y=NULL, 
       title="Standard Deviation Chlorophyll", color=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(0, 0, 0, 0), "cm")) +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
  guides(fill = FALSE,
       color = guide_colorbar(title.position = "top",
                              frame.colour = "black",
                              barwidth = .35,
                              barheight = 7,
                              label.position = 'right')) +
  NULL

plot_grid(mean_sst, var_sst, mean_chl, var_chl, ncol=2)

ggsave("figures/5-Map-ESM-SST-CHL.png")

# ---------------------------------------------------------------------






# 6. Species richness and/or shannon diversity binned into zero, low, med and high div
# ---------------------------------------------------------------------

fdat6 <- as.data.frame(read_csv('~/Projects/World-fishing-rich-diver-equit/data/full_gfw_cmip_dat.csv'))
fdat6s <- as.data.frame(read_csv('data/shannon-diversity-bins.csv'))

fdat6$richness <- ifelse(is.na(fdat6$richness), 0, fdat6$richness)
fdat6$richness <- ifelse(fdat6$richness == 1, 1, fdat6$richness)
fdat6$richness <- ifelse(fdat6$richness > 1, ifelse(fdat6$richness <= 3, 2, fdat6$richness), fdat6$richness)
fdat6$richness <- ifelse(fdat6$richness > 3, 3, fdat6$richness)

# Shannon Diversity Cut
# Categories (3, interval[float64]): [(-0.000885, 0.499] < (0.499, 0.88] < (0.88, 2.464]]
# y          
# -0.0  28631
#  1.0   5808
#  2.0   5808
#  3.0   5808  

fdat6s %>% group_by(H) %>% summarise(nn = n())

rich <- ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = fdat6, aes(lon, lat, color=factor(richness)), size = 0.44, inherit.aes = FALSE, shape=15) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  geom_polygon()  +
  scale_color_manual(labels = c("None", "Low", "Medium", "High"), values = viridis(5)[2:5]) +
  theme_minimal() +
  labs(x=NULL, y=NULL,
       title="Flag Richness", color="Richness") +
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(0, 0, 0, 0), "cm")) +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
  guides(color = guide_legend(override.aes = list(size = 3))) +
  NULL
  



shannon <- ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = fdat6s, aes(lon, lat, color=factor(H)), size = 0.44, inherit.aes = FALSE, shape=15) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  geom_polygon()  +
  scale_color_manual(labels = c("None", "Low", "Medium", "High"), values = viridis(5)[2:5]) +
  theme_minimal() +
  labs(x=NULL, y=NULL,
       title="Shannon Diversity", color="Diversity") +
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(0, 0, 0, 0), "cm")) +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
  guides(color = guide_legend(override.aes = list(size = 3))) +
  NULL


plot_grid(rich, shannon, ncol=1)


ggsave("figures/6-Map-Richness-Shannon-Diversity.png")

# ---------------------------------------------------------------------






# 7. Predicted species richness and/or shannon diversity now and in the future
# ---------------------------------------------------------------------

fdat5 <- read_csv('data/rf_model_results.csv')

pred1 <- ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = fdat5, aes(lon, lat, color=factor(pred_2000_2014)), size = 0.22, inherit.aes = FALSE, shape=15) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  geom_polygon()  +
  scale_color_manual(labels = c("None", "Low", "Medium", "High"), values = viridis(5)[2:5]) +
  theme_bw() +
  labs(x=NULL, y=NULL,
       title="Predicted Flag Richness 2000-2014", color="Richness") +
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(.50, 0, 0, 0), "cm")) +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
  guides(color = guide_legend(override.aes = list(size = 3))) +
  NULL


pred2 <- ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = filter(fdat5, !is.na(pred_2075_2090)), aes(lon, lat, color=factor(pred_2075_2090)), size = 0.22, inherit.aes = FALSE, shape=15) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  geom_polygon()  +
  scale_color_manual(labels = c("None", "Low", "Medium", "High"), values = viridis(5)[2:5]) +
  theme_bw() +
  labs(x=NULL, y=NULL,
       title="Predicted Flag Richness 2075-2090", color="Richness") +
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(.50, 0, 0, 0), "cm")) +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
  guides(color = guide_legend(override.aes = list(size = 3))) +
  NULL

plot_grid(pred1, pred2, ncol=1)

ggsave("figures/7-Map-Predicted-Richness.png")


# ---------------------------------------------------------------------






# 8. Classification error map (predicted species richness now - observed species richness)
# ---------------------------------------------------------------------

fdat8 <- as.data.frame(read_csv("~/Projects/World-fishing-rich-diver-equit/data/richness_cross_validation_results.csv"))

fdat8$diff <- fdat8$y_pred - fdat8$y_true

nrow(filter(fdat8, diff == 0))/nrow(fdat8)

mean(unique(fdat8$score))

ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = fdat8, aes(lon, lat, color=factor(diff)), size = 0.44, inherit.aes = FALSE, shape=15) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  geom_polygon()  +
  scale_color_brewer(palette = "RdBu", "div", labels = c("-2", "-1", "0", "+1", "+2"), direction = -1) +
  theme_bw() +
  labs(x=NULL, y=NULL,
       title="Classification Error Map of Flag Richness", color="Richness") +
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(0, 0, 0, 0), "cm")) +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
  guides(color = guide_legend(override.aes = list(size = 3), title = "Error")) +
  NULL


ggsave("figures/8-Map-Richness-Classification-errors.png")






fdat8 <- as.data.frame(read_csv("~/Projects/World-fishing-rich-diver-equit/data/shannon_cross_validation_results.csv"))

fdat8$diff <- fdat8$y_pred - fdat8$y_true

nrow(filter(fdat8, diff == 0))/nrow(fdat8)

mean(unique(fdat8$score))
unique(fdat8$diff)

uneven_pal <- c("#3366AC", "#67A9CF", "#D1E5F0", "white", "#EF8A63", "#B2172B")


ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = filter(fdat8, !is.na(diff)), aes(lon, lat, color=factor(diff)), size = 0.44, inherit.aes = FALSE, shape=15) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  geom_polygon()  +
  scale_color_manual(values = uneven_pal, "div", labels = c("-3", "-2", "-1", "0", "+1", "+2")) +
  theme_bw() +
  labs(x=NULL, y=NULL,
       title="Classification Error Map of Shannon Diversity", color="Error") +
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(0, 0, 0, 0), "cm")) +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
  guides(color = guide_legend(override.aes = list(size = 3), title = "Error")) +
  NULL


ggsave("figures/8-Map-Shannon-Classification-errors.png")

# ---------------------------------------------------------------------







# ---------------------------------------------------------------------
# 10. Richness Difference map: future div (modeled) - present div (observed)

fdat10 <- as.data.frame(read_csv("~/Projects/World-fishing-rich-diver-equit/data/rf_model_results.csv"))
fdat10b <- as.data.frame(read_csv("~/Projects/World-fishing-rich-diver-equit/data/cross_validation_results.csv"))

fdat10b <- dplyr::select(fdat10b, lat, lon, y_true)

fdat10 <- left_join(fdat10, fdat10b, by = c("lat", "lon"))

fdat10$diff_2045 <- fdat10$pred_2030_2045 - fdat10$y_true
fdat10$diff_2090 <- fdat10$pred_2075_2090 - fdat10$y_true
fdat10$diff_2075 <- fdat10$pred_2075_2090 - fdat10$pred_2030_2045

fdat10 <- dplyr::select(fdat10, lat, lon, diff_2045, diff_2090, diff_2075) 

names(fdat10) <- c("lat", "lon", "2030-2045 Difference from Obs", "2075-2090 Difference from Obs.", "2075-2090 Difference from 2030-2045")

fdat10 <- gather(fdat10, key = diff, value = value, -lat, -lon)

unique(fdat10$diff)
unique(fdat10$value)

fdat10$diff <- factor(fdat10$diff, levels = c("2030-2045 Difference from Obs", "2075-2090 Difference from Obs.", "2075-2090 Difference from 2030-2045"))

ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = filter(fdat10, !is.na(value)), aes(lon, lat, color=factor(value)), size = 0.44, inherit.aes = FALSE, shape=15) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  geom_polygon()  +
  scale_color_brewer(palette = "RdBu", "div", labels = c("-2", "-1", "0", "+1", "+2"), direction = -1) +
  theme_bw() +
  labs(x=NULL, y=NULL,
       title="Flag Richness Predicted Differences", color="Diff. Diversity") +
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(0, 0, 0, 0), "cm"),
        legend.position = c(0.56, 0.22)) +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
  guides(color = guide_legend(override.aes = list(size = 3), title = "Diff. Richness")) +
  facet_wrap(~diff, ncol=2) +
  NULL



ggsave("figures/10-Map-Richness-Prediction-Differences.png")


# ---------------------------------------------------------------------


fdat10 <- as.data.frame(read_csv("~/Projects/World-fishing-rich-diver-equit/data/rf_shannon_model_results.csv"))
fdat10b <- as.data.frame(read_csv("~/Projects/World-fishing-rich-diver-equit/data/shannon_cross_validation_results.csv"))

fdat10b <- dplyr::select(fdat10b, lat, lon, y_true)

fdat10 <- left_join(fdat10, fdat10b, by = c("lat", "lon"))

fdat10$diff_2045 <- fdat10$pred_2030_2045 - fdat10$y_true
fdat10$diff_2090 <- fdat10$pred_2075_2090 - fdat10$y_true
fdat10$diff_2075 <- fdat10$pred_2075_2090 - fdat10$pred_2030_2045

fdat10 <- dplyr::select(fdat10, lat, lon, diff_2045, diff_2090, diff_2075) 

names(fdat10) <- c("lat", "lon", "2030-2045 Difference from Obs", "2075-2090 Difference from Obs.", "2075-2090 Difference from 2030-2045")

fdat10 <- gather(fdat10, key = diff, value = value, -lat, -lon)

unique(fdat10$diff)
unique(fdat10$value)

fdat10$diff <- factor(fdat10$diff, levels = c("2030-2045 Difference from Obs", "2075-2090 Difference from Obs.", "2075-2090 Difference from 2030-2045"))

ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = filter(fdat10, !is.na(value)), aes(lon, lat, color=factor(value)), size = 0.44, inherit.aes = FALSE, shape=15) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  geom_polygon()  +
  scale_color_brewer(palette = "RdBu", "div", labels = c("-3", "-2", "-1", "0", "+1", "+2", "+3"), direction = -1) +
  theme_bw() +
  labs(x=NULL, y=NULL,
       title="Shannon Diversity Predicted Differences", color="Diff. Diversity") +
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(0, 0, 0, 0), "cm"),
        legend.position = c(0.56, 0.22)) +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
  guides(color = guide_legend(override.aes = list(size = 3), title = "Diff. Diversity")) +
  facet_wrap(~diff, ncol=2) +
  NULL


ggsave("figures/10-Map-Shannon-Prediction-Differences.png")

# ---------------------------------------------------------------------









# 11. RFMO statistics: those RFMOs with the number of competing nations going up/down
# --------------------------------------------------------------------------------------------
fdat11 <- as.data.frame(read_csv("~/Projects/World-fishing-rich-diver-equit/data/rf_richness_model_results.csv"))
fdat11a <- as.data.frame(read_csv("~/Projects/World-fishing-rich-diver-equit/data/richness_cross_validation_results.csv"))
fdat11b <- as.data.frame(read_csv("~/Projects/World-fishing-rich-diver-equit/data/rfmo_check_dat.csv"))
rfmos <- read_sf('data/RFMO_shapefile/RFMO_coords.shp')

rfmos$RFMO <- ifelse(rfmos$RFMO == "ICCAT_West", "ICCAT", rfmos$RFMO) 
rfmos$RFMO <- ifelse(rfmos$RFMO == "ICCAT_East", "ICCAT", rfmos$RFMO)

fdat11$lat_lon <- paste0(fdat11$lat, "_", fdat11$lon)
fdat11a$lat_lon <- paste0(fdat11a$lat, "_", fdat11a$lon)
fdat11b$lat_lon <- paste0(fdat11b$lat, "_", fdat11b$lon)

fdat11a <- dplyr::select(fdat11a, lat_lon, y_true)
fdat11b <- dplyr::select(fdat11b, lat_lon, rfmo)

fdat11 <- left_join(fdat11, fdat11a, by = c("lat_lon"))
fdat11 <- left_join(fdat11, fdat11b, by = c("lat_lon"))

fdat11 <- filter(fdat11, rfmo != 0)

fdat11$diff_2045 <- fdat11$pred_2030_2045 - fdat11$y_true
fdat11$diff_2090 <- fdat11$pred_2075_2090 - fdat11$y_true
fdat11$diff_2075 <- fdat11$pred_2075_2090 - fdat11$pred_2030_2045

fdat11 <- dplyr::select(fdat11, lat, lon, rfmo, diff_2045, diff_2090, diff_2075) 

names(fdat11) <- c("lat", "lon", "rfmo", "2030-2045 Difference from Obs", "2075-2090 Difference from Obs.", "2075-2090 Difference from 2030-2045")

fdat11 <- gather(fdat11, key = diff, value = value, -lat, -lon, -rfmo)

fdat11 <- fdat11 %>% group_by(rfmo, diff) %>% summarise(lat = mean(lat),
                                              lon = mean(lon), 
                                              mean_value = mean(value, na.rm = TRUE),
                                              sd_value = sd(value, na.rm = TRUE),
                                              se_value = 1.96*(sd_value / sqrt(n())),
                                              min_value = mean_value - se_value,
                                              max_value = mean_value + se_value)

fdat11_2090 <- filter(fdat11, diff == "2075-2090 Difference from Obs.")

rfmos_2090 <- left_join(rfmos, fdat11_2090, by = c("RFMO" = "rfmo"))
rfmos_2090$lon <- ifelse(rfmos_2090$RFMO == "ICCAT", 320, rfmos_2090$lon)
rfmos_2090$lat <- ifelse(rfmos_2090$RFMO == "ICCAT", 30, rfmos_2090$lat)

ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = rfmos_2090, aes(fill = factor(round(mean_value, 4))), alpha = 0.5, size = 0.1, inherit.aes = FALSE) +
  geom_text(data = rfmos_2090, aes(lon, lat, label = RFMO), inherit.aes = FALSE) +
  geom_polygon()  +
  scale_fill_brewer(palette = "Reds") +
  theme_bw() +
  labs(x=NULL, y=NULL, 
       title="Average Increase in Flag Richness per grid (2075-2090 Difference from Obs.)", color=NULL, fill=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = "bottom") +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
  NULL







# --------------------------------------------------------------------------------------------
