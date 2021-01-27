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

# ToDO

# 3. Network figure of spatial overlap in those areas (color coded by cluster)





# 13. Fishing Hours continuous regression predicted values now and the future

# 15. Fishing Hours error (prediction now - obs)
# 17. Fishing hours difference maps
# 19. Fishing Hours Regression Latitudinal Gradient change in flag-diversity



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


# 12. Shannon diversity continuous regression predicted values now and the future
# 13. Shannon Diversity error (prediction now - obs)
# 14. Shannon Diversity difference maps
# 15. Shannon Diversity Regression Latitudinal Gradient change in flag-diversity
# -------------------------------------------------------


# -------------------------------------------------------
# 9. Table + text of Random forest deets



# -------------------------------------------------------


setwd("~/Projects/World-fishing-rich-diver-equit/")

eezs <- read_sf('data/World_EEZ_v11_20191118_HR_0_360/eez_boundaries_v11_0_360.shp')

mpas <- read_sf('data/mpa_shapefiles/vlmpa.shp')
mpas <- st_shift_longitude(mpas)

rfmos <- read_sf('data/RFMO_shapefile/RFMO_coords.shp')

# Setup map
mp1 <- ggplot2::fortify(maps::map(fill=TRUE, plot=FALSE))
mp2 <- mp1
mp2$long <- mp2$long + 360
mp2$group <- mp2$group + max(mp2$group) + 1
mp <- rbind(mp1, mp2)

nipy_spectral <- c("#000000", "#6a009d", "#0035dd", "#00a4bb", "#009b0f",
                   "#00e100", "#ccf900", "#ffb000", "#e50000")


seascape_labels <- data.frame(seascape_class = seq(1, 33),
                              seascape_name = c("NORTH ATLANTIC SPRING \n ACC TRANSITION",
                                                    "SUBPOLAR TRANSITION",
                                                    "TROPICAL SUBTROPICAL TRANSITION",
                                                    "WESTERN WARM POOL SUBTROPICAL",
                                                    "SUBTROPICAL GYRE TRANSITION",
                                                    "ACC, NUTRIENT STRESS",
                                                    "TEMPERATE \n TRANSITION",
                                                    "INDOPACIFIC SUBTROPICAL GYRE",
                                                    "EQUATORIAL TRANSITION",
                                                    "HIGHLY OLIGOTROPHIC SUBTROPICAL GYRE",
                                                    "TROPICAL/SUBTROPICAL UPWELLING",
                                                    "SUBPOLAR",
                                                    "SUBTROPICAL GYRE MESOSCALE INFLUENCED",
                                                    "TEMPERATE BLOOMS \n UPWELLING",
                                                    "TROPICAL SEAS",
                                                    "MEDITTERANEAN RED SEA",
                                                    "SUBTROPICAL TRANSITION \n LOW NUTRIENT STRESS",
                                                    "MEDITTERANEAN RED SEA",
                                                    "ARTIC/ SUBPOLAR SHELVES",
                                                    "SUBTROPICAL, FRESH INFLUENCED COASTAL",
                                                    "WARM, BLOOMS \n HIGH NUTS",
                                                    "ARCTIC LATE SUMMER",
                                                    "FRESHWATER INFLUENCED POLAR SHELVES",
                                                    "ANTARCTIC SHELVES",
                                                    "ICE PACK",
                                                    "ANTARCTIC ICE EDGE",
                                                    "HYPERSALINE EUTROPHIC, \n PERSIAN GULF, RED SEA",
                                                    "ARCTIC ICE EDGE","ANTARCTIC",
                                                    "ICE EDGE  BLOOM",
                                                    "1-30% ICE PRESENT",
                                                    "30-80% MARGINAL ICE","PACK ICE"))





# ---------------------------------------------------------------------
# 1. Maps of Chinese, US, Spanish and Japanese fishing effort globally

fdat <- as.data.frame(read_csv("data/total_fishing_effort_nation.csv"))

fdat$lon <- ifelse(fdat$lon < 0, fdat$lon + 360, fdat$lon)
fdat$lon <- round(fdat$lon, 1)
fdat$lat <- round(fdat$lat, 1)
fdat$lat_lon <- paste0(fdat$lat, "_", fdat$lon)

fdat1 <- filter(fdat, flag %in% c("USA", "CHN", "JPN", "ESP"))
fdat1$flag <- ifelse(fdat1$flag == "USA", "United States", fdat1$flag)
fdat1$flag <- ifelse(fdat1$flag == "CHN", "China", fdat1$flag)
fdat1$flag <- ifelse(fdat1$flag == "ESP", "Spain", fdat1$flag)
fdat1$flag <- ifelse(fdat1$flag == "JPN", "Japan", fdat1$flag)



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

# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Fishing Effort and Shannon Diversity
fdat <- as.data.frame(read_csv("data/total_fishing_effort_nation.csv"))
fdat$lon <- ifelse(fdat$lon < 0, fdat$lon + 360, fdat$lon)
fdat$lon <- round(fdat$lon, 1)
fdat$lat <- round(fdat$lat, 1)
fdat$lat_lon <- paste0(fdat$lat, "_", fdat$lon)
fdat <- fdat %>% group_by(lat_lon) %>% summarise(lat = mean(lat), lon = mean(lon), fishing_hours = sum(fishing_hours))


sdat <- as.data.frame(read_csv("data/shannon_div_equ.csv"))
sdat$lon <- ifelse(sdat$lon < 0, sdat$lon + 360, sdat$lon)
sdat$lon <- round(sdat$lon, 1)
sdat$lat <- round(sdat$lat, 1)
sdat$lat_lon <- paste0(sdat$lat, "_", sdat$lon)
sdat <- sdat %>% group_by(lat_lon) %>% summarise(lat = mean(lat), lon = mean(lon), H = mean(H, na.rm=TRUE))


p1 <- ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_tile(data = fdat, aes(lon, lat, fill=log(1 + fishing_hours)), alpha = 0.65, inherit.aes = FALSE) + 
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  geom_polygon()  +
  scale_fill_gradientn(colors = nipy_spectral[c(1:8, 9, 9)], limits = c(0, 10), breaks = seq(0, 10, 2)) +
  theme_bw() +
  labs(x=NULL, y=NULL, 
       title="Total Fishing Effort (log cumulative hours 2012-2016)", fill=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = "right") +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
  guides(fill = guide_colorbar(title.position = "top",
                              frame.colour = "black",
                              barwidth = .30,
                              barheight = 6,
                              label.position = 'right')) +
  NULL


p2 <- ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.22, inherit.aes = FALSE) +
  # geom_point(data = sdat, aes(lon, lat, color=H), size = 0.233, inherit.aes = FALSE, shape=15, alpha=0.35) +
  geom_tile(data = sdat, aes(lon, lat, fill=H), alpha = 0.65, inherit.aes = FALSE) + 
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  geom_polygon()  +
  scale_fill_gradientn(colors = nipy_spectral[c(3:6, 7, 8, 9, 9, 9)]) +
  # scale_fill_gradientn(colors = nipy_spectral) +
  theme_bw() +
  labs(x=NULL, y=NULL, 
       title="Shannon Diversity (2012-2014)", color=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = 'right') +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
  guides(fill = guide_colorbar(title.position = "top",
                              frame.colour = "black",
                              barwidth = .30,
                              barheight = 6,
                              label.position = 'right')) +
  NULL

plot_grid(p1, p2, ncol=1)


ggsave("figures/1-Map-Fishing-Effort-Shannon-Diversity.png")
# ---------------------------------------------------------------------






# 2. Spatial overlap maps, Heatmap matrixes of spatial overlap in Pacific RFMO and/or Patagonia shelf
# ---------------------------------------------------------------------
fdat2 <- as.data.frame(read_csv("data/total_fishing_effort_nation.csv"))
fdat2$lon <- ifelse(fdat2$lon < 0, fdat2$lon + 361, fdat2$lon)

flags_ <- c("CHN", "JPN", "ESP", "CAN")

# fdat2 <- filter(fdat2, flag %in% flags_)
fdat2$lat_lon <- paste0(fdat2$lat, "_", fdat2$lon)
fdat2 <- filter(fdat2, fishing_hours > 0)

for (i in 1:length(flags_)){
  flag_ = flags_[i]
  mdat <- filter(fdat2, flag == flag_ | flag == "USA") 
  mdat <- mdat %>% group_by(lat_lon) %>% mutate(nn = n())
  mdat$flag_overlap <- ifelse(mdat$nn == 2, paste0("USA-", flag_), mdat$flag)
  mdat$flag_overlap <- factor(mdat$flag_overlap, levels = c("USA", flag_, paste0("USA-", flag_)))
  mdat <- as.data.frame(mdat)
  
  
p1 <- ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = mdat, aes(lon, lat, color=flag_overlap), size =0.54, inherit.aes = FALSE, shape=15) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  # First segment to the east
  geom_polygon()  + 
  scale_color_viridis_d() +
  theme_bw() +
  labs(x=NULL, y=NULL,
       title=paste0("USA-", flag_, " Interactions"), color=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
        legend.title=element_blank(),
        legend.position = c(0.5, 0.12),
        legend.text=element_text(size=5),
        # legend.margin=margin(c(1, 1, 1, 1), "pt"),
        legend.key.size = unit(.5, "cm")) +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand=FALSE) +
  guides(color = guide_legend(override.aes = list(size = 2), direction = "horizontal")) +
  NULL
  
  # ggsave("figures/test.png")
  saveRDS(p1, file = paste0("data/", flag_, "-USA.rds"))
  print(flag_)

}


p1 <- readRDS("data/CHN-USA.rds")
p2 <- readRDS("data/JPN-USA.rds")
p3 <- readRDS("data/ESP-USA.rds")
p4 <- readRDS("data/CAN-USA.rds")

plot_grid(p1, p2, p3, p4, ncol=2)

ggsave("figures/2-Map-Spatial-Overlap-USA-CHN-JPN-ESP-CAN.png")
 
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
  "#e50000","#de0000","#da0000","#d60000","#d30000","#d00000","#cc0000","#cc2727","#cc5050","#cc7a7a","#cca3a3")


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





# 5b. ESM data in the future: mean and var SST, Chl and one other globally
# ---------------------------------------------------------------------

fdat5 <- as.data.frame(read_csv('~/Projects/World-fishing-rich-diver-equit/data/full_dat_ssp126_2075_2090_dat.csv'))

fdat5 <- dplyr::select(fdat5, lat, lon, `mean_tos_2075-2090`, `var_tos_2075-2090`,
                       `mean_chlos_2075-2090`, `var_chlos_2075-2090`)

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
  geom_point(data = fdat5, aes(lon, lat, color=`mean_tos_2075-2090`), size = 0.5,  inherit.aes = FALSE) +
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
  geom_point(data = fdat5, aes(lon, lat, color=sqrt(`var_tos_2075-2090`)), size = 0.5,  inherit.aes = FALSE) +
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
  geom_point(data = fdat5, aes(lon, lat, color= (`mean_chlos_2075-2090`)*10000000), size = 0.5, inherit.aes = FALSE) +
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
  geom_point(data = fdat5, aes(lon, lat, color=sqrt(`var_chlos_2075-2090`)), size = 0.5, inherit.aes = FALSE) +
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

ggsave("figures/5b-Map-Projections-ESM-SST-CHL.png")

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
  geom_point(data = fdat6, aes(lon, lat, color=factor(richness)), size = 0.10, alpha = 0.15, inherit.aes = FALSE) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  geom_polygon()  +
  scale_color_manual(labels = c("None", "Low", "Medium", "High"), values = nipy_spectral[c(3, 6, 7, 9)]) +
  theme_bw() +
  labs(x=NULL, y=NULL,
       title="Flag Richness", color="Richness") +
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(0, 0, 0, 0), "cm")) +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
  guides(color = guide_legend(override.aes = list(size = 3, alpha = 0.5, shape = 15), reverse = TRUE)) +
  NULL
  



ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = fdat6s, aes(lon, lat, color=factor(H)), size = 0.45, alpha = 0.35, inherit.aes = FALSE) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  geom_polygon()  +
  scale_color_manual(labels = c("None", "Low", "Medium", "High"), values = nipy_spectral[c(3, 6, 7, 9)]) +
  # theme_bw() +
  labs(x=NULL, y=NULL,
       title="Shannon Diversity", color="Diversity") +
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(0, 0, 0, 0), "cm"),
        legend.key = element_rect(fill = NA)) +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
  guides(color = guide_legend(override.aes = list(size = 3, alpha = 0.5, shape = 15, fill=NA), reverse = TRUE)) +
  NULL


# plot_grid(rich, shannon, ncol=1)


ggsave("figures/6-Map-Shannon-Diversity-Binned.png")

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

fdat10 <- as.data.frame(read_csv("~/Projects/World-fishing-rich-diver-equit/data/rf_richness_model_results.csv"))
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
  # scale_color_brewer(palette = "RdBu", "div", labels = c("-2", "-1", "0", "+1", "+2"), direction = -1) +
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
  geom_point(data = filter(fdat10, !is.na(value) & value != 0), aes(lon, lat, color=factor(value)), size = 0.20, inherit.aes = FALSE, shape = 15) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  geom_polygon()  +
  # scale_color_brewer(palette = "RdBu", "div", labels = c("+3", "+2", "+1", "0", "-1", "-2", "-3"), direction = 1) +
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















# 12. Shannon diversity continuous regression predicted values now and the future
# --------------------------------------------------------------------------------------------

fdat12 <- as.data.frame(read_csv('data/rf_shannon_regression_model_results.csv'))

pred1 <- ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = fdat12, aes(lon, lat, color=(pred_2000_2014)), size = 0.215, alpha=0.3, inherit.aes = FALSE, shape=15) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  geom_polygon()  +
  scale_color_gradientn(colors = nipy_spectral[c(3:6, 8, 9, 9, 9)], limits = c(0, 2.0)) +
  theme_bw() +
  labs(x=NULL, y=NULL,
       title="Predicted Shannon Diversity 2000-2014", color=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(.50, 0, 0, 0), "cm")) +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
 guides(color = guide_colorbar(title.position = "top",
                      frame.colour = "black",
                      barwidth = .35,
                      barheight = 6,
                      label.position = 'right')) +
  NULL


pred2 <- ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = filter(fdat12, !is.na(pred_2075_2090*-1)), aes(lon, lat, color=(pred_2075_2090)), size = 0.215, alpha = 0.3, inherit.aes = FALSE, shape=15) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  geom_polygon()  +
  scale_color_gradientn(colors = nipy_spectral[c(3:6, 8, 9, 9, 9)], limits = c(0, 2.0)) +
  theme_bw() +
  labs(x=NULL, y=NULL,
       title="Predicted Shannon Diversity 2075-2090", color=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(.50, 0, 0, 0), "cm")) +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
  guides(color = guide_colorbar(title.position = "top",
                      frame.colour = "black",
                      barwidth = .35,
                      barheight = 6,
                      label.position = 'right')) +
  NULL

plot_grid(pred1, pred2, ncol=1)

ggsave("figures/12-Map-Predicted-Shannon-Div-Reg.png")






# --------------------------------------------------------------------------------------------
# Shannon diversity binned results



fdat12 <- as.data.frame(read_csv('data/rf_shannon_model_results.csv'))


pred1 <- ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = fdat12, aes(lon, lat, color=factor(pred_2000_2014)), size = 0.215, alpha=0.25, inherit.aes = FALSE, shape=15) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  geom_polygon()  +
  scale_color_manual(labels = c("None", "Low", "Medium", "High"), values = nipy_spectral[c(3, 6, 7, 9)]) +
  theme_bw() +
  labs(x=NULL, y=NULL,
       title="Predicted Shannon Diversity 2000-2014", color=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(.50, 0, 0, 0), "cm")) +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
  guides(color = guide_legend(override.aes = list(size = 3, shape = 15, alpha = 0.5), reverse = TRUE)) +
  NULL



pred2 <- ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = filter(fdat12, !is.na(pred_2030_2045)), aes(lon, lat, color=factor(pred_2030_2045)), size = 0.215, alpha = 0.25, inherit.aes = FALSE, shape=15) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  geom_polygon() +
  scale_color_manual(labels = c("None", "Low", "Medium", "High"), values = nipy_spectral[c(3, 6, 7, 9)], breaks = c(0, 1, 3, 2)) +
  theme_bw() +
  labs(x=NULL, y=NULL,
       title="Predicted Shannon Diversity 2030-2045", color=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(.50, 0, 0, 0), "cm")) +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
  guides(color = guide_legend(override.aes = list(size = 3, shape = 15, alpha = 0.5), reverse = TRUE)) +
  NULL



plot_grid(pred1, pred2, ncol=1)

ggsave("figures/12-Map-Predicted-Shannon-Div-Classification.png")

# ------------------------------------------------------
















# --------------------------------------------------------------------------------------------
# 13. Shannon Diversity error (prediction now - obs)
fdat13 <- as.data.frame(read_csv("~/Projects/World-fishing-rich-diver-equit/data/shannon-regression_cross_validation_results.csv"))

fdat13$diff <- fdat13$y_pred - fdat13$y_true

nrow(filter(fdat13, diff == 0))/nrow(fdat13)

mean(fdat13$r2_test)
min(fdat13$diff)
max(fdat13$diff)

ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = fdat13, aes(lon, lat, color=(diff)), size = 0.44, inherit.aes = FALSE, shape=15) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  geom_polygon()  +
  scale_color_gradientn(colors =  rev(c("#1171B0","white","#CA0020")), values = rescale(c(-1.395429, 0, 1.108251)), limits = c(-1.395429, 1.108251)) +
  theme_bw() +
  labs(x=NULL, y=NULL,
       title="Shannon Diversity Regression Error Map", color=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(0, 0, 0, 0), "cm")) +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
  guides(color = guide_colorbar(title.position = "top",
                      frame.colour = "black",
                      barwidth = .35,
                      barheight = 17,
                      label.position = 'right')) +
  NULL


ggsave("figures/13-Map-Shannon-Diversity-Regression-errors.png")


# --------------------------------------------------------------------------------------------







# --------------------------------------------------------------------------------------------
# 14. Shannon Diversity difference maps
fdat14 <- as.data.frame(read_csv("~/Projects/World-fishing-rich-diver-equit/data/rf_shannon_regression_model_results.csv"))
fdat14b <- as.data.frame(read_csv("~/Projects/World-fishing-rich-diver-equit/data/shannon-regression_cross_validation_results.csv"))

fdat14b <- dplyr::select(fdat14b, lat, lon, y_true)

fdat14 <- left_join(fdat14, fdat14b, by = c("lat", "lon"))

fdat14$diff_2045 <- fdat14$pred_2030_2045 - fdat14$pred_2000_2014
fdat14$diff_2090 <- fdat14$pred_2075_2090 - fdat14$pred_2000_2014
fdat14$diff_2075 <- fdat14$pred_2075_2090 - fdat14$pred_2030_2045

fdat14 <- dplyr::select(fdat14, lat, lon, diff_2045, diff_2090, diff_2075) 

names(fdat14) <- c("lat", "lon", "2030-2045 Difference from 2000-2014", "2075-2090 Difference from 2000-2014", "2075-2090 Difference from 2030-2045")

fdat14 <- gather(fdat14, key = diff, value = value, -lat, -lon)

unique(fdat14$diff)
unique(fdat14$value)
min(fdat14$value, na.rm = TRUE)
max(fdat14$value, na.rm = TRUE)

fdat14$diff <- factor(fdat14$diff, levels = c("2030-2045 Difference from 2000-2014", "2075-2090 Difference from 2000-2014", "2075-2090 Difference from 2030-2045"))



ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = filter(fdat14, !is.na(value) | (value >= -0.5 & value <= 0.5)), aes(lon, lat, color=(value)), size = 0.64, inherit.aes = FALSE, shape=15) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  geom_polygon()  +
  scale_color_gradientn(colors =  (c("#1171B0","white","#CA0020")), values = rescale(c(-.25, 0, .25)), limits = c(-.25, .25)) +
  theme_bw() +
  labs(x=NULL, y=NULL,
       title="Shannon Diversity Regression Predicted Differences", color=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(0, 0, 0, 0), "cm"),
        legend.position = c(0.56, 0.22)) +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
    guides(color = guide_colorbar(title.position = "top",
                      frame.colour = "black",
                      barwidth = .35,
                      barheight = 8,
                      label.position = 'right')) +
  facet_wrap(~diff, ncol=2) +
  NULL



ggsave("figures/14-Map-Shannon-Regression-Differences.png")

# --------------------------------------------------------------------------------------------











# --------------------------------------------------------------------------------------------
# 15. Shannon Diversity Latitudinal Gradient change in flag-diversity

fdat15 <- as.data.frame(read_csv("~/Projects/World-fishing-rich-diver-equit/data/rf_shannon_regression_model_results.csv"))
fdat15b <- as.data.frame(read_csv("~/Projects/World-fishing-rich-diver-equit/data/shannon-regression_cross_validation_results.csv"))

fdat15b <- dplyr::select(fdat15b, lat, lon, y_true)

fdat15 <- left_join(fdat15, fdat15b, by = c("lat", "lon"))

fdat15$diff_2045 <- fdat15$pred_2030_2045 - fdat15$y_true
fdat15$diff_2090 <- fdat15$pred_2075_2090 - fdat15$y_true
fdat15$diff_2075 <- fdat15$pred_2075_2090 - fdat15$pred_2030_2045

fdat15 <- dplyr::select(fdat15, lat, lon, diff_2045, diff_2090, diff_2075) 

names(fdat15) <- c("lat", "lon", "2030-2045 Difference from Obs", "2075-2090 Difference from Obs.", "2075-2090 Difference from 2030-2045")

fdat15 <- gather(fdat15, key = diff, value = value, -lat, -lon)

unique(fdat15$diff)
unique(fdat15$value)

fdat15$diff <- factor(fdat15$diff, levels = c("2030-2045 Difference from Obs", "2075-2090 Difference from Obs.", "2075-2090 Difference from 2030-2045"))

fdat20 <- fdat15 %>% group_by(lat, diff) %>% summarise(grad = mean(value),
                                                       grad_sd = sd(value),
                                                       se = 1.96*(grad_sd^2/sqrt(n())),
                                                       grad_min = grad - se,
                                                       grad_max = grad + se)

fdat20$div <- ifelse(fdat20$grad < 0, 1, 0)

fdat20 <- filter(fdat20, diff != "2075-2090 Difference from 2030-2045")

ggplot(fdat20, aes(lat, grad*-1, color=factor(div))) + 
  geom_point(size = 0.8) +
  geom_errorbar(aes(ymin=grad_min*-1, ymax=grad_max*-1), width = 1) +
  theme_bw() + 
  scale_color_manual(values = c("#67A9CF", "#EF8A61")) +
  labs(x="Latitude", y="Average Longitudinal Shannon Diversity") +
  scale_x_continuous(breaks = seq(-90, 90, 10), limits = c(-80, 80)) +
  scale_y_continuous(breaks = seq(-0.1, 0.1, 0.1), limits = c(-0.1, 0.1)) +
  theme(legend.position = "none") +
  facet_wrap(~diff, ncol=1, scales="free") +
  NULL


ggsave("figures/15-Shannon-Diversity-Latitudinal-Gradient.png")





fdat15 <- as.data.frame(read_csv("~/Projects/World-fishing-rich-diver-equit/data/rf_shannon_model_results.csv"))
fdat15b <- as.data.frame(read_csv("~/Projects/World-fishing-rich-diver-equit/data/shannon_cross_validation_results.csv"))

fdat15b <- dplyr::select(fdat15b, lat, lon, y_true)

fdat15 <- left_join(fdat15, fdat15b, by = c("lat", "lon"))

fdat15$pred_2030_2045 <- ifelse(fdat15$pred_2030_2045 == 1, 200, fdat15$pred_2030_2045)
fdat15$pred_2030_2045 <- ifelse(fdat15$pred_2030_2045 == 2, 300, fdat15$pred_2030_2045)
fdat15$pred_2030_2045 <- ifelse(fdat15$pred_2030_2045 == 3, 200, fdat15$pred_2030_2045)
fdat15$pred_2030_2045 <- fdat15$pred_2030_2045/100

fdat15$diff_2045 <- fdat15$pred_2030_2045 - fdat15$pred_2000_2014
fdat15$diff_2090 <- fdat15$pred_2075_2090 - fdat15$pred_2000_2014
fdat15$diff_2075 <- fdat15$pred_2075_2090 - fdat15$pred_2030_2045

fdat15 <- dplyr::select(fdat15, lat, lon, diff_2045, diff_2090, diff_2075) 

names(fdat15) <- c("lat", "lon", "2030-2045 Difference from Obs", "2075-2090 Difference from Obs.", "2075-2090 Difference from 2030-2045")

fdat15 <- gather(fdat15, key = diff, value = value, -lat, -lon)

unique(fdat15$diff)
unique(fdat15$value)

fdat15$diff <- factor(fdat15$diff, levels = c("2030-2045 Difference from Obs", "2075-2090 Difference from Obs.", "2075-2090 Difference from 2030-2045"))

fdat15 <- fdat15 %>% group_by(lat, diff) %>% summarise(grad = mean(value),
                                             grad_sd = sd(value),
                                             se = 1.96*(grad_sd^2/sqrt(n())),
                                             grad_min = grad - se,
                                             grad_max = grad + se)

fdat15$div <- ifelse(fdat15$grad < 0, 1, 0)

fdat15 <- filter(fdat15, diff != "2075-2090 Difference from 2030-2045")

ggplot(fdat15, aes(lat, grad*-1, color=factor(div))) + 
  geom_point(size = 0.8) +
  geom_errorbar(aes(ymin=grad_min*-1, ymax=grad_max*-1), width = 1) +
  theme_bw() + 
  scale_color_manual(values = c("#67A9CF", "#EF8A61")) +
  labs(x="Latitude", y="Average Longitudinal Shannon Diversity") +
  scale_x_continuous(breaks = seq(-90, 90, 10), limits = c(-80, 80)) +
  scale_y_continuous(breaks = seq(-0.5, 0.5, 0.1), limits = c(-0.5, 0.5)) +
  theme(legend.position = "none") +
  facet_wrap(~diff, ncol=1, scales="free") +
  NULL


ggsave("figures/15-Shannon-Diversity-Classification-Latitudinal-Gradient.png")

# ------------------------------------------------------













# 16. Fishing Hours continuous regression predicted values now and the future
# --------------------------------------------------------------------------------------------

fdat16 <- as.data.frame(read_csv('data/rf_fishing_effort_regression_model_results.csv'))

pred1 <- ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = fdat16, aes(lon, lat, color=(pred_2000_2014)), size = 0.488, alpha = 0.20, inherit.aes = FALSE, shape=15) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  geom_polygon()  +
  scale_color_gradientn(colors = nipy_spectral[c(3:6, 8, 9, 9, 9)], limits = c(0, 2.5)) +
  theme_bw() +
  labs(x=NULL, y=NULL,
       title="Predicted Fishing Effort 2000-2014", color=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(.50, 0, 0, 0), "cm")) +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
  guides(color = guide_colorbar(title.position = "top",
                      frame.colour = "black",
                      barwidth = .35,
                      barheight = 6,
                      label.position = 'right')) +
  NULL



pred2 <- ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = filter(fdat16, !is.na(pred_2075_2090)), aes(lon, lat, color=(pred_2075_2090)), size = 0.488, alpha = 0.35, inherit.aes = FALSE, shape=15) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  geom_polygon()  +
  scale_color_gradientn(colors = nipy_spectral[c(3:6, 8, 9, 9, 9)], limits = c(0, 2.5)) +
  theme_bw() +
  labs(x=NULL, y=NULL,
       title="Predicted Fishing Effort 2075-2090", color=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(.50, 0, 0, 0), "cm")) +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
  guides(color = guide_colorbar(title.position = "top",
                      frame.colour = "black",
                      barwidth = .35,
                      barheight = 6,
                      label.position = 'right')) +
  NULL

plot_grid(pred1, pred2, ncol=1)

ggsave("figures/16-Map-Predicted-Fishing-Hours-Reg.png")

# --------------------------------------------------------------------------------------------












# --------------------------------------------------------------------------------------------
# 17. Shannon Diversity error (prediction now - obs)
fdat17 <- as.data.frame(read_csv("~/Projects/World-fishing-rich-diver-equit/data/fishing-effort-regression_cross_validation_results.csv"))

fdat17$diff <- fdat17$y_pred - fdat17$y_true

nrow(filter(fdat17, diff == 0))/nrow(fdat17)

mean(fdat17$r2_test)
min(fdat17$diff)
max(fdat17$diff)

ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = fdat17, aes(lon, lat, color=(diff)), size = 0.64, inherit.aes = FALSE, shape=15) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  geom_polygon()  +
  scale_color_gradientn(colors =  rev(c("#1171B0","white","#CA0020")), values = rescale(c(min(fdat17$diff), 0, max(fdat17$diff))), limits = c(min(fdat17$diff), max(fdat17$diff))) +
  theme_bw() +
  labs(x=NULL, y=NULL,
       title="Fishing Effort Regression Error Map", color=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(0, 0, 0, 0), "cm")) +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
  guides(color = guide_colorbar(title.position = "top",
                      frame.colour = "black",
                      barwidth = .35,
                      barheight = 17,
                      label.position = 'right')) +
  NULL


ggsave("figures/17-Map-Fishing-Effort-Regression-errors.png")

# --------------------------------------------------------------------------------------------







# --------------------------------------------------------------------------------------------
# 18. Shannon Diversity difference maps
fdat18 <- as.data.frame(read_csv("~/Projects/World-fishing-rich-diver-equit/data/rf_fishing_effort_regression_model_results.csv"))
fdat18b <- as.data.frame(read_csv("~/Projects/World-fishing-rich-diver-equit/data/fishing-effort-regression_cross_validation_results.csv"))

fdat18b <- dplyr::select(fdat18b, lat, lon, y_true)

fdat18 <- left_join(fdat18, fdat18b, by = c("lat", "lon"))

fdat18$diff_2045 <- fdat18$pred_2030_2045 - fdat18$pred_2000_2014
fdat18$diff_2090 <- fdat18$pred_2075_2090 - fdat18$pred_2000_2014
fdat18$diff_2075 <- fdat18$pred_2075_2090 - fdat18$pred_2030_2045

fdat18 <- dplyr::select(fdat18, lat, lon, diff_2045, diff_2090, diff_2075) 

names(fdat18) <- c("lat", "lon", "2030-2045 Difference from 2000-2014", "2075-2090 Difference from 2000-2014", "2075-2090 Difference from 2030-2045")

fdat18 <- gather(fdat18, key = diff, value = value, -lat, -lon)

unique(fdat18$diff)
unique(fdat18$value)
min(fdat18$value, na.rm = TRUE)
max(fdat18$value, na.rm = TRUE)

fdat18$diff <- factor(fdat18$diff, levels = c("2030-2045 Difference from 2000-2014", "2075-2090 Difference from 2000-2014", "2075-2090 Difference from 2030-2045"))

fdat18$value <- ifelse(fdat18$value > 0.75, 0.70, fdat18$value)
fdat18$value <- ifelse(fdat18$value < -0.75, 0.70, fdat18$value)

ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = filter(fdat18, !is.na(value) | (value >= -0.5 & value <= 0.5)), aes(lon, lat, color=(value)), size = 0.44, inherit.aes = FALSE, shape=15) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  geom_polygon()  +
  scale_color_gradientn(colors =  (c("#1171B0","white","#CA0020")), values = rescale(c(-0.75, 0, 0.75)), 
                        limits = c(-0.75, 0.75), breaks = c(-0.75, -0.50, -0.25, 0, 0.25, 0.50, 0.75)) +
  theme_bw() +
  labs(x=NULL, y=NULL,
       title="Fishing Effort Regression Predicted Differences", color=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(0, 0, 0, 0), "cm"),
        legend.position = c(0.56, 0.22)) +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
    guides(color = guide_colorbar(title.position = "top",
                      frame.colour = "black",
                      barwidth = .35,
                      barheight = 8,
                      label.position = 'right')) +
  facet_wrap(~diff, ncol=2) +
  NULL



ggsave("figures/18-Map-Fishing-Effort-Regression-Differences.png")

# --------------------------------------------------------------------------------------------











# --------------------------------------------------------------------------------------------
# 19. Latitudinal Gradient change in flag-diversity

fdat19 <- as.data.frame(read_csv("~/Projects/World-fishing-rich-diver-equit/data/rf_fishing_effort_regression_model_results.csv"))
fdat19b <- as.data.frame(read_csv("~/Projects/World-fishing-rich-diver-equit/data/fishing-effort-regression_cross_validation_results.csv"))

fdat19b <- dplyr::select(fdat19b, lat, lon, y_true)

fdat19 <- left_join(fdat19, fdat19b, by = c("lat", "lon"))

fdat19$diff_2045 <- fdat19$pred_2030_2045 - fdat19$y_true
fdat19$diff_2090 <- fdat19$pred_2075_2090 - fdat19$y_true
fdat19$diff_2075 <- fdat19$pred_2075_2090 - fdat19$pred_2030_2045

fdat19 <- dplyr::select(fdat19, lat, lon, diff_2045, diff_2090, diff_2075) 

names(fdat19) <- c("lat", "lon", "2030-2045 Difference from Obs", "2075-2090 Difference from Obs.", "2075-2090 Difference from 2030-2045")

fdat19 <- gather(fdat19, key = diff, value = value, -lat, -lon)

unique(fdat19$diff)
unique(fdat19$value)

fdat19$diff <- factor(fdat19$diff, levels = c("2030-2045 Difference from Obs", "2075-2090 Difference from Obs.", "2075-2090 Difference from 2030-2045"))

fdat19 <- fdat19 %>% group_by(lat, diff) %>% summarise(grad = mean(value),
                                                       grad_sd = sd(value),
                                                       se = 1.96*(grad_sd^2/sqrt(n())),
                                                       grad_min = grad - se,
                                                       grad_max = grad + se)

fdat19$div <- ifelse(fdat19$grad > 0, 1, 0)

fdat19 <- filter(fdat19, diff != "2075-2090 Difference from 2030-2045")

ggplot(fdat19, aes(lat, grad, color=factor(div))) + 
  geom_point(size = 0.8) +
  geom_errorbar(aes(ymin=grad_min, ymax=grad_max), width = 1) +
  theme_bw() + 
  scale_color_manual(values = c("#67A9CF", "#EF8A61")) +
  labs(x="Latitude", y="Average Longitudinal Fishing Effort") +
  scale_x_continuous(breaks = seq(-90, 90, 10), limits = c(-80, 80)) +
  # scale_y_continuous(breaks = seq(-0.1, 0.1, 0.1), limits = c(-0.1, 0.1)) +
  theme(legend.position = "none") +
  facet_wrap(~diff, ncol=1, scales="free") +
  NULL


ggsave("figures/19-Fishing-Effort-Latitudinal-Gradient.png")












# --------------------------------------------------------------------------------------------
# 19. Shannon Div Classification and Fishing Effort Regression Normalization map

# Shannon Div Classification

fdat20a <- as.data.frame(read_csv("~/Projects/World-fishing-rich-diver-equit/data/rf_shannon_model_results.csv"))
fdat20ab <- as.data.frame(read_csv("~/Projects/World-fishing-rich-diver-equit/data/shannon_cross_validation_results.csv"))
fdat20ab <- dplyr::select(fdat20ab, lat, lon, y_true)

fdat20a <- left_join(fdat20a, fdat20ab, by = c("lat", "lon"))

fdat20a$diff_2045 <- fdat20a$pred_2030_2045 - fdat20a$pred_2000_2014
fdat20a$diff_2090 <- fdat20a$pred_2075_2090 - fdat20a$pred_2000_2014
fdat20a$diff_2075 <- fdat20a$pred_2075_2090 - fdat20a$pred_2000_2014

fdat20a <- dplyr::select(fdat20a, lat, lon, diff_2045, diff_2090, diff_2075) 

names(fdat20a) <- c("lat", "lon", "2030-2045 Difference from 2000-2014", "2075-2090 Difference from 2000-2014", "2075-2090 Difference from 2030-2045")

fdat20a <- gather(fdat20a, key = diff, value = value, -lat, -lon)

fdat20a$diff <- factor(fdat20a$diff, levels = c("2030-2045 Difference from 2000-2014", "2075-2090 Difference from 2000-2014", "2075-2090 Difference from 2030-2045"))
fdat20a <- filter(fdat20a, diff %in% c( "2075-2090 Difference from 2000-2014"))

ggplot(fdat20a, aes(lon, lat, color=value)) + geom_point()

fdat20a$norm_diff_shannon <- 0
fdat20a$norm_diff_shannon <- ifelse(fdat20a$value > 0, -1, fdat20a$norm_diff_shannon)
fdat20a$norm_diff_shannon <- ifelse(fdat20a$value < 0, 1, fdat20a$norm_diff_shannon)
fdat20a$lat_lon <- paste0(fdat20a$lat, "_", fdat20a$lon)



# Fishing Effort Regression Model

fdat20b <- as.data.frame(read_csv("~/Projects/World-fishing-rich-diver-equit/data/rf_fishing_effort_regression_model_results.csv"))
fdat20bb <- as.data.frame(read_csv("~/Projects/World-fishing-rich-diver-equit/data/fishing-effort-regression_cross_validation_results.csv"))
fdat20bb <- dplyr::select(fdat20bb, lat, lon, y_true)

fdat20b <- left_join(fdat20b, fdat20bb, by = c("lat", "lon"))

fdat20b$diff_2045 <- fdat20b$pred_2030_2045 - fdat20b$pred_2000_2014
fdat20b$diff_2090 <- fdat20b$pred_2075_2090 - fdat20b$pred_2000_2014
fdat20b$diff_2075 <- fdat20b$pred_2075_2090 - fdat20b$pred_2000_2014

fdat20b <- dplyr::select(fdat20b, lat, lon, diff_2045, diff_2090, diff_2075) 

names(fdat20b) <- c("lat", "lon", "2030-2045 Difference from 2000-2014", "2075-2090 Difference from 2000-2014", "2075-2090 Difference from 2030-2045")

fdat20b <- gather(fdat20b, key = diff, value = value, -lat, -lon)

fdat20b <- filter(fdat20b, diff %in% c( "2075-2090 Difference from 2000-2014"))

fdat20b$norm_diff_feffort <- 0
fdat20b$norm_diff_feffort <- ifelse(fdat20b$value < 0, -1, fdat20b$norm_diff_feffort)
fdat20b$norm_diff_feffort <- ifelse(fdat20b$value > 0, 1, fdat20b$norm_diff_feffort)
fdat20b$lat_lon <- paste0(fdat20b$lat, "_", fdat20b$lon)


# Bind data

fdat20a <- dplyr::select(fdat20a, lat_lon, diff, norm_diff_shannon)
fdat20b <- dplyr::select(fdat20b, lat_lon, diff, lat, lon, norm_diff_feffort)

fdat20c <- left_join(fdat20a, fdat20b, by = c("lat_lon", "diff"))
fdat20c$norm_diff <- fdat20c$norm_diff_feffort + fdat20c$norm_diff_shannon

# fdat20c <- fdat20c %>% replace(is.na(.), 0)

unique(fdat20c$norm_diff)


ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = filter(fdat20c, !is.na(norm_diff)), aes(lon, lat, color=factor(norm_diff)), size = .5, alpha= 0.5, inherit.aes = FALSE, shape = 15) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  geom_polygon()  +
  # scale_color_brewer(palette = "RdBu", "div", labels = c("+3", "+2", "+1", "0", "-1", "-2", "-3"), direction = 1) +
  scale_color_brewer(palette = "RdBu", direction = -1, labels = c("-2", "-1", "0", "+1", "+2")) +
  theme_bw() +
  labs(x=NULL, y=NULL,
       title="Shannon Diveristy and Fishing Effort Normalization", color=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.margin = unit(c(0, 0, 0, 0), "cm")) +
  coord_sf(xlim = c(0, 360), ylim = c(-80, 80), expand = FALSE) +
  guides(color = guide_legend(override.aes = list(size = 3, shape = 15), reverse = TRUE)) +
  facet_wrap(~diff) +
  NULL

ggsave("figures/20-Map-Normalization-Shannon-Fishing-Effort.png")

# --------------------------------------------------------------------------------------------









# --------------------------------------------------------------------------------------------
# Seascape figure

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
sdat <- read_feather('~/Projects/predicting-illegal-fishing/data/full_gfw_10d_effort_model_data_8DAY_2012-01-01_2016-12-26.feather')
sdat1 <- read_csv('data/seascape_fishing_hours.csv')
sdat2 <- read_csv('data/seascape_richness.csv')
sdat3 <- read_csv('data/seascape_shannon_div.csv')

s_top <- c(14, 7, 21, 12, 1)
sdat1 <- left_join(sdat1, seascape_labels, by = "seascape_class")

sdat1 <- sdat1 %>% group_by(seascape_class) %>% summarise(fishing_hours = sum(fishing_hours))
sdat1 <- left_join(sdat1, seascape_labels, by = "seascape_class")

sdat1 %>% arrange(-fishing_hours)
sdat1 <- filter(sdat1, seascape_class %in% stop)

sdat1$seascape_class <- factor(sdat1$seascape_class, levels = c(14, 7, 21, 12, 1))

ggplot(sdat1, aes(factor(seascape_class), fishing_hours)) + 
  geom_bar(stat='identity', width = .5) + 
  geom_text(aes(label = seascape_name), size = 3.75, vjust= -0.5 ) +
  geom_text(aes(label = comma(round(fishing_hours, 0))), size = 3.5, vjust= 1.2, color="white") +
  scale_y_continuous(labels = comma, limits = c(0, 25000)) +
  # ylim(0, 25000) +
  labs(x="Seascape Class", y="Total Fishing Hours") +
  theme_bw() +
  NULL

ggsave("figures/21-Seascape-Class-Fishing-Effort.png")



sdat3 <- filter(sdat3, seascape_class %in% s_top & H > 0) %>% group_by(seascape_class) 


# sdat3 <- sdat3 %>% group_by(seascape_class) %>% summarise(mean_H = mean(H, na.rm = TRUE),
#                                                sd_H = sd(H, na.rm = TRUE),
#                                                se = sd_H^2 / sqrt(n()),
#                                                min_h = mean_H - se*1.96,
#                                                max_h = mean_H + se*1.96)

sdat3 %>% arrange(-H)
sdat3 <- left_join(sdat3, seascape_labels, by = "seascape_class")
sdat3$seascape_name <- factor(sdat3$seascape_name, levels = c("TEMPERATE BLOOMS \n UPWELLING", "WARM, BLOOMS \n HIGH NUTS", 
                                                                "TEMPERATE \n TRANSITION", "SUBPOLAR", "NORTH ATLANTIC SPRING \n ACC TRANSITION"))

ggplot(sdat3, aes(factor(seascape_name), H)) + 
  # geom_bar(stat='identity', width = .5) +
  # geom_errorbar(aes(min=min_h, max=max_h)) +
  geom_tufteboxplot(median.type = "line", hoffset = 0, width = 2, size = 1.5) +
  # geom_text(aes(label = seascape_name), size = 3.75, vjust= -0.5 ) +
  # geom_text(aes(label = comma(round(fishing_hours, 0))), size = 3.5, vjust= 1.2, color="white") +
  # scale_y_continuous(labels = comma, limits = c(0, 25000)) +
  # ylim(0, 25000) +
  labs(x=NULL, y="Shannon Diversity") +
  theme_bw() +
  NULL

ggsave("figures/21-Seascape-Class-Shannon-Diversity.png")


