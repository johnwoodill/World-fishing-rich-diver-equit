library(tidyverse)
library(marmap)
library(ggmap)
library(viridis)
library(mapdata)
library(sf)
library(raster)


setwd("~/Projects/World-fishing-rich-diver-equit/")

eezs <- read_sf('data/World_EEZ_v11_20191118_HR_0_360/eez_boundaries_v11_0_360.shp')

mpas <- read_sf('data/WDPA_WDOECM_marine_shp-points.shp')
mpas <- st_shift_longitude(mpas)

mpas <- read_sf('data/mpa_shapefiles/vlmpa.shp')
mpas <- st_shift_longitude(mpas)

fdat <- as.data.frame(read_csv("data/total_fishing_effort.csv"))

fdat$lon <- ifelse(fdat$lon < 0, fdat$lon + 360, fdat$lon)

# Subset within WCP
fdat1 <- filter(fdat, lon <= -150 + 360 & lon >= 100 & lat >= 0)
fdat2 <- filter(fdat, lon <= -130 + 360 & lon >= 140 & lat < 0 & lat >= -55)
fdat3 <- filter(fdat, lon <= -130 + 360 & lon >= 150 & lat <= -55 & lat >= -60)

# Bind and filter out fishing_hours > 0
pdat <- rbind(fdat1, fdat2, fdat3)
pdat <- filter(pdat, fishing_hours > 0)

# Setup map
mp1 <- fortify(map(fill=TRUE, plot=FALSE))
mp2 <- mp1
mp2$long <- mp2$long + 360
mp2$group <- mp2$group + max(mp2$group) + 1
mp <- rbind(mp1, mp2)

ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = pdat, aes(lon, lat, color=fishing_hours), size = 0.25, inherit.aes = FALSE, shape=15) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  
    # First segment to the east
  annotate("segment", x = -150 + 360, xend = -150 + 360, y = 0, yend = 60) +
  annotate("segment", x = -150 + 360, xend = -130 + 360, y = 0, yend = 0) +
  annotate("segment", x = -130 + 360, xend = -130 + 360, y = 0, yend = -60) +
  annotate("segment", x = -130 + 360, xend = -210 + 360, y = -60, yend = -60) +
  annotate("segment", x = -210 + 360, xend = -210 + 360, y = -60, yend = -55) +
  annotate("segment", x = -210 + 360, xend = -220 + 360, y = -55, yend = -55) +
  annotate("segment", x = -220 + 360, xend = -220 + 360, y = -55, yend = -38) +
  geom_polygon()  +
  scale_color_viridis(direction = -1) +
  theme_bw() +
  labs(x="Longitude", y="Latitude", 
       title="Average Annual Fishing Effort in the Western-Central Pacific Ocean \n 2012-2016", color=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = c(.94, 0.27)) +
  coord_sf(xlim = c(100, 250), ylim = c(-60, 60)) +
  guides(fill = FALSE,
       color = guide_colorbar(title.position = "top",
                              frame.colour = "black",
                              barwidth = .25,
                              barheight = 6,
                              label.position = 'left')) +
  NULL

ggsave("figures/total_fishing_effort.png", width=8, height=4)
#








### Fishing Effort by Flag

# Chinese
setwd("~/Projects/World-fishing-rich-diver-equit/")

fdat <- as.data.frame(read_csv("data/total_fishing_effort_nation.csv"))

fdat$lon <- ifelse(fdat$lon < 0, fdat$lon + 360, fdat$lon)

# Subset within WCP
fdat1 <- filter(fdat, lon <= -150 + 360 & lon >= 100 & lat >= 0)
fdat2 <- filter(fdat, lon <= -130 + 360 & lon >= 140 & lat < 0 & lat >= -55)
fdat3 <- filter(fdat, lon <= -130 + 360 & lon >= 150 & lat <= -55 & lat >= -60)

# Bind and filter out fishing_hours > 0
pdat <- rbind(fdat1, fdat2, fdat3)
pdat <- filter(pdat, fishing_hours > 0)

pdat <- filter(pdat, flag == "CHN")

ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = pdat, aes(lon, lat, color=fishing_hours), size = 0.25, inherit.aes = FALSE, shape=15) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  # First segment to the east
  annotate("segment", x = -150 + 360, xend = -150 + 360, y = 0, yend = 60) +
  annotate("segment", x = -150 + 360, xend = -130 + 360, y = 0, yend = 0) +
  annotate("segment", x = -130 + 360, xend = -130 + 360, y = 0, yend = -60) +
  annotate("segment", x = -130 + 360, xend = -210 + 360, y = -60, yend = -60) +
  annotate("segment", x = -210 + 360, xend = -210 + 360, y = -60, yend = -55) +
  annotate("segment", x = -210 + 360, xend = -220 + 360, y = -55, yend = -55) +
  annotate("segment", x = -220 + 360, xend = -220 + 360, y = -55, yend = -38) +
  geom_polygon()  + 
  scale_color_viridis(direction = -1, limits = c(0.01, 12)) +
  theme_bw() +
  labs(x="Longitude", y="Latitude", 
       title="Chinese Average Annual Fishing Effort in the Western-Central Pacific Ocean \n 2012-2016", color=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = c(.94, 0.27)) +
  # coord_cartesian(xlim = c(100, 250), ylim = c(-60, 60)) +
  coord_sf(xlim = c(100, 250), ylim = c(-60, 60)) +
  guides(fill = FALSE,
       color = guide_colorbar(title.position = "top",
                              frame.colour = "black",
                              barwidth = .25,
                              barheight = 6,
                              label.position = 'left')) +
  NULL

ggsave("figures/total_fishing_effort_CHN.png", width=8, height=4)




### Fishing Effort by Flag

# USA
setwd("~/Projects/World-fishing-rich-diver-equit/")

fdat <- as.data.frame(read_csv("data/total_fishing_effort_nation.csv"))

fdat$lon <- ifelse(fdat$lon < 0, fdat$lon + 360, fdat$lon)

# Subset within WCP
fdat1 <- filter(fdat, lon <= -150 + 360 & lon >= 100 & lat >= 0)
fdat2 <- filter(fdat, lon <= -130 + 360 & lon >= 140 & lat < 0 & lat >= -55)
fdat3 <- filter(fdat, lon <= -130 + 360 & lon >= 150 & lat <= -55 & lat >= -60)

# Bind and filter out fishing_hours > 0
pdat <- rbind(fdat1, fdat2, fdat3)
pdat <- filter(pdat, fishing_hours > 0)

pdat <- filter(pdat, flag == "USA")

ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = pdat, aes(lon, lat, color=fishing_hours), size = 0.25, inherit.aes = FALSE, shape=15) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  # First segment to the east
  annotate("segment", x = -150 + 360, xend = -150 + 360, y = 0, yend = 60) +
  annotate("segment", x = -150 + 360, xend = -130 + 360, y = 0, yend = 0) +
  annotate("segment", x = -130 + 360, xend = -130 + 360, y = 0, yend = -60) +
  annotate("segment", x = -130 + 360, xend = -210 + 360, y = -60, yend = -60) +
  annotate("segment", x = -210 + 360, xend = -210 + 360, y = -60, yend = -55) +
  annotate("segment", x = -210 + 360, xend = -220 + 360, y = -55, yend = -55) +
  annotate("segment", x = -220 + 360, xend = -220 + 360, y = -55, yend = -38) +
  geom_polygon()  + 
  scale_color_viridis(direction = -1, limits = c(0.01, 12)) +
  theme_bw() +
  labs(x="Longitude", y="Latitude", 
       title="USA Average Annual Fishing Effort in the Western-Central Pacific Ocean \n 2012-2016", color=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = c(.94, 0.27)) +
  # coord_cartesian(xlim = c(100, 250), ylim = c(-60, 60)) +
  coord_sf(xlim = c(100, 250), ylim = c(-60, 60)) +
  guides(fill = FALSE,
       color = guide_colorbar(title.position = "top",
                              frame.colour = "black",
                              barwidth = .25,
                              barheight = 6,
                              label.position = 'left')) +
  NULL

ggsave("figures/total_fishing_effort_USA.png", width=8, height=4)




### Fishing Effort top 5 flags

# Chinese
setwd("~/Projects/World-fishing-rich-diver-equit/")

fdat <- as.data.frame(read_csv("data/total_fishing_effort_nation.csv"))

fdat %>% group_by(flag) %>% summarise(sum_fishing_hours = sum(fishing_hours)) %>% arrange(-sum_fishing_hours) %>% head(6)



#   flag  sum_fishing_hours
#   <chr>             <dbl>
# 1 TWN              32963.
# 2 CHN              30935.
# 3 JPN              20514.
# 4 KOR              16572.
# 5 ESP              15380.
# 6 USA              10619.

top_5 <- c("TWN", "CHN", "JPN", "KOR", "ESP")

fdat$lon <- ifelse(fdat$lon < 0, fdat$lon + 360, fdat$lon)

# Subset within WCP
fdat1 <- filter(fdat, lon <= -150 + 360 & lon >= 100 & lat >= 0)
fdat2 <- filter(fdat, lon <= -130 + 360 & lon >= 140 & lat < 0 & lat >= -55)
fdat3 <- filter(fdat, lon <= -130 + 360 & lon >= 150 & lat <= -55 & lat >= -60)

# Bind and filter out fishing_hours > 0
pdat <- rbind(fdat1, fdat2, fdat3)
pdat <- filter(pdat, fishing_hours > 0)

i = 3

for (i in 1:length(top_5)){
  flag_ = top_5[i]
  mdat <- filter(pdat, flag == flag_ | flag == "USA") 
  mdat <- mdat %>% group_by(lon_lat) %>% mutate(nn = n())
  mdat$flag_overlap <- ifelse(mdat$nn == 2, paste0(flag_, "-USA"), mdat$flag)
  mdat$flag_overlap <- factor(mdat$flag_overlap, levels = c("USA", flag_, paste0(flag_, "-USA")))
  mdat <- as.data.frame(mdat)
  
  
p1 <- ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = mdat, aes(lon, lat, color=flag_overlap), size = 0.25, inherit.aes = FALSE, shape=15) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
    # First segment to the east
    annotate("segment", x = -150 + 360, xend = -150 + 360, y = 0, yend = 60) +
    annotate("segment", x = -150 + 360, xend = -130 + 360, y = 0, yend = 0) +
    annotate("segment", x = -130 + 360, xend = -130 + 360, y = 0, yend = -60) +
    annotate("segment", x = -130 + 360, xend = -210 + 360, y = -60, yend = -60) +
    annotate("segment", x = -210 + 360, xend = -210 + 360, y = -60, yend = -55) +
    annotate("segment", x = -210 + 360, xend = -220 + 360, y = -55, yend = -55) +
    annotate("segment", x = -220 + 360, xend = -220 + 360, y = -55, yend = -38) +
    geom_polygon()  + 
    scale_color_viridis_d() +
    theme_bw() +
    labs(x="Longitude", y="Latitude", 
         title=paste0("USA-", flag_, " Interactions"), color=NULL) +
    theme(plot.title = element_text(hjust = 0.5),
          legend.position = c(.917, 0.092)) +
    coord_sf(xlim = c(100, 250), ylim = c(-60, 60)) +
    guides(color = guide_legend(override.aes = list(size = 3))) +
    NULL
  
  ggsave(plot=p1, paste0('figures/', "USA-", flag_, "_interactions.png"), width = 8, height = 6)
  saveRDS(p1, file = paste0("data/", flag_, "-USA.rds"))
  print(flag_)

}


p1 <- readRDS("data/TWN-USA.rds")
p2 <- readRDS("data/CHN-USA.rds")
p3 <- readRDS("data/JPN-USA.rds")
p4 <- readRDS("data/KOR-USA.rds")
p5 <- readRDS("data/ESP-USA.rds")

plot_grid(p1, p2, p3, p4, p5, ncol=2)

# plot_grid(p1, p1, p1, p1, p1, ncol=2)

ggsave("figures/all_interactions.png", width = 16, height = 6*3)

### Species Richness


setwd("~/Projects/World-fishing-rich-diver-equit/")

fdat <- as.data.frame(read_csv("data/total_species_richness.csv"))

fdat$lon <- ifelse(fdat$lon < 0, fdat$lon + 360, fdat$lon)

# Subset within WCP
fdat1 <- filter(fdat, lon <= -150 + 360 & lon >= 100 & lat >= 0)
fdat2 <- filter(fdat, lon <= -130 + 360 & lon >= 140 & lat < 0 & lat >= -55)
fdat3 <- filter(fdat, lon <= -130 + 360 & lon >= 150 & lat <= -55 & lat >= -60)

# Bind and filter out fishing_hours > 0
pdat <- rbind(fdat1, fdat2, fdat3)

ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = pdat, aes(lon, lat, color=fishing_hours), size = 0.25, inherit.aes = FALSE, shape=15) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  # First segment to the east
  annotate("segment", x = -150 + 360, xend = -150 + 360, y = 0, yend = 60) +
  annotate("segment", x = -150 + 360, xend = -130 + 360, y = 0, yend = 0) +
  annotate("segment", x = -130 + 360, xend = -130 + 360, y = 0, yend = -60) +
  annotate("segment", x = -130 + 360, xend = -210 + 360, y = -60, yend = -60) +
  annotate("segment", x = -210 + 360, xend = -210 + 360, y = -60, yend = -55) +
  annotate("segment", x = -210 + 360, xend = -220 + 360, y = -55, yend = -55) +
  annotate("segment", x = -220 + 360, xend = -220 + 360, y = -55, yend = -38) +
  geom_polygon()  + 
  scale_color_viridis_c(direction = -1, limits = c(1, 30)) +
  theme_bw() +
  labs(x="Longitude", y="Latitude", 
       title="Species Richness in the Western-Central Pacific Ocean \n 2012-2016", color=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = c(.94, 0.27)) +
  # coord_cartesian(xlim = c(100, 250), ylim = c(-60, 60)) +
  coord_sf(xlim = c(100, 250), ylim = c(-60, 60)) +
  guides(fill = FALSE,
       color = guide_colorbar(title.position = "top",
                              frame.colour = "black",
                              barwidth = .25,
                              barheight = 6,
                              label.position = 'left')) +
  NULL

ggsave("figures/species_richness.png", width=8, height=4)






### Shannon Diversity


setwd("~/Projects/World-fishing-rich-diver-equit/")

fdat <- as.data.frame(read_csv("data/shannon_div_equ.csv"))

fdat$lon <- ifelse(fdat$lon < 0, fdat$lon + 360, fdat$lon)

# Subset within WCP
fdat1 <- filter(fdat, lon <= -150 + 360 & lon >= 100 & lat >= 0)
fdat2 <- filter(fdat, lon <= -130 + 360 & lon >= 140 & lat < 0 & lat >= -55)
fdat3 <- filter(fdat, lon <= -130 + 360 & lon >= 150 & lat <= -55 & lat >= -60)

# Bind and filter out fishing_hours > 0
pdat <- rbind(fdat1, fdat2, fdat3)

ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = pdat, aes(lon, lat, color=fishing_hours), size = 0.25, inherit.aes = FALSE, shape=15) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  # First segment to the east
  annotate("segment", x = -150 + 360, xend = -150 + 360, y = 0, yend = 60) +
  annotate("segment", x = -150 + 360, xend = -130 + 360, y = 0, yend = 0) +
  annotate("segment", x = -130 + 360, xend = -130 + 360, y = 0, yend = -60) +
  annotate("segment", x = -130 + 360, xend = -210 + 360, y = -60, yend = -60) +
  annotate("segment", x = -210 + 360, xend = -210 + 360, y = -60, yend = -55) +
  annotate("segment", x = -210 + 360, xend = -220 + 360, y = -55, yend = -55) +
  annotate("segment", x = -220 + 360, xend = -220 + 360, y = -55, yend = -38) +
  geom_polygon()  + 
  scale_color_viridis_c(direction = -1, limits = c(0, 3)) +
  theme_bw() +
  labs(x="Longitude", y="Latitude", 
       title="Shannon Diversity Index in the Western-Central Pacific Ocean \n 2012-2016", color=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = c(.94, 0.27)) +
  coord_sf(xlim = c(100, 250), ylim = c(-60, 60)) +
  guides(fill = FALSE,
       color = guide_colorbar(title.position = "top",
                              frame.colour = "black",
                              barwidth = .25,
                              barheight = 6,
                              label.position = 'left')) +
  NULL

ggsave("figures/shannon_div_index.png", width=8, height=4)





### Shannon Equity


setwd("~/Projects/World-fishing-rich-diver-equit/")

fdat <- as.data.frame(read_csv("data/shannon_div_equ.csv"))

fdat$lon <- ifelse(fdat$lon < 0, fdat$lon + 360, fdat$lon)

# Subset within WCP
fdat1 <- filter(fdat, lon <= -150 + 360 & lon >= 100 & lat >= 0)
fdat2 <- filter(fdat, lon <= -130 + 360 & lon >= 140 & lat < 0 & lat >= -55)
fdat3 <- filter(fdat, lon <= -130 + 360 & lon >= 150 & lat <= -55 & lat >= -60)

# Bind and filter out fishing_hours > 0
pdat <- rbind(fdat1, fdat2, fdat3)

ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_sf(data = mpas, color = 'grey', alpha = 0.5, fill = 'grey', size = 0.1, inherit.aes = FALSE) +
  geom_point(data = pdat, aes(lon, lat, color=fishing_hours), size = 0.25, inherit.aes = FALSE, shape=15) +
  geom_sf(data = eezs, color = '#374a6d', alpha = 0.5, fill = NA, size = 0.1, inherit.aes = FALSE) +
  # First segment to the east
  annotate("segment", x = -150 + 360, xend = -150 + 360, y = 0, yend = 60) +
  annotate("segment", x = -150 + 360, xend = -130 + 360, y = 0, yend = 0) +
  annotate("segment", x = -130 + 360, xend = -130 + 360, y = 0, yend = -60) +
  annotate("segment", x = -130 + 360, xend = -210 + 360, y = -60, yend = -60) +
  annotate("segment", x = -210 + 360, xend = -210 + 360, y = -60, yend = -55) +
  annotate("segment", x = -210 + 360, xend = -220 + 360, y = -55, yend = -55) +
  annotate("segment", x = -220 + 360, xend = -220 + 360, y = -55, yend = -38) +
  geom_polygon()  + 
  scale_color_viridis_c(direction = -1, limits = c(0, 1)) +
  theme_bw() +
  labs(x="Longitude", y="Latitude", 
       title="Shannon Equitability Index in the Western-Central Pacific Ocean \n 2012-2016", color=NULL) +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = c(.94, 0.27)) +
  # coord_cartesian(xlim = c(100, 250), ylim = c(-60, 60)) +
  coord_sf(xlim = c(100, 250), ylim = c(-60, 60)) +
  guides(fill = FALSE,
       color = guide_colorbar(title.position = "top",
                              frame.colour = "black",
                              barwidth = .25,
                              barheight = 6,
                              label.position = 'left')) +
  NULL

ggsave("figures/shannon_equ_index.png", width=8, height=4)




### Heat map of interactions

idat <- as.data.frame(read_csv("data/flag_interactions.csv"))

idat <- filter(idat, interaction != 0 & interaction != 1) %>% arrange(-interaction)

ggplot(idat, aes(flag_1, flag_2, fill=interaction)) + geom_tile()
