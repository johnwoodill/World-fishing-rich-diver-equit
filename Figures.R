library(tidyverse)
library(marmap)
library(ggmap)
library(viridis)
library(mapdata)

setwd("~/Projects/World-fishing-rich-diver-equit/")

fdat <- as.data.frame(read_csv("data/total_fishing_effort.csv"))

fdat$lon <- ifelse(fdat$lon < 0, fdat$lon + 360, fdat$lon)

# Subset within WCP
fdat1 <- filter(fdat, lon <= -150 + 360 & lon >= 100 & lat >= 0)
fdat2 <- filter(fdat, lon <= -130 + 360 & lon >= 140 & lat <= 0 & lat >= -55)
fdat3 <- filter(fdat, lon <= -130 + 360 & lon >= 150 & lat <= -55 & lat >= -60)

# Bind and filter out fishing_hours > 0
pdat <- rbind(fdat1, fdat2, fdat3)
pdat <- filter(pdat, fishing_hours > 0)

# Setup map
# mp1 <- fortify(map(fill=TRUE, plot=FALSE))
# mp2 <- mp1
# mp2$long <- mp2$long + 360
# mp2$group <- mp2$group + max(mp2$group) + 1
# mp <- rbind(mp1, mp2)

ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_point(data = pdat, aes(lon, lat, color=fishing_hours), size = 0.5, inherit.aes = FALSE) +
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
        legend.position = c(.97, 0.27)) +
  coord_cartesian(xlim = c(100, 250), ylim = c(-60, 60)) +
  guides(fill = FALSE,
       color = guide_colorbar(title.position = "top",
                              frame.colour = "black",
                              barwidth = .5,
                              barheight = 10,
                              label.position = 'left')) +
  NULL

ggsave("figures/total_fishing_effort.png", width=10, height=6)
#








### Fishing Effort by Flag

# Chinese
setwd("~/Projects/World-fishing-rich-diver-equit/")

fdat <- as.data.frame(read_csv("data/total_fishing_effort_nation.csv"))

fdat$lon <- ifelse(fdat$lon < 0, fdat$lon + 360, fdat$lon)

# Subset within WCP
fdat1 <- filter(fdat, lon <= -150 + 360 & lon >= 100 & lat >= 0)
fdat2 <- filter(fdat, lon <= -130 + 360 & lon >= 140 & lat <= 0 & lat >= -55)
fdat3 <- filter(fdat, lon <= -130 + 360 & lon >= 150 & lat <= -55 & lat >= -60)

# Bind and filter out fishing_hours > 0
pdat <- rbind(fdat1, fdat2, fdat3)
pdat <- filter(pdat, fishing_hours > 0)

pdat <- filter(pdat, flag == "CHN")

ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_point(data = pdat, aes(lon, lat, color=fishing_hours), size = 0.5, inherit.aes = FALSE) +
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
        legend.position = c(.97, 0.27)) +
  coord_cartesian(xlim = c(100, 250), ylim = c(-60, 60)) +
  guides(fill = FALSE,
       color = guide_colorbar(title.position = "top",
                              frame.colour = "black",
                              barwidth = .5,
                              barheight = 10,
                              label.position = 'left')) +
  NULL

ggsave("figures/total_fishing_effort_CHN.png", width=10, height=6)




### Fishing Effort by Flag

# USA
setwd("~/Projects/World-fishing-rich-diver-equit/")

fdat <- as.data.frame(read_csv("data/total_fishing_effort_nation.csv"))

fdat$lon <- ifelse(fdat$lon < 0, fdat$lon + 360, fdat$lon)

# Subset within WCP
fdat1 <- filter(fdat, lon <= -150 + 360 & lon >= 100 & lat >= 0)
fdat2 <- filter(fdat, lon <= -130 + 360 & lon >= 140 & lat <= 0 & lat >= -55)
fdat3 <- filter(fdat, lon <= -130 + 360 & lon >= 150 & lat <= -55 & lat >= -60)

# Bind and filter out fishing_hours > 0
pdat <- rbind(fdat1, fdat2, fdat3)
pdat <- filter(pdat, fishing_hours > 0)

pdat <- filter(pdat, flag == "USA")

ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_point(data = pdat, aes(lon, lat, color=fishing_hours), size = 0.5, inherit.aes = FALSE) +
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
        legend.position = c(.97, 0.27)) +
  coord_cartesian(xlim = c(100, 250), ylim = c(-60, 60)) +
  guides(fill = FALSE,
       color = guide_colorbar(title.position = "top",
                              frame.colour = "black",
                              barwidth = .5,
                              barheight = 10,
                              label.position = 'left')) +
  NULL

ggsave("figures/total_fishing_effort_USA.png", width=10, height=6)




### Species Richness


setwd("~/Projects/World-fishing-rich-diver-equit/")

fdat <- as.data.frame(read_csv("data/total_species_richness.csv"))

fdat$lon <- ifelse(fdat$lon < 0, fdat$lon + 360, fdat$lon)

# Subset within WCP
fdat1 <- filter(fdat, lon <= -150 + 360 & lon >= 100 & lat >= 0)
fdat2 <- filter(fdat, lon <= -130 + 360 & lon >= 140 & lat <= 0 & lat >= -55)
fdat3 <- filter(fdat, lon <= -130 + 360 & lon >= 150 & lat <= -55 & lat >= -60)

# Bind and filter out fishing_hours > 0
pdat <- rbind(fdat1, fdat2, fdat3)

ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_point(data = pdat, aes(lon, lat, color=flag), size = 0.5, inherit.aes = FALSE) +
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
        legend.position = c(.97, 0.27)) +
  coord_cartesian(xlim = c(100, 250), ylim = c(-60, 60)) +
  guides(fill = FALSE,
       color = guide_colorbar(title.position = "top",
                              frame.colour = "black",
                              barwidth = .5,
                              barheight = 10,
                              label.position = 'left')) +
  NULL

ggsave("figures/species_richness.png", width=10, height=6)






### Shannon Diversity


setwd("~/Projects/World-fishing-rich-diver-equit/")

fdat <- as.data.frame(read_csv("data/shannon_div_equ.csv"))

fdat$lon <- ifelse(fdat$lon < 0, fdat$lon + 360, fdat$lon)

# Subset within WCP
fdat1 <- filter(fdat, lon <= -150 + 360 & lon >= 100 & lat >= 0)
fdat2 <- filter(fdat, lon <= -130 + 360 & lon >= 140 & lat <= 0 & lat >= -55)
fdat3 <- filter(fdat, lon <= -130 + 360 & lon >= 150 & lat <= -55 & lat >= -60)

# Bind and filter out fishing_hours > 0
pdat <- rbind(fdat1, fdat2, fdat3)

ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_point(data = pdat, aes(lon, lat, color=H), size = 0.5, inherit.aes = FALSE) +
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
        legend.position = c(.97, 0.27)) +
  coord_cartesian(xlim = c(100, 250), ylim = c(-60, 60)) +
  guides(fill = FALSE,
       color = guide_colorbar(title.position = "top",
                              frame.colour = "black",
                              barwidth = .5,
                              barheight = 10,
                              label.position = 'left')) +
  NULL

ggsave("figures/shannon_div_index.png", width=10, height=6)





### Shannon Equity


setwd("~/Projects/World-fishing-rich-diver-equit/")

fdat <- as.data.frame(read_csv("data/shannon_div_equ.csv"))

fdat$lon <- ifelse(fdat$lon < 0, fdat$lon + 360, fdat$lon)

# Subset within WCP
fdat1 <- filter(fdat, lon <= -150 + 360 & lon >= 100 & lat >= 0)
fdat2 <- filter(fdat, lon <= -130 + 360 & lon >= 140 & lat <= 0 & lat >= -55)
fdat3 <- filter(fdat, lon <= -130 + 360 & lon >= 150 & lat <= -55 & lat >= -60)

# Bind and filter out fishing_hours > 0
pdat <- rbind(fdat1, fdat2, fdat3)

ggplot(data = mp, aes(x = long, y = lat, group = group)) + 
  geom_point(data = pdat, aes(lon, lat, color=E), size = 0.5, inherit.aes = FALSE) +
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
        legend.position = c(.97, 0.27)) +
  coord_cartesian(xlim = c(100, 250), ylim = c(-60, 60)) +
  guides(fill = FALSE,
       color = guide_colorbar(title.position = "top",
                              frame.colour = "black",
                              barwidth = .5,
                              barheight = 10,
                              label.position = 'left')) +
  NULL

ggsave("figures/shannon_equ_index.png", width=10, height=6)
