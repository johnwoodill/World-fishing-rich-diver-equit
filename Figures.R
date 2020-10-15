library(tidyverse)
library(marmap)

setwd("~/Projects/World-fishing-rich-diver-equit/")

mdat <- read_csv("data/daily_species_richness.csv")

mdat$month <- month(mdat$date)
mdat$day <- day(mdat$date)

pdat <- filter(mdat, month == 2 & day == 1)

ggplot(mdat, aes(lon, lat, fill=flag)) + 
  stat_density2d()


