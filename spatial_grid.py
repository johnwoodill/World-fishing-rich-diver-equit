import geopandas as gpd
from shapely.geometry import Polygon, Point
import numpy as np



def build_cv_grids(delta):
    # Getting range of lat/lon 
    # (set start of lon to delta to deal with border issues)
    xmin, ymin, xmax, ymax = (delta, -90, 361, 91)

    # Build out range of lat/lon
    lats = list(range(int(np.floor(ymin)), int(np.ceil(ymax)), delta))
    lons = list(range(int(np.floor(xmin)), int(np.ceil(xmax)), delta))
    lats.reverse()
        
    # Get count grids to alternative across lon
    even_grids = [0, 1] * int(len(lats))
    odd_grids = [1, 0] * int(len(lats))

    # Iterate through each and build polygon grids
    polygons = []
    grids = []
    for i in range(0, len(lons)):
        x = lons[i]
        if i % 2 == 1:
            grids.extend(even_grids[0:len(lats)])
        else:
            grids.extend(odd_grids[0:len(lats)])
        for y in lats:        
            polygons.append( Polygon([(x, y), (x - delta, y), (x - delta, y + delta), (x, y + delta)]) )
            
    grid = gpd.GeoDataFrame({'geometry':polygons, 'grid': np.linspace(0, len(polygons), len(polygons)).astype(int), 'cv_grid': grids})
    return grid





def check_grid(lat, lon, grid_data):
    # If on the edge, don't center
    if lon == 0:
        lon = lon + 0.01
        lat = lat - 0.01
    else: 
        lon = lon - 0.01
        lat = lat - 0.01
    
    # Convert lon/lat to Point()
    p1 = Point(lon, lat)

    # Loop through each grid and check if in polygon
    for i in range(0, len(grid_data)):
        # Get polygon
        poly = grid_data.loc[i, 'geometry']
        
        # Check if in poly
        check = p1.within(poly)
        
        ### Get Variables if true
        if check == True:
            # print("true")
            n_grid = grid_data.loc[i, 'grid']
            cv_n_grid = grid_data.loc[i, 'cv_grid']
            return (n_grid, cv_n_grid)
    return (999, 999)


ndat = full_dat = pd.read_csv("data/full_gfw_cmip_dat.csv")
# ndat = ndat[(ndat['lon'] >= 350) & (ndat['lat'] >= 60)].iloc[:, 1:3]
# ndat = ndat[(ndat['lon'] <= 10) & (ndat['lat'] >= 60)].iloc[:, 1:3]
# 
ndat

grid_data = build_cv_grids(.1)
grid_data

grid_results = ndat.apply(lambda x: check_grid(x['lat'], x['lon'], grid_data), axis=1)
ndat = ndat.assign(grid = [x[0] for x in grid_results], cv_grid = [x[1] for x in grid_results])

ndat.to_csv("data/test.csv")

