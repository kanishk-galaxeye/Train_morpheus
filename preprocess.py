import json
from shapely.geometry import shape
from osgeo import gdal as gd
from rasterio.transform import rowcol
import rasterio
from pyproj import  Transformer
from rasterio.windows import Window, from_bounds
from rasterio.transform import Affine, rowcol
from PIL import Image
import os 
import numpy as np
from rasterio.mask import mask
from PIL import Image, ImageOps
from shapely.ops import transform
from shapely.geometry import box 
import glob
import ipdb
import pdb, traceback,sys


def polygon_to_hbb(geojson_geometry):
    """
    geojson_geometry = dict of geometry (type + coordinates)
    """
    geom = shape(geojson_geometry)

    # merge multi parts into one geometry
    merged = geom.union(geom)

    # shapely gives box as (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = merged.bounds

    return minx, miny, maxx, maxy

def save_hbb_patch(src, out_path, xmin, ymin, xmax, ymax):
    
        # convert float coords to ints

    xmin, xmax = sorted([xmin, xmax])
    ymin, ymax = sorted([ymin, ymax])

    xmin = int(np.floor(xmin))
    ymin = int(np.floor(ymin))
    xmax = int(np.ceil(xmax))
    ymax = int(np.ceil(ymax))
    # xmin, ymin = int(xmin), int(ymin)
    # xmax, ymax = int(xmax), int(ymax)

    # # width/height of patch
    w = xmax - xmin
    h = ymax - ymin

    # window reads the subset
    window = Window(xmin, ymin, w, h)

    patch = src.read(window=window)

    # update metadata
    meta = src.meta.copy()
    meta.update({
        "height": h,
        "width": w,
        "transform": rasterio.windows.transform(window, src.transform)
    })

    # save patch
  
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(patch)
  
def normalize(image):
    min =  np.percentile(image, 1)
    max = np.percentile(image, 99)
    image = np.clip(image, min, max)
    return (image - image.min())/(image.max() - image.min())

def extract_centroid(json_path):
    """
    Extracts subclass and centroids from the json - annotation file  
    """
    if json_path.split('.')[-1] == "xml":
        pass 
    else:

        with open(json_path) as j:
            
            data = json.load(j)
        json_crs = data['crs']['properties']['name']
        data = data['features']
        targets = []
        for i in data:
            geom = shape(i['geometry'])
            # cx, cy = geom.centroid.x , geom.centroid.y
            id = i['properties']['Subclass']
            targets.append((id, geom))
        return targets, json_crs
      
def target_centers(image_path, json_path, hbb = True, save_dir= ""):
    from shapely.geometry import Polygon, MultiPolygon

    '''
    Saves images in .npy and .png format by getting target location and class using extract_centroid
    use the target geometry from annotation and created image of exactly same size of the bounding box 

    Parameters
    --------------

    image_path: path of the image
    json_path: path of the json file 
    save_dir: directory to save the data
    
    '''


    # extracting target bounds and crs
    print(json_path)
    targets, json_crs = extract_centroid(json_path)
    print('number of targets', len(targets))
        
    image_name = os.path.basename(image_path).split('.')[0]
    with rasterio.open(image_path) as src:
        i_crs = src.crs
        i_tranform = src.transform
        out_meta = src.meta
    # targets - [('class', cx,cy)]
        project =Transformer.from_crs(json_crs, i_crs, always_xy = True) # transforming coordinated if crs difers
        num = 1
        for target in targets:
            subclass, geom = target
            geom =transform(project.transform, geom)
            patch, out_transform = mask(src , [geom], crop = True)
            image_name_save = image_name+ '_' + str(num)+'.tif'
            
            if subclass == None:
                subclass = 'Undefined'


            save_path = os.path.join(save_dir, subclass)

            os.makedirs(save_path, exist_ok = True)
            save_image_path = os.path.join(save_path, image_name_save)
            if hbb ==False:
                # print("save file" ,save_image_path)
            
                out_meta.update({
                    "driver": "GTiff",
                    "height": patch.shape[1],
                    "width": patch.shape[2],
                    "transform": out_transform,
                    "dtype": patch.dtype
                    })
                with rasterio.open(save_image_path, 'w', **out_meta) as out:
                    out.write(patch)
                num +=1
            else:
                rows = []
                cols = []
                if geom.geom_type == "Polygon":
                    geometries = [geom]
                elif geom.geom_type == "MultiPolygon":
                    geometries = list(geom.geoms)
                else:
                    raise ValueError(f"Unsupported geometry type: {geom.geom_type}")

                for g in geometries:
                    for x, y in g.exterior.coords:
                        r, c = rowcol(i_tranform, x, y)
                        rows.append(r)
                        cols.append(c)

                # Convert polygon coordinates to pixel (row, col)
                

                # for x, y in geom.exterior.coords:
                #     r, c = rowcol(i_tranform, x, y)
                #     rows.append(r)
                #     cols.append(c)

                # Pixel-space horizontal bounding box
                row_min, row_max = min(rows), max(rows)
                col_min, col_max = min(cols), max(cols)

                # Build raster window
                window = Window(
                    col_off=col_min,
                    row_off=row_min,
                    width=col_max - col_min,
                    height=row_max - row_min
                )

                # Read data (NO resampling, NO NoData padding)
                hbb_patch = src.read(window=window)

                # Update metadata
                out_meta.update({
                    "driver": "GTiff",
                    "height": hbb_patch.shape[1],
                    "width": hbb_patch.shape[2],
                    "transform": rasterio.windows.transform(window, i_tranform),
                    "dtype": hbb_patch.dtype
                })

                # Save
                with rasterio.open(save_image_path, "w", **out_meta) as out:
                    out.write(hbb_patch)

                num += 1
                #----------mycode-------
                # minx, miny, maxx, maxy = polygon_to_hbb(geom)
                # try:
                #     window = from_bounds(minx, miny, maxx, maxy , i_tranform)
                #     hbb_patch = src.read(window = window)
                #     out_meta.update({
                #     "driver": "GTiff",
                #     "height": patch.shape[1], 
                #     "width": patch.shape[2],
                #     "transform":rasterio.windows.transform(window, i_tranform)
                #     })
                #     with rasterio.open(save_image_path, "w", **out_meta) as out:
                #         out.write(hbb_patch)
                #     num += 1
                # except Exception:
                #     gg = box(minx, miny, maxx, maxy)
                #     patch, out_transform = mask(src, [gg], crop = True) 
                #     # pdb.set_trace()
                #     out_meta.update({
                #         "height": patch.shape[1],
                #         "width": patch.shape[2],
                #         "transform": out_transform
                #     })    
                #     with rasterio.open(save_image_path, "w",**out_meta ) as dest:
                #         dest.write(patch)
                #     num += 1

            # # Defining bounding box
            # row, col = rowcol( i_tranform, cx,cy)
            # row_start   = row - image_size//2
            # col_start = col - image_size//2
            # window = Window(col_start, row_start, image_size, image_size)
            # patch = src.read(window = window)


            # patch = np.transpose(patch, (1,2,0))
            # if patch.shape[-1] ==1:
            #     patch = patch[:,:,0]
            # # patch  = normalize(patch)
            # # np.save(save_image_path+'.npy', patch)
            # # patch = np.uint8(patch *255)
            # img = Image.fromarray(patch)
            # img.save(save_image_path+'.tif')
            # num +=1

        

def train_data(ann_dir , save_dir, hbb = True):
    """
    create training directory by taking json and image from the annotation directory and save 
    data in the new directory

    Parameters
    -------------

    ann_dir:
         directory where the original image and annotation json files are there 
    save_dir: 
        directory where images will be saved 
    
    """
    files = glob.glob(ann_dir + '/*')
    tifs = []
    jsons = []
    no_use = []
    for i in files:
        if i.split('.')[-1] == 'tif':
            tifs.append(i)
        elif i.split('.')[-1] == ".xml":
            no_use.append(i)
        else:
            jsons.append(i)
    jsons = sorted(jsons)
    tifs = sorted(tifs)
    # ipdb.set_trace()

    for json, tif in zip(jsons, tifs):
        target_centers(tif, json, save_dir = save_dir, hbb = hbb)


if __name__ == "__main__":

    save_dir = "hbbdata2"
    ann_dir = "/media/sphere/744c0eb8-a6d5-469f-9d27-449a9eef9aa1/home/admin123/Kanishk/Kan"

    train_data(ann_dir, save_dir, hbb=True)