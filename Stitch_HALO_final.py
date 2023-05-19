from aicsimageio import AICSImage
import os
import dask.array as da
import numpy as np
import re
import logging

from skimage.measure import label, regionprops
from skimage.morphology import closing, square

from tifffile import TiffWriter
import tifffile
from skimage.transform import resize
from xml.etree import ElementTree as ET

import sys, argparse

def main(argv):
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Stitch a bunch of HALO unmixed regions")
    parser.add_argument("Input File Path", help='Directory of image files')
    parser.add_argument('--debug', '-d', action='store_true')

    args = parser.parse_args()
    input_args = vars(args)

    fpath = input_args["Input File Path"]
    logging.info("fpath is %s" % fpath)

    outDir = '/vast/scratch/users/whitehead/stitches/'

    flist = os.listdir(fpath)
    # regex to find coords
    # todo can possibly use group 1 and 2 for output filename?
    # todo probably easiest to manually name or something though...
    reg0 = r"(.*)_\[([0-9]+),([0-9]+)\]_component_data\.tif"

    # just find files that match the regex - check filenames...
    this_image = [f for f in flist if re.match(reg0, f)]
    this_image = sorted(this_image)
    logging.info("There are %s tiles found" % (len(this_image)))

    # todo from metadata probably? allow setting
    pixel_scale = 0.49641833810888253

    im_boxes = []

    for idx, f in enumerate(this_image):
        m = re.match(reg0, f)
        img_label = m.group(1)
        logging.info("Image label is %s" % img_label)
        x_val = int(int(m.group(2)) / pixel_scale)
        y_val = int(int(m.group(3)) / pixel_scale)

        # Parse region
        img_path = os.path.join(fpath, f)
        img = AICSImage(img_path)
        z, c, t, x, y = img.shape
        im_boxes.append([x_val, y_val, x_val + x, y_val + y])

        if (idx == 0):
            # If first image, open using tiffffile to read channel names
            imgData = tifffile.TiffFile(img_path)
            imgData_s0 = imgData.series[0]
            names = [(ET.fromstring(page.description).find('Name').text) for page in imgData_s0]
            colors = [(ET.fromstring(page.description).find('Color').text) for page in imgData_s0]

    new_colors = []
    for cl in colors:
        cl = str.split(cl, ',')
        a = 1
        r = int(cl[0])
        g = int(cl[1])
        b = int(cl[2])
        RGBint = int.from_bytes([r, g, b, a], byteorder="big", signed=True)
        new_colors.append(RGBint)

    xvals = [x[0] for x in im_boxes]
    yvals = [x[1] for x in im_boxes]
    x_end_vals = [x[2] for x in im_boxes]
    y_end_vals = [x[3] for x in im_boxes]

    im_xmin = min(xvals)
    im_ymin = min(yvals)
    im_xmax = max(x_end_vals)
    im_ymax = max(y_end_vals)

    final_dimensions = (im_xmax - im_xmin, im_ymax - im_ymin)

    # todo also allow setting by cli
    scale_by = 2
    small_version = da.zeros((int(final_dimensions[0] / scale_by), int(final_dimensions[1] / scale_by)))

    for idx, f in enumerate(this_image):
        impath = os.path.join(fpath, f)

        m = re.match(reg0, f)
        x_val = int(int(m.group(2)) / pixel_scale)
        y_val = int(int(m.group(3)) / pixel_scale)

        img_path = os.path.join(fpath, f)
        img = AICSImage(img_path)
        z, c, t, h, w = img.shape
        xStart = x_val - im_xmin
        yStart = y_val - im_ymin
        xEnd = xStart + w
        yEnd = yStart + h

        xStart = int(xStart / scale_by)
        yStart = int(yStart / scale_by)
        yEnd = int(yEnd / scale_by)
        xEnd = int(xEnd / scale_by)

        try:
            small_version[xStart:xEnd, yStart:yEnd] = 1
        except:
            logging.warning("SHAPE MISMATCH")
            logging.warning("ROI:%i - %s" % (idx, f))

    # sometimes pixels go wonky, filter first
    small_version = small_version.compute()
    clsed = closing(small_version, square(1))
    lbl = label(clsed)
    regions = regionprops(lbl)

    logging.info("There are %i regions" % len(regions))

    # Now do the full thing
    # First initialise the dask array
    full_height = final_dimensions[0]
    full_width = final_dimensions[1]

    stitched_image = da.zeros((1, int(c), 1, full_height, full_width))
    stitched_image = stitched_image.rechunk((1, int(c), 1,
                                             int(full_height / 10),
                                             int(full_width / 10)))

    # The "meat"
    logging.info("Putting images where they go")
    for idx, f in enumerate(this_image):
        impath = os.path.join(fpath, f)

        m = re.match(reg0, f)
        x_val = int(int(m.group(2)) / pixel_scale)
        y_val = int(int(m.group(3)) / pixel_scale)

        # Parse region
        img_path = os.path.join(fpath, f)
        img = AICSImage(img_path)
        z, c, t, h, w = img.shape
        im_data = img.get_image_data()
        im_data = np.swapaxes(im_data, 3, 4)

        xStart = x_val - im_xmin
        yStart = y_val - im_ymin
        xEnd = xStart + w
        yEnd = yStart + h

        xc, yc = xStart + w / 2, yStart + w / 2

        try:
            stitched_image[:, :, :, xStart:xEnd, yStart:yEnd] = im_data

        except ValueError:
            pass

        except:
            roiShape = stitched_image[:, :, :, xStart:xEnd, yStart:yEnd].shape

            stitched_image[:, :, :, xStart:xEnd, yStart:yEnd] = im_data[:, :, :, 0:roiShape[3], 0:roiShape[4]]

    logging.info("DONE")

    full_image = stitched_image

    # now save out over regions
    for idx, r in enumerate(regions):
        logging.info("Doing region number %i" % idx)
        y0, x0, y1, x1 = r.bbox

        x0 = x0 * scale_by
        y0 = y0 * scale_by
        y1 = y1 * scale_by
        x1 = x1 * scale_by
        print(x0, x1, y0, y1)
        stitched_image = full_image[:, :, :, y0:y1, x0:x1]

        im_to_write = stitched_image
        sub_fig1 = resize(im_to_write, (im_to_write.shape[0],
                                        im_to_write.shape[1],
                                        im_to_write.shape[2],
                                        im_to_write.shape[3] / 4,
                                        im_to_write.shape[4] / 4))

        sub_fig2 = resize(sub_fig1, (sub_fig1.shape[0],
                                     sub_fig1.shape[1],
                                     sub_fig1.shape[2],
                                     sub_fig1.shape[3] / 4,
                                     sub_fig1.shape[4] / 4))

        outFileName =  "%s_stitch_roi_%i.ome.tif" % (img_label,idx)

        out_path = '%s%s.ome.tif' % (outDir, outFileName)
        logging.info("output is in %s" % out_path)

        with TiffWriter(out_path, bigtiff=True) as tif:
            options = dict(tile=(512, 512), compression='zlib',
                           metadata={'axes': 'TCZYX',
                                     'PhysicalSizeX': pixel_scale,
                                     'PhysicalSizeY': pixel_scale,
                                     'Channel': {'Name': names, 'Color': new_colors}
                                     }
                           )

            tif.write(im_to_write, subifds=2, **options)
            # save pyramid levels to the two subifds
            tif.write(sub_fig1, subfiletype=1, **options)
            tif.write(sub_fig2, subfiletype=1, **options)


if __name__ == "__main__":
    main(sys.argv[1:])
