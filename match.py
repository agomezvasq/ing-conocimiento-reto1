import numpy as np
import cv2
import os


class HistMatcher:

    def __init__(self, template):
        self.template = template
        bt, gt, rt = cv2.split(self.template.astype("float"))
        self.bt_quantiles, self.bt_values = self.get(bt)
        self.gt_quantiles, self.gt_values = self.get(gt)
        self.rt_quantiles, self.rt_values = self.get(rt)

    def get(self, template):
        template = template.ravel()
        t_values, t_counts = np.unique(template, return_counts=True)
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]
        return t_quantiles, t_values


    def hist_match(self, source, t_quantiles, t_values):
        """
        Adjust the pixel values of a grayscale image such that its histogram
        matches that of a target image

        Arguments:
        -----------
            source: np.ndarray
                Image to transform; the histogram is computed over the flattened
                array
            template: np.ndarray
                Template image; can have different dimensions to source
        Returns:
        -----------
            matched: np.ndarray
                The transformed output image
        """

        oldshape = source.shape
        source = source.ravel()

        # get the set of unique pixel values and their corresponding indices and
        # counts
        s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                                return_counts=True)

        # take the cumsum of the counts and normalize by the number of pixels to
        # get the empirical cumulative distribution functions for the source and
        # template images (maps pixel value --> quantile)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]

        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

        return interp_t_values[bin_idx].reshape(oldshape)

    def hist_match_rgb(self, source):
        bs, gs, rs = cv2.split(source.astype("float"))

        b = self.hist_match(bs, self.bt_quantiles, self.bt_values).astype("uint8")
        g = self.hist_match(gs, self.gt_quantiles, self.gt_values).astype("uint8")
        r = self.hist_match(rs, self.rt_quantiles, self.rt_values).astype("uint8")

        return cv2.merge((b, g, r))


SHOW = False
SAVE = True

if SHOW:
    windows = ["template", "img", "match"]
    i = 0
    j = 0
    size = 500
    for window in windows:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, size, size)
        cv2.moveWindow(window, i * (size + 100), j * (size + 100))
        i += 1
        if i > 2:
            i = 0
            j += 1


templates = {"chocorramo": "2018-02-23-092723.png",
             "flow_blanca": "2018-02-23-092210.png",
             "flow_negra": "2018-02-23-092420.png",
             "frunas_amarilla": "2018-02-23-093203.png",
             "frunas_naranja": "2018-02-23-093036.png",
             "frunas_roja": "2018-02-23-093124.png",
             "frunas_verde": "2018-02-23-092925.png",
             "jet_azul": "2018-02-23-143047.png",
             "jumbo_naranja": "2018-02-23-092250.png",
             "jumbo_roja": "2018-02-23-092648.png"}


for subdir, dirs, files in os.walk("data/train/cropped2"):
    if len(files) > 0:
        template = cv2.imread(subdir + "/" + templates[os.path.basename(subdir)])

        matcher = HistMatcher(template)

        for filename in files[1:]:
            if filename.endswith(".png"):
                img = cv2.imread(subdir + "/" + filename)

                match = matcher.hist_match_rgb(img)

                if SAVE:
                    path = "data/train/match/" + os.path.basename(subdir) + "/" + filename
                    cv2.imwrite(path, match.astype("uint8"))
                    print(path)

                if SHOW:
                    images = [template, img, match]
                    for i in range(len(windows)):
                        cv2.imshow(windows[i], images[i])
                    cv2.waitKey(0)