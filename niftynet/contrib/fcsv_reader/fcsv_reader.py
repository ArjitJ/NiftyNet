import numpy as np
import pandas as pd
from niftynet.layer.base_layer import Layer


class FCSVReader(Layer):
    def __init__(self, name="fcsv_reader"):
        self.fid_pts = None
        super(FCSVReader, self).__init__(name=name)

    def initialise(self, path_to_fcsv, fiducials=["RE"]):
        """
         Assuming that the input file format is label,x,y,z,sel,vis
        :param path_to_fcsv:
        :param fiducials:
        :return:
        """
        df = pd.read_csv(
            path_to_fcsv,
            header=None,
            comment="#",
            names=["fiducial_labels", "xcoord", "ycoord", "zcoord", "sel", "vis"],
        )
        self.fid_pts = {}
        for fid in fiducials:
            fid_df = df[df["fiducial_labels"] == fid]
            # the sign flipping in x and y is to convert RAS(used by slicer) to LPS(used in DICOM and itk)
            xcoord = -1 * fid_df["xcoord"].values.reshape(-1, 1)
            ycoord = -1 * fid_df["ycoord"].values.reshape(-1, 1)
            zcoord = fid_df["zcoord"].values.reshape(-1, 1)
            one_d_vec = np.concatenate((xcoord, ycoord, zcoord), axis=1).reshape(-1, 1)
            self.fid_pts[fid] = one_d_vec
        return self

    def layer_op(self, fid="RE"):
        return self.fid_pts[fid]
