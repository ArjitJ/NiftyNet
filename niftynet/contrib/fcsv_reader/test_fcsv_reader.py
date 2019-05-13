from fcsv_reader import *

my_fcsv_reader = FCSVReader()
my_fcsv_reader = my_fcsv_reader.initialise(
    path_to_fcsv="0001_27588.fcsv", fiducials=["LE", "RE"]
)
print(my_fcsv_reader.layer_op())
print(my_fcsv_reader.layer_op(fid="LE"))
