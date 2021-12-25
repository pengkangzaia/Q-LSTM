import torch.cuda

# from psm_main import *
# test_psm(nrows=87841)
from swat_main import *
test_swat(nrows=449919)

# swat test 449919 train 495000
# 87841 test
# df = pd.read_csv('data/psm/test.csv')
# shape = df.shape
# pass
# a = torch.cuda.is_available()


# from smd_main import *

# train_smd(nrows=1000)
# test_smd_for_all_entity(nrows=None)



pass