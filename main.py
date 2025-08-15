import pandas as pd
from vintagecreator import VintageCreator
import vintageopt
import matplotlib.pyplot as plt
vintagecreator = VintageCreator('2022-01-01')
vintages = vintagecreator.create_vintage()

for i,j in enumerate(['Low', 'Medium', 'High']):
    vo = vintageopt.VintageOpt(vintages.iloc[:,i])
    A,B = vo.optimiseSciPy(2000)
    vo.plotSciPy(A,B,j)
