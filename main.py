import pandas as pd
from vintagecreator import VintageCreator
import vintageopt

vintagecreator = VintageCreator('2024-01-01')
vintages = vintagecreator.create_vintage()

vo = vintageopt.VintageOpt(vintages.iloc[:,0])
result = vo.optimiseSciPy(300)
print(result)
vo.plotSciPy(**result)
