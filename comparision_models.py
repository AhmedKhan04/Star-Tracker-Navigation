

import numpy as np 
import Anchoring_Data as AD
import re
import numpy as np


class comparision_extractor:
    def __init__(self, data, NameStar):
        self.date_array_real, self.model_real, model_string_real = data
        self.star_name = NameStar
        self.anchoring_instance = AD.anchoringData(NameStar)

        self.model_string_real = re.sub(r'π', 'np.pi', self.model_string_real)
        self.model_string_real = re.sub(r'\bsin\b', 'np.sin', self.model_string_real)
        self.model_string_real = re.sub(r'\s+', ' ', self.model_string_real)

        self.model_anchored_real_time = eval(self.model_string_real, {"np": np, "t": self.date_array_real})
        
        # normalize both of these models. 
        self.model_anchored_real_time = self.model_anchored_real_time / np.max(np.abs(self.model_anchored_real_time))
        self.model_real = self.model_real / np.max(np.abs(self.model_real))
        



        # evaluate the anchored model at the real dates

        



    
