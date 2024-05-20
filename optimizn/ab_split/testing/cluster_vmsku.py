import importlib.resources
import pandas as pd

# Note MANIFEST.in file in the root specifies that this CSV should be packaged.
f1 = importlib.resources.open_text("optimizn.ab_split.testing",
                                   "Canary_ClusterVMSKU.csv")

cluster_vmsku = pd.read_csv(f1)
