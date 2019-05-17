"""
License: Apache-2.0
Author: Eduardo Ch. Colorado
E-mail: edxz7c@gmail.com
"""
import sys
# Available NET Architectures: resnet-ish, lenet-like, capsule_net
model_name = sys.argv[1]
model_url  = sys.argv[2]

with open("app/avaible_models.txt", "a") as f:
    f.write("{} {}".format(model_name, model_url) + "\n")




