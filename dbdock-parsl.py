#!/mnt/lustre_fs/users/kavotaw/apps/anaconda/anaconda3/envs/parsl_py36/bin/python3
import parsl, os
from config import mcullaghcluster
from library import *
from parsl.data_provider.files import File
parsl.set_stream_logger()
parsl.load(mccluster)

# RIGID DOCKING
ligands=open('list.dat','r')
for i in ligands:
    run = vina(inputs=[i],outputs=[rigid_docking_data])
run.result()

# RIGID ML
names, mols = getNamesMols(inputs=[ligands,rigid_docking_data])
names, features = getAllFeatures(inputs=[names,mols],outputs=[features_file])
train_and_test_svm_and_nn(inputs=[rigid_docking_data, features_file],outputs=[svm_model, r2_svm, nn_model, r2_nn, rigid_features_and_energies])

# SELECT TOP RESULTS
percent='10'
select_top_percent(inputs=[percent,rigid_features_and_energies],outputs=[top_ligands])

# FLEXIBLE DOCKING
for i in top_ligands:
    run = adfr(inputs=[i])
run.result()

# FLEXIBLE ML
train_and_test_svm_and_nn(inputs=[flexible_docking_data, rigid_features_and_energies],outputs=[svm_model, r2_svm, nn_model, r2_nn, flexible_features_and_energies])

# SELECT TOP RESULTS
percent='1'
select_top_percent(inputs=[percent,flexible_feaures_and_energies],outputs=[top_ligands])
np.savetxt('top_flexible.dat',top_ligands)
