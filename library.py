#!/mnt/lustre_fs/users/kavotaw/apps/anaconda/anaconda3/envs/parsl_py36/bin/python3
import parsl
from parsl.app.app import python_app, bash_app
from dbdock.preprocessing import computeFeatures
from dbdock.ml import get_data, train_SVM, train_NN, pca, autocorr

@bash_app
def rigid_docking(inputs=[], outputs=[],stderr='std-vina.err', stdout='std-vina.out'):
    return './vina.sh %s %s' % (inputs[0],outputs[0])
@bash_app
def flexible_docking(inputs=[],stderr='std-adfr.err', stdout='std-adfr.out'):
    return './adfr.sh %s' % (inputs[0])
@bash_app
def select_top_percent(inputs=[percent,svm_model,nn_model],outputs=[top_ligands])
    return './select.sh %s %s %s %s' % (inputs[0],inputs[1],inputs[2],outputs[0])

@python_app
def getNamesMols(inputs=[input_ligands_path,data_binaries_dir]):
    try:
            print(" Attempting to load RDKit mol objects.")
            allValidMols = np.load("ligand_name_rdkit_mol.npy")
            print(" {} RDKit mol objects loaded.".format(len(allValidMols)))
    except:
            print(" No RDKit mol object binary found. Using ligand PDB / PDBQT files to generate new features...")
            allValidMols = []
            ligand_list = os.listdir(input_ligands_path)
            fails = 0
            for ligand_file in ligand_list:
                    if ligand_file[-6:] == ".pdbqt":
                            try:
                                    convert_PDBQT_to_PDB("{}{}".format(input_ligands_path,ligand_file))
                            except IOError:
                                    fails += 1
                                    continue
            ligand_list = os.listdir(input_ligands_path)
            for ligand_file in ligand_list:
                    if ligand_file[-4:] == ".pdb":
                            ligand_name = ligand_file[:-4]
                            mol = Chem.MolFromPDBFile("{}{}".format(input_ligands_path,ligand_file))
                            if (mol != None):
                                    allValidMols.append([ligand_name,mol])
            print " Read in {} ligand files, encountered {} failures.".format(len(ligand_list),fails)
    
    allValidMols = np.asarray(allValidMols)
    np.save("{}ligand_name_rdkit_mol.npy".format(data_binaries_dir),allValidMols)
    print(allValidMols[0])
    names,mols = allValidMols[:,0],allValidMols[:,1]
    return names,mols

@python_app
def getAllFeatures(inputs=[names,ligands],outputs=[features_bin_dir]):
    features = []
    labels = []
    count = 0
    fails = 0
    print("Using generated RDKit mol objects to produce feature sets...")
    for lig in range(len(ligands)):
            if (count % 100 == 0):
                    print("Ligand No: {} / {}".format(count,len(ligands)))
            f_data = computeFeatures(ligands[lig])
            d_hist = f_data[13:33]                                                                                                                  
            all_zero = True                                                                                                                        
            keep = True                                                                                                                             
            for d in d_hist:
                    if d != 0:
                            all_zero = False
                    elif d == 99999:
                            fails += 1
                            keep = False
                            break
            if all_zero:
                    fails += 1
                    keep = False
            if keep:
                    features.append(f_data)
            count += 1
    features = np.asarray(features)
    print("Collected  {}  features per sample for {} samples ({} failures)".format(len(features[0]),len(features),fails))
    features = preprocessing.normalize(features,norm='l2',axis=0)
    allValidFeatures = [names, [features]]
    np.save("{}".format(features_bin_dir),allValidFeatures)
return names, features

@python_app
def train_and_test_svm_and_nn(inputs=[ligand_file, label_file], outputs=[svm_model, r2_svm, nn_model, r2_nn, features_plus_energies]):
     svm_tr_X, svm_tr_y, svm_ts_X, svm_ts_y, sr1,sr2 = get_data(ligand_file, label_file, batch=False,subset="all_features")
     nn_tr_X, nn_tr_y, nn_ts_X, nn_ts_y, nr1,nr2 = get_data(ligand_file, label_file, batch=True,subset="all_features")
     avg_ac, std_ac, autocorr_mat, x_vects = autocorr(svm_tr_X,svm_ts_X)	
     pc1 = pca(autocorr_mat,x_vects)
     svm_model, r2_svm = train_SVM(svm_tr_X, svm_tr_y, svm_ts_X, svm_ts_y)
     nn_model, r2_nn = train_NN(nn_tr_X, nn_tr_y, nn_ts_X, nn_ts_y)
     svm_pred = svm_model.predict(svm_ts_X)
     nn_pred = []
     actual = []
     losses = []
     for batch in range(len(nn_ts_X)):
     	batch_prediction = nn_model(Variable(torch.FloatTensor(nn_ts_X[batch])))
     	batch_prediction = batch_prediction.data.numpy()
     	for p in range(len(batch_prediction)):
     		nn_pred.append(batch_prediction[p])
     		actual.append(nn_ts_y[batch][p])
     		losses.append(abs(nn_pred[-1] - nn_ts_y[batch][p]))
     plt.figure()
     plt.plot(actual,svm_pred,'x',color='b',ms=2,mew=3)
     plt.plot(actual,actual,'x',color='r',ms=2,mew=3)
     plt.suptitle("SVM Prediction Performance\nTraining Set: {}  Test Set: {}".format(training_set_size,test_set_size))
     plt.ylabel("Predicted Binding Affinity")
     plt.xlabel("Known Binding Affinity")
     plt.savefig('svm_rigid.png',dpi=600,transparent='True')
     plt.close()
     
     plt.figure()
     plt.plot(actual,nn_pred,'x',color='g',ms=2,mew=3)
     plt.plot(actual,actual,'x',color='r',ms=2,mew=3)
     plt.suptitle("NN Prediction Performance\nTraining Set: {}  Test Set: {}".format(training_set_size,test_set_size))
     plt.ylabel("Predicted Binding Affinity")
     plt.xlabel("Known Binding Affinity")
     plt.savefig('nn_rigid.png',dpi=600,transparent='True')
     plt.close()
     
     fig, ax = plt.subplots()
     plt.plot(actual,svm_pred,'o',color='b',ms=2,label='SVM')
     plt.plot(actual,nn_pred,'o',color='g',ms=2,label='NN')
     plt.plot(actual,actual,'-',color='r',ms=2,mew=3)
     plt.suptitle("SVM vs NN Prediction Performance\nTraining Set: {}  Test Set: {}".format(training_set_size,test_set_size))
     plt.ylabel("Predicted Binding Affinity")
     plt.xlabel("Known Binding Affinity")
     ax.set_aspect('equal')
     plt.savefig('svm-nn_rigid.png',dpi=600,transparent='True')
     plt.close()       
                                 	
return svm_model, r2_svm, nn_model, r2_nn
