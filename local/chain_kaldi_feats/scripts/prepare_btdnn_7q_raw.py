import sys

path = sys.argv[1]

flag = 1
bnn_flag = 0

count = 1
with open(path+'/make_mdl/prior.raw', 'w') as out_f:
    with open(path+'/0.raw', 'r') as f:
        for line1 in f.readlines():
            if line1.startswith('<LinearParamsMean>'):
                out_f.write(line1)
                bnn_flag = 0
                if count == 1:
                    with open(path+'/make_mdl/mid.mdl', 'r') as f1:
                        flag = 0
                        for line in f1.readlines():
                            if line.startswith('<ComponentName> tdnn'+str(count)+'.affine'):
                                flag = 1
                                bnn_flag = 1
                            elif line.startswith('<BiasParams>') and bnn_flag == 1:
                                flag = 0
                                break
                            if flag == 1 and not line.startswith('<ComponentName> tdnn'+str(count)+'.affine') and not line.startswith('<LinearParams>'):
                                out_f.write(line)
                else:
                    with open(path+'/make_mdl/mid.mdl', 'r') as f1:
                        flag = 0
                        for line in f1.readlines():
                            if line.startswith('<ComponentName> tdnnf'+str(count)+'.affine'):
                                flag = 1
                                bnn_flag = 1
                            elif line.startswith('<BiasParams>') and bnn_flag == 1:
                                flag = 0
                                break
                            if flag == 1 and not line.startswith('<ComponentName> tdnnf'+str(count)+'.affine') and not line.startswith('<LinearParams>'):
                                out_f.write(line)

            if line1.startswith('<LinearParamsStd>'):
                flag = 1


            if line1.startswith('<LinearPriorMean>'):
                out_f.write(line1)
                bnn_flag = 0
                if count == 1:
                    with open(path+'/make_mdl/final_txt.mdl', 'r') as f1:
                        flag = 0
                        for line in f1.readlines():
                            if line.startswith('<ComponentName> tdnn'+str(count)+'.affine'):
                                flag = 1
                                bnn_flag = 1
                            elif line.startswith('<BiasParams>') and bnn_flag == 1:
                                flag = 0
                                break
                            if flag == 1 and not line.startswith('<ComponentName> tdnn'+str(count)+'.affine') and not line.startswith('<LinearParams>'):
                                out_f.write(line)
                else:
                    with open(path+'/make_mdl/final_txt.mdl', 'r') as f1:
                        flag = 0
                        for line in f1.readlines():
                            if line.startswith('<ComponentName> tdnnf'+str(count)+'.affine'):
                                flag = 1
                                bnn_flag = 1
                            elif line.startswith('<BiasParams>') and bnn_flag == 1:
                                flag = 0
                                break
                            if flag == 1 and not line.startswith('<ComponentName> tdnnf'+str(count)+'.affine') and not line.startswith('<LinearParams>'):
                                out_f.write(line)

            if line1.startswith('<LinearPriorStd>'):
                flag = 1
                count += 1

            if flag == 1:
                out_f.write(line1)
