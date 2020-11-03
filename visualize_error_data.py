import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

#not sure if all the inner arrays are numpy or lists
# so redefining these
def error_sum(errors):
    err_sum = np.zeros(errors.shape[0])
    for i in range(errors.shape[0]):
        err_sum[i] = np.sum(errors[i])
    return err_sum

def error_mean(errors):
    mean = np.zeros(errors.shape[0])
    for i in range(errors.shape[0]):
        mean[i] = np.mean(errors[i])
    return mean

def error_max(errors):
    max_val = np.zeros(errors.shape[0])
    for i in range(errors.shape[0]):
        max_val[i] = np.max(errors[i])
    return max_val

def error_min(errors):
    min_val = np.zeros(errors.shape[0])
    for i in range(errors.shape[0]):
        min_val[i] = np.min(errors[i])
    return min_val

def error_stddev(errors):
    stddev = np.zeros(errors.shape[0])
    for i in range(errors.shape[0]):
        stddev[i] = np.std(errors[i])
    return stddev

def error_var(errors):
    var = np.zeros(errors.shape[0])
    for i in range(errors.shape[0]):
        var[i] = np.var(errors[i])
    return var
#----------------------------------------------

def compare_means(error_gpr_new, error_gpr_old_constant, error_gpr_old_linear):

    num_kernels = len(error_gpr_new.keys())
    num_error_funcs = len(error_gpr_new[list(error_gpr_new.keys())[0]].keys())
    fig, ax = plt.subplots(num_kernels, num_error_funcs)

    # print the mean errors for all folds between gpr old and gpr new
    for i, key1 in enumerate(error_gpr_new.keys()):
        for j, key2 in enumerate(error_gpr_new[key1].keys()):
            
            # error from new gpr
            err_n = np.array(error_gpr_new[key1][key2])
            err_n_val = error_sum(err_n)

            err_o_c = np.array(error_gpr_old_constant[key1][key2])
            err_o_c_val = error_sum(err_o_c)

            err_o_l = np.array(error_gpr_old_linear[key1][key2])
            err_o_l_val = error_sum(err_o_l)

            if num_error_funcs > 1:
                ax[i,j].xaxis.set_major_locator(MaxNLocator(integer=True))
                ax[i,j].title.set_text(f"{key1}-{key2}")
                ax[i,j].plot(err_o_c_val,label="cnst")
                ax[i,j].plot(err_o_l_val, label="linr")
                ax[i,j].plot(err_n_val, label="new")
                ax[i,j].legend()
            elif num_error_funcs == 1:
                ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))
                ax[i].title.set_text(f"{key1}-{key2}")
                ax[i].plot(err_o_c_val,label="cnst")
                ax[i].plot(err_o_l_val, label="linr")
                ax[i].plot(err_n_val, label="new")
                ax[i].legend()

    plt.show()

def compare_ND(error_gpr_old_c, error_gpr_old_l, error_gpr_old_noND_c, error_gpr_old_noND_l):

    num_kernels = len(error_gpr_old_c.keys())
    num_error_funcs = len(error_gpr_old_c[list(error_gpr_old_c.keys())[0]].keys())
    fig, ax = plt.subplots(num_kernels, num_error_funcs)

    # print the mean errors for all folds between gpr old and gpr new
    for i, key1 in enumerate(error_gpr_new.keys()):
        for j, key2 in enumerate(error_gpr_new[key1].keys()):
            
            err_o_c = np.array(error_gpr_old_c[key1][key2])
            err_o_c_val = error_mean(err_o_c)

            err_o_l = np.array(error_gpr_old_l[key1][key2])
            err_o_l_val = error_mean(err_o_l)

            err_o2_c = np.array(error_gpr_old_noND_c[key1][key2])
            err_o2_c_val = error_mean(err_o2_c)

            err_o2_l = np.array(error_gpr_old_noND_l[key1][key2])
            err_o2_l_val = error_mean(err_o2_l)

            if num_error_funcs > 1:
                ax[i,j].title.set_text(f"{key1}-{key2}")
                ax[i,j].plot(err_o_c_val,'b',label="cnst")
                ax[i,j].plot(err_o_l_val, label="linr")
                ax[i,j].plot(err_o2_c_val,'--b',label="cnst_noND")
                ax[i,j].plot(err_o2_l_val, label="linr_noND")
                ax[i,j].legend()
            elif num_error_funcs == 1:
                ax[i].title.set_text(f"{key1}-{key2}")
                ax[i].plot(err_o_c_val,'b',label="cnst")
                ax[i].plot(err_o_l_val,'g',label="linr")
                ax[i].plot(err_o2_c_val,'--b',label="cnst_noND")
                ax[i].plot(err_o2_l_val,'--g',label="linr_noND")
                ax[i].legend()

    plt.show()

def plot_all_folds(error_gpr_new, error_gpr_old_constant, error_gpr_old_linear):
    
    key1 = list(error_gpr_new.keys())[0]
    key2 = list(error_gpr_new[key1].keys())[0]

    num_kernels = len(error_gpr_new)
    num_error_funcs = len(error_gpr_new[key1])
    num_folds = len(error_gpr_new[key1][key2])
    num_samples_per_fold = error_gpr_new[key1][key2][0].shape[0]

    fig, ax = plt.subplots(num_kernels + num_error_funcs, num_folds)

    # print the mean errors for all folds between gpr old and gpr new
    for i, key1 in enumerate(error_gpr_new.keys()):
        for j, key2 in enumerate(error_gpr_new[key1].keys()):
            
            # error from new gpr
            err_n = np.array(error_gpr_new[key1][key2])
        
            # error from old gpr
            err_o_c = error_gpr_old_constant[key1][key2]
            err_o_l = error_gpr_old_linear[key1][key2]

            for k in range(num_folds):
                ax[i+j,k].set_xticks(np.arange(0, num_samples_per_fold, 4))
                ax[i+j,k].title.set_text(f"{key1}-{key2}-{k}")
                ax[i+j,k].plot(err_o_c[k],label="cnst")
                ax[i+j,k].plot(err_o_l[k], label="linr")
                ax[i+j,k].plot(err_n[k], label="new")
                ax[i+j,k].legend()

    plt.show()

def print_fold_summary(error_gpr):
    summary = {"mean":[],"sum":[],"min":[],"max":[],"var":[]}
    for key1 in error_gpr.keys():
        for key2 in error_gpr[key1].keys():
            err = np.array(error_gpr[key1][key2])
            summary["mean"].append(error_mean(err))
            summary["sum"].append(error_sum(err))
            summary["min"].append(error_min(err))
            summary["max"].append(error_max(err))
            summary["var"].append(error_var(err))

            num_folds = len(err) # yup, not efficient

    # legend for summary
    for key1 in error_gpr.keys():
        for key2 in error_gpr[key1].keys():
            print(f'\t\t\t---{key1}-{key2}',end="\t\t")
    print()

    # foldwise
    for i in range(num_folds):
        for j in range(len(summary["mean"])):
            print(f'{i} Sum: {summary["sum"][j][i]:6.3f} Mean: {summary["mean"][j][i]:.3f} ({np.sqrt(summary["var"][j][i]):.3f})',end=" ")
            print(f'Min: {summary["min"][j][i]:.3f} Max: {summary["max"][j][i]:6.3f} ',end=" ")
        print()
    
    # overall
    for j in range(len(summary["mean"])):
        print(f'  SUM: {np.mean(summary["sum"][j]):6.3f} MEAN: {np.mean(summary["mean"][j]):.3f} ({np.sqrt(np.mean(summary["var"][j])):.3f})',end=" ")
        print(f'MIN: {np.min(summary["min"][j]):.3f} MAX: {np.max(summary["max"][j]):6.3f} ',end=" ")
    print()

if __name__ == "__main__":
    error_data = pickle.load(open("./error_data.pckl","rb"))
    
    error_gpr_new = error_data['new']
    error_gpr_old_constant = error_data['old_constant']
    error_gpr_old_linear = error_data['old_linear']
    error_gpr_old_noND_constant = error_data['old_noND_constant']
    error_gpr_old_noND_linear = error_data['old_noND_linear']

    
    print("New GPR")
    print_fold_summary(error_gpr_new)
    
    print("\nOld GPR - constant")
    print_fold_summary(error_gpr_old_constant)
    print("\nOld GPR - constant - No ND")
    print_fold_summary(error_gpr_old_noND_constant)
    
    print("\nOld GPR - linear")
    print_fold_summary(error_gpr_new)
    print("\nOld GPR - linear - No ND")
    print_fold_summary(error_gpr_old_noND_linear)
    

    #compare_means(error_gpr_new, error_gpr_old_constant, error_gpr_old_linear)
    #compare_ND(error_gpr_old_constant, error_gpr_old_linear, error_gpr_old_noND_constant, error_gpr_old_noND_linear)
    #plot_all_folds(error_gpr_new, error_gpr_old_constant, error_gpr_old_linear)
    #plot_all_folds(error_gpr_new, error_gpr_old_noND_constant, error_gpr_old_noND_linear)
   