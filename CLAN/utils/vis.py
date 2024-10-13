# visualization
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import RocCurveDisplay


def visualization_roc(class_num, y_onehot_test, y_score, vis_title, vis_path, pos_label = 1):
    # visualization        
    fig, ax = plt.subplots(figsize=(6, 6))
    if(len(class_num)<=5):
        colors = cycle(["tomato", "darkorange", "gold", "darkseagreen","dodgerblue"])
    else:
        colors = cycle(["firebrick", "tomato", "sandybrown", "darkorange", "olive", "gold", 
                        "darkseagreen", "darkgreen", "dodgerblue", "royalblue","slategrey",
                        "slateblue", "mediumpurple","indigo", "orchid", "hotpink"])
    for class_id, color in zip(range(len(class_num)), colors):
        #print(y_score[class_id].tolist())
        print(class_num, class_id)
        if class_num[class_id] != -1:
            RocCurveDisplay.from_predictions(
                y_onehot_test[class_id],
                y_score[class_id],
                name=f"{(class_num[class_id]+1)}",
                pos_label = pos_label,
                color=color,
                ax=ax, 
            )
        else:
            RocCurveDisplay.from_predictions(
                y_onehot_test[class_id],
                y_score[class_id],
                pos_label = pos_label,
                name=f"Multi",
                color="black",
                ax=ax, 
            )

            
    plt.axis("square")
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Chance Level (0.5)')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(vis_title)
    plt.legend()
    plt.show()
    plt.savefig(vis_path)
    
def box_plot_vis(labels, scores, path):
    # Generate data
    known_data = [-1 * b_elem for a_elem, b_elem in zip(labels, scores) if a_elem == 0] 
    new_data = [-1 * b_elem for a_elem, b_elem in zip(labels, scores) if a_elem == 1]

    # Set the figure size
    plt.figure(figsize=(4,3))

    # Create a DataFrame for boxplot
    data = pd.DataFrame({"Known": pd.Series(known_data).reindex_like(pd.Series(new_data)), "New": new_data})


    # Plot individual data points for 'Known'
    x = np.random.normal(1, 0.1, len(known_data))
    plt.plot(x, known_data, mfc='red', mec='k', ms=7, marker="o", linestyle="None")

    # Plot individual data points for 'New'
    x = np.random.normal(2, 0.1, len(new_data))
    plt.plot(x, new_data, mfc='black', mec='grey', ms=7, marker="o", linestyle="None")

    # Create boxplot
    boxplot_dict = data.boxplot(return_type='dict')

    # Set linewidth for each boxplot element
    linewidth = 2
    for line in boxplot_dict['boxes']:
        line.set_linewidth(linewidth)
    for whisker in boxplot_dict['whiskers']:
        whisker.set_linewidth(linewidth)
    for cap in boxplot_dict['caps']:
        cap.set_linewidth(linewidth)
    for median in boxplot_dict['medians']:
        median.set_linewidth(linewidth)
    for flier in boxplot_dict['fliers']:
        flier.set_marker('+')
        flier.set_markersize(8)
        flier.set_markeredgewidth(linewidth)

    plt.show()
    plt.savefig(path)
    
def box_plot_vis_rv(labels, scores, path):
    # Generate data
    known_data = [b_elem for a_elem, b_elem in zip(labels, scores) if a_elem == 1] 
    new_data = [b_elem for a_elem, b_elem in zip(labels, scores) if a_elem == 0]

    # Set the figure size
    plt.figure(figsize=(4,3))
    
    # Create a DataFrame for boxplot
    data = pd.DataFrame({"Known": pd.Series(known_data).reindex_like(pd.Series(new_data)), "New": new_data})

    # Plot individual data points for 'Known'
    x = np.random.normal(1, 0.1, len(known_data))
    plt.plot(x, known_data, mfc='red', mec='k', ms=7, marker="o", linestyle="None")

    # Plot individual data points for 'New'
    x = np.random.normal(2, 0.1, len(new_data))
    plt.plot(x, new_data, mfc='black', mec='grey', ms=7, marker="o", linestyle="None")

    # Create boxplot
    boxplot_dict = data.boxplot(return_type='dict')

    # Set linewidth for each boxplot element
    linewidth = 2
    for line in boxplot_dict['boxes']:
        line.set_linewidth(linewidth)
    for whisker in boxplot_dict['whiskers']:
        whisker.set_linewidth(linewidth)
    for cap in boxplot_dict['caps']:
        cap.set_linewidth(linewidth)
    for median in boxplot_dict['medians']:
        median.set_linewidth(linewidth)
    for flier in boxplot_dict['fliers']:
        flier.set_marker('+')
        flier.set_markersize(8)
        flier.set_markeredgewidth(linewidth)

    plt.show()
    plt.savefig(path)
    
def print_rs(final_1, final_2, final_3, final_4, final_5, save_path):
     # for extrating results to an excel file
    final_rs =[]
    for i in final_1:
        final_rs.append(i)
    for i in final_2:
        final_rs.append(i)
    for i in final_3:
        final_rs.append(i)
    for i in final_4:
        final_rs.append(i)
    for i in final_5:
        final_rs.append(i)
        
    df = pd.DataFrame(final_rs, columns=['mean', 'std'])
    df.to_excel(save_path, sheet_name='the results')
