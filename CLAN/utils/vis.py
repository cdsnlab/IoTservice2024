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
