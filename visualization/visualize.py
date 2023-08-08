
'''
Author: Anonymous submission 
Time: 2023-04-02 09:46 
'''

import matplotlib.pyplot as plt 
import numpy as np 

plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True 

plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams['ps.fonttype'] = 42 

# sentence = "the crab cakes are delicious and the bbq rib was perfect ."

# tag_y = ["1", "2", "4", "8"] 
# tag_x = ["1", "2", "4", "8"] 


'''
A = [[0.0024, 0.0042, 0.0056, 0.0273, 0.3829, 0.0109, 0.0026, 0.0037, 0.0034,
         0.0975, 0.4541, 0.0054],
        [0.0032, 0.0047, 0.0085, 0.0366, 0.3736, 0.0120, 0.0034, 0.0043, 0.0047,
         0.1019, 0.4381, 0.0091],
        [0.0029, 0.0049, 0.0062, 0.0258, 0.3805, 0.0117, 0.0032, 0.0044, 0.0041,
         0.0940, 0.4564, 0.0057],
        [0.0076, 0.0094, 0.0141, 0.0399, 0.3441, 0.0182, 0.0082, 0.0092, 0.0099,
         0.1087, 0.4171, 0.0134],
        [0.0407, 0.0362, 0.0475, 0.0670, 0.2260, 0.0447, 0.0442, 0.0371, 0.0423,
         0.1091, 0.2580, 0.0474],
        [0.0085, 0.0104, 0.0172, 0.0500, 0.3418, 0.0201, 0.0091, 0.0100, 0.0113,
         0.1152, 0.3883, 0.0180],
        [0.0031, 0.0051, 0.0074, 0.0331, 0.3738, 0.0131, 0.0033, 0.0046, 0.0043,
         0.1057, 0.4393, 0.0072],
        [0.0029, 0.0042, 0.0081, 0.0365, 0.3740, 0.0113, 0.0030, 0.0038, 0.0043,
         0.0997, 0.4435, 0.0087],
        [0.0019, 0.0035, 0.0046, 0.0233, 0.3867, 0.0092, 0.0021, 0.0031, 0.0028,
         0.0877, 0.4708, 0.0043],
        [0.0312, 0.0317, 0.0434, 0.0716, 0.2448, 0.0452, 0.0331, 0.0315, 0.0350,
         0.1218, 0.2672, 0.0434],
        [0.0479, 0.0421, 0.0537, 0.0706, 0.2039, 0.0498, 0.0519, 0.0432, 0.0487,
         0.1074, 0.2266, 0.0541],
        [0.0031, 0.0047, 0.0065, 0.0265, 0.3787, 0.0110, 0.0033, 0.0043, 0.0042,
         0.0924, 0.4592, 0.0060]] 
'''


def draw(sentence, A): 

    tag = sentence.split() 

    fig, ax = plt.subplots() 

    ax.set_xticks(np.arange(len(tag))) 
    ax.set_yticks(np.arange(len(tag))) 
    ax.set_xticklabels(tag, rotation=45, ha='left', rotation_mode='anchor', fontsize=11) 
    ax.set_yticklabels(tag, fontsize=11) 
    ax.xaxis.set_label_position('top') 
    im = ax.imshow(A, cmap='Reds', origin='upper') 

    plt.colorbar(im) 
    # for i in range(len(tag)): 
    #     for j in range(len(tag)): 
    #         ax.text(j, i, A[i][j], ha='center', va='center', color='black') 

    # default_font = {'weight': 'bold', 'size': 16}
    # plt.xlabel("m", default_font) 
    # plt.ylabel("n", default_font) 
    # plt.figure(figsize=(30, 30)) 
    plt.savefig('words_attention.pdf', dpi=600, bbox_inches='tight') 